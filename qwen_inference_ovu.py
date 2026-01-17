import os
import json
import torch
import torch.nn.functional as F
import glob
import time
import re
import uuid
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import DynamicCache
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from decord import cpu, VideoReader
import argparse

# Import KV cache compression utilities
from kvcache_utils import process_kv_cache

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning - adapted from eval_lvu_cache.py"""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        data_path: str,
        dataset: str,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path
        self.dataset = dataset

        list_data_dict = []

        if dataset == "videomme":
            data_list = load_dataset("lmms-lab/Video-MME")
            
            for item in data_list:
                video_ytid = item["url"].split("watch?v=")[-1]
                video_path = os.path.join(self.data_path, "video", f"{video_ytid}.mp4")
                for fmt in self.video_formats:
                    temp_path = os.path.join(self.data_path, "video", f"{video_ytid}{fmt}")
                    if os.path.exists(temp_path):
                        video_path = temp_path
                        break

                subtitle_path = os.path.join(
                    self.data_path, "subtitle", f"{video_ytid}.srt"
                )
                
                list_data_dict.append(
                    {
                        "questions": item["question"],
                        "video": video_path,
                        "subtitle": subtitle_path,
                        "video_name": video_ytid,
                        "answer": item["answer"],
                        "duration": item["duration"],
                        "task_type": item["task_type"],
                        "choices": item["options"],
                    }
                )

        elif "mlvu" in dataset or "sample" in dataset:

            json_name = "json" # Full

            json_folder_path = os.path.join(data_path, json_name)
            json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
            data_list = {}

            for json_file in json_files:
                task_name = re.sub(r'^\d+_(.+)\.json$', r'\1', json_file)            
                # Create tuple with required format
                data_tuple = (
                    f"{json_name}/{json_file}",  # json file path
                    f"video/{json_file.replace('.json', '')}", # video folder path
                    "video"  # constant value
                )
                data_list[task_name] = data_tuple
            
            for k, v in data_list.items():
                with open(os.path.join(data_path, v[0]), "r") as f:
                    json_data = json.load(f)
                for data in json_data:
                    question, answer = self.qa_template(data)
                    
                    list_data_dict.append(
                        {
                            "task_type": k,
                            "video": os.path.join(self.data_path, v[1], data["video"]),
                            "video_name": data["video"],
                            "questions": data["question"],
                            "prompt": question,
                            "answer": answer,
                            "duration": data["duration"],
                            "choices": data["candidates"]
                        }
                    )
            
        elif "lvb" in dataset:

            json_name = "wo_subtitle"
            
            json_folder_path = os.path.join(data_path, json_name)
            json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
            data_list = {}

            for json_file in json_files:
                task_name = re.sub(r'^\d+_(.+)\.json$', r'\1', json_file)            
                # Create tuple with required format
                data_tuple = (
                    f"{json_name}/{json_file}",  # json file path
                    f"video/{json_file.replace('.json', '')}", # video folder path
                    "video"  # constant value
                )
                data_list[task_name] = data_tuple
            
            for k, v in data_list.items():
                with open(os.path.join(data_path, v[0]), "r") as f:
                    json_data = json.load(f)
                for data in json_data:
                    question, _ = self.qa_template(data, abcd=False)
                    answer = data["answer"]
                    
                    list_data_dict.append(
                        {
                            "task_type": k[:-5],
                            "video": os.path.join(self.data_path, v[-1], data["video"]),
                            "video_name": data["video"],
                            "questions": data["question"],
                            "prompt": question,
                            "answer": answer,
                            "duration": data["duration"],
                            "choices": data["candidates"]
                        }
                    )

        elif dataset == "egoschema":
            answer_list = json.load(open(os.path.join(data_path, "subset_answers.json"), "r"))
            questions_list = json.load(open(os.path.join(data_path, "questions.json"), "r"))

            questions_list_org = {}

            for q in questions_list:
                questions_list_org[q["q_uid"]] = q

            for key, answer in answer_list.items():
                data = questions_list_org[key]
                
                question = data["question"]
                a0 = data["option 0"]
                a1 = data["option 1"]
                a2 = data["option 2"]
                a3 = data["option 3"]
                a4 = data["option 4"]
                prompt = f"Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nRespond with only the letter (A, B, C, D or E) of the correct option."
                options = f"0.{a0}, 1.{a1}, 2.{a2}, 3.{a3}, 4.{a4}"
                
                list_data_dict.append(
                    {   
                        "video": os.path.join(self.data_path, "video", f"{key}.mp4"),
                        "video_name": key,
                        "questions": question,
                        "prompt": prompt,
                        "answer": {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}.get(answer),
                        "choices": options,
                        "duration": None,
                        "task_type": None,
                    }
                )
        else:
            raise NotImplementedError("No dataset available (please choose {videomme, mlvu, egoschema, lvb})")
        
        self.data = list_data_dict
    
    def qa_template(self, data, abcd=True):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        if abcd:
            question += (
                "Respond with only the letter (A, B, C or D) of the correct option.\n"
            )
        else:
            question += (
                "Please answer with the letter for the correct option.\n"
            )
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


class OfflineVideoEval:
    """
    Offline Video Understanding Evaluation
    """
    
    def __init__(self, model_path, max_frames_num=32, block_size=-1, compress_frame_num=0, 
                 compression_method="uniform", tar_ratio=0.5, query_ratio=0.25, adaptive_pooling=False, 
                 load_dumped=False, input_compression="none", per_frame=False, verbose=False
                 ):
        """
        Initialize OfflineVideoEval class for Qwen2VL inference on video benchmarks
        
        Args:
            model_path: Path to the Qwen2VL model
            max_frames_num: Maximum number of frames to sample from video
            block_size: Size of blocks for block processing (-1 for no blocking)
            compress_frame_num: Number of frames to compress in kv cache
            compression_method: Method for KV cache compression
            tar_ratio: Ratio for tar vs other methods
            query_ratio: Ratio of query frames for tar method
            adaptive_pooling: Whether to use adaptive pooling
            load_dumped: Whether to load dumped preprocessed inputs if available
            input_compression: Type of input compression ("none", "entropy", "similarity")
            per_frame: Whether to select complete frames (for val_norm method)
        """
        self.model_path = model_path
        self.max_frames_num = max_frames_num
        self.block_size = block_size
        self.compress_frame_num = compress_frame_num
        self.compression_method = compression_method
        self.tar_ratio = tar_ratio
        self.query_ratio = query_ratio
        self.adaptive_pooling = adaptive_pooling
        self.load_dumped = load_dumped
        self.input_compression = input_compression
        self.per_frame = per_frame
        self.model = None
        self.processor = None
        self.verbose = verbose
        
        # For entropy-based compression: store previous entropy scores
        self.prev_entropy_scores = None
        self.prev_vision_start_idx = None
        
        # For similarity-based compression: store previous vision embeddings
        self.prev_vision_embeds = None
        
        # Initialize model
        self._initialize_model()
    
    def _print(self, message):
        if self.verbose:
            print(f">>> {message}", flush=True)

    def _initialize_model(self):
        """Initialize the Qwen2VL model and processor"""
        self._print("Loading Qwen2VL model...")
        if "2.5-vl" in self.model_path.lower():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                dtype="auto", 
                device_map="auto", 
                attn_implementation="flash_attention_2"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                dtype="auto", 
                device_map="auto", 
                attn_implementation="flash_attention_2"
            )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._print("Model loaded successfully!")
    
    def load_dataset(self, dataset_name, data_path):
        """
        Load dataset using EvalDataset class
        
        Args:
            dataset_name: Name of dataset ('videomme', 'mlvu', 'lvb', 'egoschema')
            data_path: Path to dataset
            
        Returns:
            EvalDataset instance
        """
        # Set default data paths if not provided
        if data_path is None:
            if "mlvu" in dataset_name:
                #data_path = "/data/ms/hf_cache/MLVU"
                data_path = "/video/MVLU/MLVU"
                #data_path = "/workspace/InfiniPot-V/MLVU"
            elif "ego" in dataset_name:
                data_path = "/data/ms/hf_cache/egoschema"
            elif "mme" in dataset_name:
                data_path = "/data/ms/hf_cache/videomme"
            elif "lvb" in dataset_name:
                data_path = "/data/ms/hf_cache/longvideobench"
            elif "sample" in dataset_name:
                data_path = "sample"
            else:
                raise ValueError(f"Please provide data_path for dataset: {dataset_name}")
        
        self._print(f"Loading dataset: {dataset_name} from {data_path}")
        dataset = EvalDataset(data_path=data_path, dataset=dataset_name)
        self._print(f"Dataset loaded: {len(dataset)} samples")
        
        return dataset
    
    def prepare_video_input(self, video_path, question_text, is_first_sample=False):
        """
        Prepare video input for Qwen2VL model
        
        Args:
            video_path: Path to video file
            question_text: Question text to ask about the video
            is_first_sample: Whether this is the first sample (for logging)
            
        Returns:
            Preprocessed inputs for the model
        """
        
        # Prepare messages for video input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_frames": self.max_frames_num,
                        "max_pixels": 128*28*28,
                    },
                    {"type": "text", "text": question_text},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate dump path for preprocessed inputs
        # Handle different video formats
        video_formats = [".mp4", ".avi", ".mov", ".mkv"]
        dump_path = video_path.replace("/MLVU/video/", "/MLVU/video_sampled_qwen_768/")
        
        for fmt in video_formats:
            if dump_path.endswith(fmt):
                dump_path = dump_path.replace(fmt, ".pt")
                break
        
        # Try to load dumped inputs if option is enabled
        if self.load_dumped and os.path.exists(dump_path):
            if is_first_sample:
                self._print("Loading dumped preprocessed inputs...")
            cur_time = time.time()
            inputs = torch.load(dump_path)
            inputs = inputs.to(self.model.device)
            load_time = time.time() - cur_time
            if is_first_sample:
                self._print(f"Loaded dumped inputs in {load_time:.4f}s")
        else:
            if self.load_dumped and is_first_sample:
                self._print("No dumped file found! Processing video...")
            elif is_first_sample:
                self._print("Processing video from scratch...")
                
            # Process vision info and prepare inputs
            _, video = process_vision_info(messages)

            cur_time = time.time()
            inputs = self.processor(
                text=[text],
                images=None,
                videos=video,
                padding=True,
                return_tensors="pt",
            )
            process_time = time.time() - cur_time
            if is_first_sample:
                self._print(f"Processed video in {process_time:.4f}s")
            
            input_ids = inputs["input_ids"]
            if self.load_dumped:    
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].half()
                inputs = inputs.to("cpu")
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                torch.save(inputs, dump_path)

            inputs = inputs.to(self.model.device)
            
        # Prepare input_ids with video tokens
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        video_length = int(inputs["pixel_values_videos"].shape[0] // 4) # patch merger 
        video_pad_tokens = torch.full((video_length,), self.model.config.video_token_id, dtype=torch.long).unsqueeze(0).to(self.model.device)
        vision_start_idx = (input_ids == self.model.config.vision_start_token_id).nonzero(as_tuple=True)[-1]
        vision_end_idx = (input_ids == self.model.config.vision_end_token_id).nonzero(as_tuple=True)[-1]
        inputs["input_ids"] = torch.cat([input_ids[:,:vision_start_idx + 1], video_pad_tokens, input_ids[:,vision_end_idx:]], dim=-1)
        inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape).to(self.model.device)
        
        return inputs
    
    def format_question(self, data_item, dataset_name):
        """
        Format question based on dataset type
        
        Args:
            data_item: Single data item from dataset
            dataset_name: Name of the dataset
            
        Returns:
            Formatted question string
        """
        if dataset_name == "videomme":
            q = data_item["questions"]
            ops = data_item["choices"]
            instruct = f"Question: {q}\n"
            instruct += "Options:\n"
            for op in ops:
                instruct += f"{op}\n"
            instruct += (
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            )
            question_text = instruct.rstrip()
            
        elif "mlvu" in dataset_name or dataset_name == "egoschema" or "lvb" in dataset_name:
            question_text = data_item["prompt"]
        
        elif "sample" in dataset_name:
            question_text = data_item["questions"] + (" Respond with which option is the correct answer and explain why it is the correct answer.")
        
        else:
            raise NotImplementedError(f"Question formatting not implemented for dataset: {dataset_name}")
        
        return question_text
    
    def extract_answer(self, response, dataset_name):
        """
        Extract answer from model response based on dataset format
        
        Args:
            response: Raw model response
            dataset_name: Name of the dataset
            
        Returns:
            Extracted answer
        """
        response = response.replace("Answer", "")
        
        if "ego" in dataset_name or "lvb" in dataset_name:
            letters = ["A", "B", "C", "D", "E"]
            pred_answer = re.findall("[\(\ ]*[A-E][\)\ ]*", response)
        else:
            letters = ["A", "B", "C", "D"]
            pred_answer = re.findall("[\(\ \[]*([A-D])[\)\.\ \]]*", response)
        
        if len(pred_answer) >= 1:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            
        if pred_answer in letters:
            pred_idx = letters.index(pred_answer)
            pred = letters[pred_idx]
        else:
            print(">>> No alphabet found!!!", "pred_answer: ", pred_answer, " response: ", response, flush=True)
            pred_idx = 2  # Default to C
            pred = letters[pred_idx]
        
        return pred
    
    def generate(self, inputs):
        """
        Standard generation using model.generate (original approach)
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Generated response text
        """
        with torch.no_grad():
            inputs.pop("second_per_grid_ts", None) ##qwen2 전용
            generated_ids = self.model.generate(**inputs, max_new_tokens=5)
            
            # Decode output (same as original code)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
        
        return response
    
    def _visual_encoding(self, inputs, input_ids, system_size, token_per_frame):
        """
        Perform visual encoding and prepare inputs_embeds with position information
        
        Args:
            inputs: Preprocessed inputs containing pixel_values_videos and video_grid_thw
            input_ids: Input token IDs
            system_size: Size of system tokens
            token_per_frame: Number of tokens per frame
            
        Returns:
            tuple: (inputs_embeds, position_ids_full)
        """
        # Calculate position_ids_full and height_width
        position_ids_full, _ = self.model.model.get_rope_index(input_ids, video_grid_thw=inputs["video_grid_thw"])
        self.model.config.height_width = position_ids_full[1,:,system_size:system_size+token_per_frame].max() - position_ids_full[1,:,system_size:system_size+token_per_frame].min() + 1
        
        # Get token embeddings
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        pixel_values_videos = inputs["pixel_values_videos"].type(self.model.dtype)
        
        # Visual Encoding
        with torch.inference_mode():
            video_embeds = self.model.get_video_features(
                pixel_values_videos,
                video_grid_thw=inputs["video_grid_thw"]
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        
        # Replace video tokens with visual embeddings
        video_mask = input_ids == self.model.config.video_token_id
        inputs_embeds[video_mask] = video_embeds
        
        return inputs_embeds, position_ids_full
    
    def _block_wise_prefill(self, inputs, input_ids, position_ids_full, past_key_values, system_size, inst_size, token_per_frame, vision_length):
        """
        Perform block-wise prefill processing
        
        Args:
            inputs: Original inputs containing pixel_values_videos and video_grid_thw
            input_ids: Input token IDs
            position_ids_full: Full position IDs
            past_key_values: KV cache to update
            system_size: Size of system tokens
            inst_size: Size of instruction tokens
            token_per_frame: Number of tokens per frame
            vision_length: Total vision token length
            
        Returns:
            Updated past_key_values after prefill
        """
        past_seen_tokens = 0
        cur_pos = system_size
        
        # Get full inputs_embeds with visual encoding first
        inputs_embeds, _ = self._visual_encoding(inputs, input_ids, system_size, token_per_frame)
        
        while cur_pos < vision_length:
            start_idx = cur_pos
            
            # Calculate visual cache length (for compression)
            vis_cache_length = (self.compress_frame_num * token_per_frame) if start_idx > system_size else 0
            end_idx = cur_pos + self.block_size * token_per_frame - vis_cache_length
            
            # Determine block type
            is_first_block = start_idx == system_size
            is_last_block = True if (end_idx - system_size) >= vision_length else False
            
            # Prepare block inputs and extract video embeddings for similarity calculation
            if is_first_block:
                # First block: include system tokens + vision tokens
                block_inputs_embeds = inputs_embeds[:, :end_idx]
                position_ids = position_ids_full[:, :, :end_idx]
                # For similarity calculation: exclude system tokens
                current_video_embeds = inputs_embeds[:, system_size:end_idx]
            elif is_last_block:
                # Last block: vision tokens (instruction will be added in generation stage)
                block_inputs_embeds = inputs_embeds[:, start_idx:system_size + vision_length]
                position_ids = position_ids_full[:, :, start_idx:system_size + vision_length]
                # For similarity calculation: all tokens are video tokens
                current_video_embeds = inputs_embeds[:, start_idx:system_size + vision_length]
            else:
                # Middle block: only vision tokens
                block_inputs_embeds = inputs_embeds[:, start_idx:end_idx]
                position_ids = position_ids_full[:, :, start_idx:end_idx]
                # For similarity calculation: all tokens are video tokens
                current_video_embeds = inputs_embeds[:, start_idx:end_idx]
            
            # Run model on current block
            with torch.inference_mode():
                outputs = self.model(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=block_inputs_embeds,
                    use_cache=True
                )
            
            # Update cache and counters
            past_key_values = outputs[1]
            past_seen_tokens += block_inputs_embeds.shape[1]
            
            # KV cache compression for non-last blocks
            if not is_last_block:
                # DynamicBP - Dynamic compression frame calculation
                assert (vision_length - (past_seen_tokens - system_size)) % token_per_frame == 0, "left length should be divided by token_per_frame number."

                res_frame_num = (vision_length - (past_seen_tokens - system_size)) // token_per_frame
                
                
                if (self.compress_frame_num + res_frame_num) >= self.block_size:
                    compress_frame_num = self.compress_frame_num
                else:
                    compress_frame_num = self.block_size - res_frame_num
                
                # Use traditional KV cache compression
                past_key_values, _ = process_kv_cache(
                    past_key_values=past_key_values,
                    model=self.model,
                    system_size=system_size,
                    inst_size=inst_size,
                    token_per_frame=token_per_frame,
                    compress_frame_num=compress_frame_num,
                    method=self.compression_method,
                    tar_ratio=self.tar_ratio,
                    query_ratio=self.query_ratio,
                    adaptive_pooling=self.adaptive_pooling,
                    is_first_block=is_first_block,
                    is_last_block=is_last_block,
                    per_frame=self.per_frame
                )
            
            # Move to next block
            cur_pos = end_idx
        
        return past_key_values
    
    def _generation_stage(self, inputs, past_key_values, position_ids_full, inst_size):
        """
        Generation stage: generate tokens autoregressively after prefill
        
        Args:
            inputs: Original inputs (to get instruction tokens)
            past_key_values: KV cache after prefill
            position_ids_full: Position IDs from prefill
            inst_size: Size of instruction tokens
            
        Returns:
            Generated response text
        """
        # Get instruction tokens
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask_indices = (input_ids[0,:] == video_token_id).nonzero(as_tuple=True)[0]
        
        # Extract instruction part after vision tokens
        inst_start = mask_indices[-1] + 1  # after vision_end token
        post_input_ids = input_ids[:, inst_start:]
        input_len = post_input_ids.shape[-1]
        
        # Initialize for generation
        input_ids_save = post_input_ids
        input_ids_current = post_input_ids
        
        # Calculate 3D position ids starting from prefill end
        position_ids = torch.arange(input_len, device=input_ids_current.device).expand(input_ids_current.shape[0], -1)
        position_ids = position_ids.add(position_ids_full[0, 0, -1]) + 1
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        # Generation loop
        generation_finished = False
        max_gen_tokens = 300  # Prevent infinite loop
        gen_token_count = 0
        
        while not generation_finished and gen_token_count < max_gen_tokens:
            with torch.no_grad():
                outputs = self.model(
                    input_ids_current, 
                    past_key_values=past_key_values, 
                    position_ids=position_ids
                )
            
            final_logits = outputs[0]
            past_key_values = outputs[1]
            next_token = final_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # Update sequences
            input_ids_save = torch.cat([input_ids_save, next_token], dim=-1)
            input_ids_current = next_token
            position_ids = position_ids[:, :, -1:] + 1
            gen_token_count += 1
            
            # Check for EOS token
            if hasattr(self.processor.tokenizer, 'eos_token_id'):
                eos_token_id = self.processor.tokenizer.eos_token_id
            else:
                eos_token_id = 151645  # fallback
            
            if next_token[0, -1] == eos_token_id:
                generation_finished = True
        
        # Extract generated part (excluding instruction)
        output_ids = input_ids_save[:, input_len:]
        
        # Decode generated tokens
        response = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return response
    
    def block_process(self, inputs):
        """
        Block processing approach with prefill and generation stages
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Generated response text
        """
        # Reset compression caches for new video
        if self.input_compression == "entropy":
            self.prev_entropy_scores = None
            self.prev_vision_start_idx = None
        elif self.input_compression == "similarity":
            self.prev_vision_embeds = None
        
        # Extract video tokens and calculate dimensions
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        frame_number = inputs["video_grid_thw"][0,0]
        
        # Calculate token positions and dimensions
        mask_indices = (input_ids[0,:] == video_token_id).nonzero(as_tuple=True)[0]
        system_size = input_ids[0,:mask_indices[0] - 1].shape[-1] + 1 # vision_start
        inst_size = input_ids[0,mask_indices[-1] + 2:].shape[-1] + 1 # including vision end token
        token_per_frame = int((input_ids == video_token_id).sum() // frame_number)
        
        # Initialize past_key_values for caching
        past_key_values = DynamicCache()
        
        # Calculate position_ids_full
        position_ids_full, _ = self.model.model.get_rope_index(input_ids, video_grid_thw=inputs["video_grid_thw"])
        self.model.config.height_width = position_ids_full[1,:,system_size:system_size+token_per_frame].max() - position_ids_full[1,:,system_size:system_size+token_per_frame].min() + 1
        
        token_length = input_ids.shape[1]
        vision_length = token_length - system_size - inst_size
        
        # Step 1: Block-wise Prefill
        past_key_values = self._block_wise_prefill(
            inputs, input_ids, position_ids_full, past_key_values,
            system_size, inst_size, token_per_frame, vision_length
        )
        
        # Step 2: Generation Stage
        response = self._generation_stage(inputs, past_key_values, position_ids_full, inst_size)
        
        return response
    
    def evaluate_dataset(self, dataset, dataset_name, output_dir, exp_tag):
        """
        Run evaluation on the loaded dataset
        
        Args:
            dataset: EvalDataset instance
            dataset_name: Name of the dataset
            output_dir: Directory to save results
            exp_tag: Experiment tag for output naming
            
        Returns:
            Dictionary containing results and accuracy statistics
        """
        results = []
        correct_count = 0
        total_count = 0
        
        # Task-specific accuracy tracking
        task_type_stats = {}
        duration_stats = {"short": {"correct": 0, "total": 0}, 
                         "medium": {"correct": 0, "total": 0}, 
                         "long": {"correct": 0, "total": 0}}
        
        print(f"Starting evaluation on {dataset_name} with {len(dataset)} samples...")

        for idx, data_item in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}", dynamic_ncols=True)):
            video_path = data_item["video"]
            video_name = data_item["video_name"]
            
            # For now, decode every video (can be optimized later for VideoMME)
            do_decode = True
            
            # # Skip if video file doesn't exist
            # if not os.path.exists(video_path):
            #     print(f"Warning: Video file not found: {video_path}")
            #     continue
            
            # Format question based on dataset
            question_text = self.format_question(data_item, dataset_name)
            
            # Prepare video input
            is_first_sample = (idx == 0)
            try:
                inputs = self.prepare_video_input(video_path, question_text, is_first_sample)

            except Exception as e:
                print(f"Error preparing video input: {e}")
                continue
            
            # Choose inference method based on settings
            start_time = time.time()
            frame_count = inputs["video_grid_thw"][0,0].item()
            use_block_processing = (self.block_size > 0 and frame_count > self.block_size and 
                                    (self.compress_frame_num > 0 or self.input_compression != "none"))
            
            if is_first_sample:
                print(f"Frame count: {frame_count}, Block size: {self.block_size}, Compress frames: {self.compress_frame_num}")
                print(f"Using {'Block Processing' if use_block_processing else 'Standard Generation'}")
                if use_block_processing and self.compress_frame_num > 0:
                    compression_display = {
                        "none": self.compression_method,
                        "entropy": "Entropy-based",
                        "similarity": "Similarity-based"
                    }
                    print(f"Compression method: {compression_display[self.input_compression]}")
            
            if use_block_processing:
                response = self.block_process(inputs)
            else:
                response = self.generate(inputs)
            inference_time = time.time() - start_time
            
            # Extract answer from response
            pred_answer = self.extract_answer(response, dataset_name)
            ground_truth = data_item["answer"]
            
            # Check correctness
            is_correct = pred_answer == ground_truth
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Track task-specific accuracy
            task_type = data_item.get("task_type", "unknown")
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {"correct": 0, "total": 0}
            task_type_stats[task_type]["total"] += 1
            if is_correct:
                task_type_stats[task_type]["correct"] += 1
            
            # Track duration-specific accuracy (for VideoMME)
            if dataset_name == "videomme" and "duration" in data_item:
                duration = data_item["duration"]
                if duration in duration_stats:
                    duration_stats[duration]["total"] += 1
                    if is_correct:
                        duration_stats[duration]["correct"] += 1
            
            # Store result
            result = {
                "video_name": video_name,
                "question": data_item["questions"],
                "ground_truth": ground_truth,
                "predicted_answer": pred_answer,
                "model_response": response,
                "is_correct": is_correct,
                "task_type": task_type,
                "inference_time": inference_time,
                "video_path": video_path
            }
            
            if dataset_name == "videomme":
                result["duration"] = data_item.get("duration")
                result["choices"] = data_item.get("choices")
            
            results.append(result)
            
            # Clear GPU cache periodically
            if idx % 100 == 0:
                torch.cuda.empty_cache()
                
        
        # Calculate accuracy statistics
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Task-specific accuracy
        task_accuracy = {}
        for task_type, stats in task_type_stats.items():
            task_accuracy[task_type] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Duration-specific accuracy (for VideoMME)
        duration_accuracy = {}
        if dataset_name == "videomme":
            for duration, stats in duration_stats.items():
                duration_accuracy[duration] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Prepare type-wise accuracy summary
        type_wise_accuracy = {}
        for task_type, stats in task_type_stats.items():
            type_wise_accuracy[task_type] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
                "percentage": f"{(stats['correct'] / stats['total'] * 100):.2f}%" if stats["total"] > 0 else "0.00%"
            }
        
        # Prepare final results
        final_results = {
            "results": results,
            "accuracy_statistics": {
                "overall_accuracy": overall_accuracy,
                "total_questions": total_count,
                "total_correct": correct_count,
                "task_type_accuracy": task_accuracy,
                "task_type_details": task_type_stats,
                "type_wise_summary": type_wise_accuracy
            },
            "experiment_config": {
                "dataset": dataset_name,
                "model_path": self.model_path,
                "max_frames": self.max_frames_num,
                "block_size": self.block_size,
                "compress_frame_num": self.compress_frame_num,
                "compression_method": self.compression_method,
                "load_dumped": self.load_dumped,
                "input_compression": self.input_compression,
                "per_frame": self.per_frame
            }
        }
        
        if dataset_name == "videomme":
            final_results["accuracy_statistics"]["duration_accuracy"] = duration_accuracy
            final_results["accuracy_statistics"]["duration_details"] = duration_stats
            
            # Add duration-wise summary for VideoMME
            duration_wise_summary = {}
            for duration, stats in duration_stats.items():
                duration_wise_summary[duration] = {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "percentage": f"{(stats['correct'] / stats['total'] * 100):.2f}%" if stats["total"] > 0 else "0.00%"
                }
            final_results["accuracy_statistics"]["duration_wise_summary"] = duration_wise_summary
        
        # Save results
        output_path = os.path.join(output_dir, self.model_path.split("/")[-1], exp_tag)
        os.makedirs(output_path, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_path, "results.json"), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save simple accuracy results (compatible with eval_lvu_cache.py format)
        if dataset_name == "videomme":
            simple_results = {
                "average_acc": overall_accuracy,
                **duration_accuracy
            }
        elif "mlvu" in dataset_name or "lvb" in dataset_name:
            simple_results = {task_type: acc for task_type, acc in task_accuracy.items()}
            simple_results["Acc"] = sum(task_accuracy.values()) / len(task_accuracy) if task_accuracy else 0
        else:  # egoschema
            simple_results = {"acc": overall_accuracy}
        
        with open(os.path.join(output_path, "accuracy.json"), 'w') as f:
            json.dump(simple_results, f, indent=2)
        
        # Print summary
        if dataset_name == "sample":
            print(f"\n=== Sample Generation Results ===")
            print(f">>> QUESTION: {question_text}")
            print(f">>> RESPONSE: {response}")
        else:
            print(f"\n=== EVALUATION SUMMARY ===")
            print(f"Dataset: {dataset_name}")
            print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct_count}/{total_count})")
            
            if task_accuracy:
                print(f"\nAccuracy by Task Type:")
                for task_type, acc in task_accuracy.items():
                    stats = task_type_stats[task_type]
                    print(f"  {task_type}: {acc:.4f} ({stats['correct']}/{stats['total']})")
            
            if dataset_name == "videomme" and duration_accuracy:
                print(f"\nAccuracy by Duration:")
                for duration, acc in duration_accuracy.items():
                    stats = duration_stats[duration]
                    print(f"  {duration}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
            print(f"\nResults saved to: {output_path}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description="Offline Video Understanding Evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Path to the Qwen2VL model")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["videomme", "mlvu", "lvb", "egoschema", "sample"],
                        help="Dataset to evaluate on")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, default="results/ovu",
                        help="Output directory for results")
    parser.add_argument("--exp_tag", type=str, default="baseline",
                        help="Experiment tag for output naming")
    
    # Video processing parameters
    parser.add_argument("--max_frames_num", type=int, default=32,
                        help="Maximum number of frames to sample")
    parser.add_argument("--load_dumped", action="store_true",
                        help="Load dumped preprocessed inputs if available")
    
    # Block processing parameters
    parser.add_argument("--use_block_processing", action="store_true",
                        help="Use block processing instead of standard generation")
    parser.add_argument("--block_size", type=int, default=-1,
                        help="Block size for block processing (-1 for no blocking)")
    parser.add_argument("--compress_frame_num", type=int, default=0,
                        help="Number of frames to compress in kv cache")
    parser.add_argument("--compression_method", type=str, default="uniform",
                        help="Method for KV cache compression")
    parser.add_argument("--tar_ratio", type=float, default=0.5,
                        help="Ratio for tar vs other methods")
    parser.add_argument("--query_ratio", type=float, default=0.25,
                        help="Ratio of query frames for tar method")
    parser.add_argument("--adaptive_pooling", action="store_true",
                        help="Use adaptive pooling for KV cache compression")
    parser.add_argument("--input_compression", type=str, default="none",
                        choices=["none", "entropy", "similarity"],
                        help="Type of input compression: none (traditional), entropy, or similarity")
    parser.add_argument("--per_frame", action="store_true",
                        help="Select complete frames instead of individual tokens (for val_norm method)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = OfflineVideoEval(
        model_path=args.model_path,
        max_frames_num=args.max_frames_num,
        block_size=args.block_size,
        compress_frame_num=args.compress_frame_num,
        compression_method=args.compression_method,
        tar_ratio=args.tar_ratio,
        query_ratio=args.query_ratio,
        adaptive_pooling=args.adaptive_pooling,
        load_dumped=args.load_dumped,
        input_compression=args.input_compression,
        per_frame=args.per_frame,
        verbose = args.verbose
    )
    
    # Load dataset
    dataset = evaluator.load_dataset(args.dataset, args.data_path)
    
    if args.verbose:
        print(f"Dataset: {args.dataset}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Model: {args.model_path}")
        print(f"Load dumped: {args.load_dumped}")
        print(f"Block processing: {args.use_block_processing}")
        if args.use_block_processing:
            print(f"Block size: {args.block_size}")
            print(f"Compress frames: {args.compress_frame_num}")
            compression_display = {
                "none": args.compression_method,
                "entropy": "Entropy-based",
                "similarity": "Similarity-based"
            }
            print(f"Compression method: {compression_display[args.input_compression]}")
        print(f"\n=== Starting Evaluation ===")
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        exp_tag=args.exp_tag
    )
    if args.dataset != "sample":
        print(f"\n=== Evaluation Complete ===")
        print(f"Overall Accuracy: {results['accuracy_statistics']['overall_accuracy']:.4f}")
        print(f"Total Questions: {results['accuracy_statistics']['total_questions']}")
        print(f"Total Correct: {results['accuracy_statistics']['total_correct']}")


if __name__ == "__main__":
    main() 