"""
Qwen2-VL Inference with Integrated Vision-LLM Block Processing

This module integrates vision encoding and LLM block processing:
- Instead of encoding all video frames at once, we process them block by block
- Each block: Vision Tower (partial frames) -> Vision Tokens -> LLM Forward -> KV Compression
"""

import os
import json
import torch
import time
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import DynamicCache
from qwen_vl_utils import process_vision_info
import argparse

# Import utilities
from kvcache_utils import process_kv_cache
from dataset_utils import EvalDataset, format_question, extract_answer, get_default_data_path


class OfflineVideoEval:
    """
    Offline Video Understanding Evaluation
    
    Features:
    - Vision encoding and LLM block processing are integrated
    - Each block processes: raw pixels -> vision tower -> vision tokens -> LLM -> KV compression
    """
    
    # Class constants
    MAX_GEN_TOKENS = 300
    DEFAULT_EOS_TOKEN_ID = 151645
    DEFAULT_MAX_PIXELS = 128 * 28 * 28
    VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    
    def __init__(self, model_path, max_frames_num=32, block_size=-1, compress_frame_num=0, 
                 compression_method="uniform", tar_ratio=0.5, query_ratio=0.25, adaptive_pooling=False, 
                 load_dumped=False, per_frame=False, verbose=False):
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
        self.per_frame = per_frame
        self.model = None
        self.processor = None
        self.verbose = verbose
        
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
        if data_path is None:
            data_path = get_default_data_path(dataset_name)
        
        self._print(f"Loading dataset: {dataset_name} from {data_path}")
        dataset = EvalDataset(data_path=data_path, dataset=dataset_name)
        self._print(f"Dataset loaded: {len(dataset)} samples")
        
        return dataset
    
    def _get_dump_path(self, video_path):
        """Generate dump path for preprocessed inputs"""
        dump_path = video_path.replace("/MLVU/video/", "/MLVU/video_sampled_qwen_768/")
        for fmt in self.VIDEO_FORMATS:
            if dump_path.endswith(fmt):
                return dump_path.replace(fmt, ".pt")
        return dump_path + ".pt"
    
    def _load_or_process_video(self, messages, text, dump_path, is_first_sample):
        """
        Load preprocessed video inputs or process from scratch
        
        Returns:
            Preprocessed inputs tensor
        """
        # Try loading cached inputs
        if self.load_dumped and os.path.exists(dump_path):
            if is_first_sample:
                self._print("Loading dumped preprocessed inputs...")
            start_time = time.time()
            inputs = torch.load(dump_path).to(self.model.device)
            if is_first_sample:
                self._print(f"Loaded dumped inputs in {time.time() - start_time:.4f}s")
            return inputs
        
        # Process video from scratch
        if is_first_sample:
            msg = "No dumped file found! Processing video..." if self.load_dumped else "Processing video from scratch..."
            self._print(msg)
        
        _, video = process_vision_info(messages)
        start_time = time.time()
        inputs = self.processor(
            text=[text],
            images=None,
            videos=video,
            padding=True,
            return_tensors="pt",
        )
        if is_first_sample:
            self._print(f"Processed video in {time.time() - start_time:.4f}s")
        
        # Save for future use if enabled
        if self.load_dumped:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].half()
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            torch.save(inputs.to("cpu"), dump_path)
        
        return inputs.to(self.model.device)
    
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
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_frames": self.max_frames_num, "max_pixels": self.DEFAULT_MAX_PIXELS},
                {"type": "text", "text": question_text},
            ],
        }]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Load or process video
        dump_path = self._get_dump_path(video_path)
        inputs = self._load_or_process_video(messages, text, dump_path, is_first_sample)
            
        # Prepare input_ids with video tokens
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        video_length = inputs["pixel_values_videos"].shape[0] // 4  # patch merger
        video_pad_tokens = torch.full((video_length,), self.model.config.video_token_id, dtype=torch.long, device=self.model.device).unsqueeze(0)
        
        vision_start_idx = (input_ids == self.model.config.vision_start_token_id).nonzero(as_tuple=True)[-1]
        vision_end_idx = (input_ids == self.model.config.vision_end_token_id).nonzero(as_tuple=True)[-1]
        
        inputs["input_ids"] = torch.cat([input_ids[:, :vision_start_idx + 1], video_pad_tokens, input_ids[:, vision_end_idx:]], dim=-1)
        inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, device=self.model.device)
        
        return inputs
    
    def generate(self, inputs):
        """
        Standard generation using model.generate (original approach)
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Generated response text
        """
        with torch.no_grad():
            inputs.pop("second_per_grid_ts", None) # for qwen2
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_GEN_TOKENS)
            
            # Decode output (same as original code)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
        
        return response
    
    def _visual_encoding_block(self, pixel_values_videos, video_grid_thw, frame_start, frame_end, total_frames):
        """
        Perform visual encoding for a specific block of frames
        
        This function enables block-wise vision encoding.
        Instead of encoding all frames at once, we encode only a subset of frames.
        
        Args:
            pixel_values_videos: Full pixel values tensor [total_patches, patch_dim]
            video_grid_thw: Original video grid info [1, 3] = [frame_num, H, W]
            frame_start: Starting frame index for this block
            frame_end: Ending frame index for this block (exclusive)
            total_frames: Total number of frames in the video
            
        Returns:
            video_embeds: Vision embeddings for this block of frames
        """
        # Calculate patches per frame
        # pixel_values_videos shape: [total_patches, patch_dim]
        # where total_patches = frame_num * H_patches * W_patches (before patch merger)
        total_patches = pixel_values_videos.shape[0]
        patches_per_frame = total_patches // total_frames
        
        # Extract pixel values for this block
        patch_start = frame_start * patches_per_frame
        patch_end = frame_end * patches_per_frame
        block_pixel_values = pixel_values_videos[patch_start:patch_end]
        
        # Create block-specific video_grid_thw
        block_frame_num = frame_end - frame_start
        block_video_grid_thw = video_grid_thw.clone()
        block_video_grid_thw[0, 0] = block_frame_num  # Update frame count
        
        # Convert to model dtype
        block_pixel_values = block_pixel_values.type(self.model.dtype)
        
        # Run vision tower on this block
        with torch.inference_mode():
            video_embeds = self.model.get_video_features(
                block_pixel_values,
                video_grid_thw=block_video_grid_thw
            )
            # get_video_features returns a list, concatenate
            video_embeds = torch.cat(video_embeds, dim=0)
        
        return video_embeds
    
    def _block_wise_prefill(self, inputs, input_ids, past_key_values, system_size, inst_size, 
                             token_per_frame, vision_length, total_frames):
        """
        Perform block-wise prefill with integrated vision encoding
        
        Features:
        - Vision encoding happens within the block loop
        - Each iteration: extract frame pixels -> vision tower -> LLM forward -> KV compression
        
        Args:
            inputs: Original inputs containing pixel_values_videos and video_grid_thw
            input_ids: Input token IDs
            past_key_values: KV cache to update
            system_size: Size of system tokens
            inst_size: Size of instruction tokens
            token_per_frame: Number of tokens per frame (after patch merger)
            vision_length: Total vision token length
            total_frames: Total number of frames in the video
            
        Returns:
            tuple: (past_key_values, position_ids_full) - Updated KV cache and position IDs for generation
        """
        cur_frame = 0  # Track by frame index instead of token position
        
        # Get system token embeddings (these are text embeddings, processed once)
        system_input_ids = input_ids[:, :system_size]
        system_embeds = self.model.model.get_input_embeddings()(system_input_ids)
        
        # Get pixel values and grid info
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]
        
        # Calculate position_ids for the full sequence (needed for 3D RoPE)
        position_ids_full, _ = self.model.model.get_rope_index(input_ids, video_grid_thw=video_grid_thw)
        
        # Store height_width for KV cache compression
        self.model.config.height_width = (
            position_ids_full[1, :, system_size:system_size + token_per_frame].max() - 
            position_ids_full[1, :, system_size:system_size + token_per_frame].min() + 1
        )
        
        # Calculate how many frames to process per block
        # Considering compression: actual new frames = block_size - compress_frame_num (except first block)
        
        while cur_frame < total_frames:
            # Calculate frame range for this block
            frame_start = cur_frame
            
            # Calculate how many frames can be processed in this block
            # For first block: full block_size frames
            # For subsequent blocks: block_size - compress_frame_num (since compress_frame_num frames are cached)
            if frame_start == 0:
                frames_in_block = min(self.block_size, total_frames)
            else:
                # Account for compressed frames in cache
                vis_cache_frames = self.compress_frame_num
                frames_in_block = min(self.block_size - vis_cache_frames, total_frames - frame_start)
            
            frame_end = frame_start + frames_in_block
            
            # Determine block type
            is_first_block = (frame_start == 0)
            is_last_block = (frame_end >= total_frames)
            
            # Step 1: Vision encoding for this block of frames
            block_video_embeds = self._visual_encoding_block(
                pixel_values_videos, video_grid_thw, 
                frame_start, frame_end, total_frames
            )
            block_video_embeds = block_video_embeds.to(system_embeds.device, system_embeds.dtype)
            
            # Step 2: Prepare inputs_embeds for this block
            if is_first_block:
                # First block: system tokens + vision tokens
                block_inputs_embeds = torch.cat([system_embeds, block_video_embeds.unsqueeze(0)], dim=1)
                # Calculate token range for position_ids
                token_start = 0
                token_end = system_size + frames_in_block * token_per_frame
            else:
                # Subsequent blocks: only vision tokens (system tokens are in cache)
                block_inputs_embeds = block_video_embeds.unsqueeze(0)
                # Token range for position_ids
                token_start = system_size + frame_start * token_per_frame
                token_end = system_size + frame_end * token_per_frame
            
            # Get position_ids for this block from the full position_ids
            position_ids = position_ids_full[:, :, token_start:token_end]
            
            # Step 3: LLM forward for this block
            with torch.inference_mode():
                outputs = self.model(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=block_inputs_embeds,
                    use_cache=True
                )
            
            # Update cache
            past_key_values = outputs[1]
            
            # Step 4: KV cache compression for non-last blocks
            if not is_last_block:
                # Calculate remaining frames
                res_frame_num = total_frames - frame_end
                
                # Dynamic compression frame calculation
                if (self.compress_frame_num + res_frame_num) >= self.block_size:
                    compress_frame_num = self.compress_frame_num
                else:
                    compress_frame_num = self.block_size - res_frame_num
                
                # Apply KV cache compression
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
            cur_frame = frame_end
        
        # Return the last position for generation stage
        return past_key_values, position_ids_full
    
    def _generation_stage(self, inputs, past_key_values, position_ids_full):
        """
        Generation stage: generate tokens autoregressively after prefill
        
        Args:
            inputs: Original inputs (to get instruction tokens)
            past_key_values: KV cache after prefill
            position_ids_full: Position IDs from prefill
            
        Returns:
            Generated response text
        """
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        
        # Extract instruction part after vision tokens
        inst_start = mask_indices[-1] + 1
        post_input_ids = input_ids[:, inst_start:]
        input_len = post_input_ids.shape[-1]
        
        # Initialize for generation
        input_ids_save = post_input_ids
        input_ids_current = post_input_ids
        
        # Calculate 3D position ids starting from prefill end
        position_ids = torch.arange(input_len, device=input_ids_current.device).expand(input_ids_current.shape[0], -1)
        position_ids = (position_ids + position_ids_full[0, 0, -1] + 1).unsqueeze(0).expand(3, -1, -1)
        
        # Get EOS token ID once
        eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id', self.DEFAULT_EOS_TOKEN_ID)
        
        # Generation loop
        for _ in range(self.MAX_GEN_TOKENS):
            with torch.no_grad():
                outputs = self.model(input_ids_current, past_key_values=past_key_values, position_ids=position_ids)
            
            past_key_values = outputs[1]
            next_token = outputs[0][:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            input_ids_save = torch.cat([input_ids_save, next_token], dim=-1)
            input_ids_current = next_token
            position_ids = position_ids[:, :, -1:] + 1
            
            if next_token[0, -1] == eos_token_id:
                break
        
        # Decode generated tokens (excluding instruction)
        output_ids = input_ids_save[:, input_len:]
        return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def block_process(self, inputs):
        """
        Block processing approach with integrated vision-LLM processing
        
        This is the main entry point for block processing.
        Vision encoding happens inside the block loop.
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Generated response text
        """
        # Extract video tokens and calculate dimensions
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        frame_number = inputs["video_grid_thw"][0, 0].item()
        
        # Calculate token positions and dimensions
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        system_size = input_ids[0, :mask_indices[0] - 1].shape[-1] + 1  # vision_start
        inst_size = input_ids[0, mask_indices[-1] + 2:].shape[-1] + 1  # including vision end token
        token_per_frame = int((input_ids == video_token_id).sum() // frame_number)
        
        # Calculate video-related dimensions
        total_frames = frame_number
        
        token_length = input_ids.shape[1]
        vision_length = token_length - system_size - inst_size
        
        # Initialize past_key_values for caching
        past_key_values = DynamicCache()
        
        self._print(f"Block Processing: {total_frames} frames, {token_per_frame} tokens/frame")
        self._print(f"Block size: {self.block_size}, Compress frames: {self.compress_frame_num}")
        
        # Step 1: Block-wise Prefill with integrated vision encoding
        past_key_values, position_ids_full = self._block_wise_prefill(
            inputs, input_ids, past_key_values,
            system_size, inst_size, token_per_frame, vision_length,
            total_frames
        )
        
        # Step 2: Generation Stage
        response = self._generation_stage(inputs, past_key_values, position_ids_full)
        
        return response
    
    def _compute_accuracy(self, stats_dict):
        """Compute accuracy from stats dict with correct/total counts"""
        return {k: v["correct"] / v["total"] if v["total"] > 0 else 0 for k, v in stats_dict.items()}
    
    def _create_summary(self, stats_dict):
        """Create detailed summary with accuracy, counts, and percentage"""
        return {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                "correct": v["correct"],
                "total": v["total"],
                "percentage": f"{v['correct'] / v['total'] * 100:.2f}%" if v["total"] > 0 else "0.00%"
            }
            for k, v in stats_dict.items()
        }
    
    def _save_results(self, final_results, dataset_name, output_dir, exp_tag):
        """Save evaluation results to JSON files"""
        output_path = os.path.join(output_dir, self.model_path.split("/")[-1], exp_tag)
        os.makedirs(output_path, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_path, "results.json"), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save simple accuracy results (compatible format)
        stats = final_results["accuracy_statistics"]
        if dataset_name == "videomme":
            simple_results = {"average_acc": stats["overall_accuracy"], **stats.get("duration_accuracy", {})}
        elif "mlvu" in dataset_name or "lvb" in dataset_name:
            task_acc = stats["task_type_accuracy"]
            simple_results = {**task_acc, "Acc": sum(task_acc.values()) / len(task_acc) if task_acc else 0}
        else:
            simple_results = {"acc": stats["overall_accuracy"]}
        
        with open(os.path.join(output_path, "accuracy.json"), 'w') as f:
            json.dump(simple_results, f, indent=2)
        
        return output_path
    
    def _print_summary(self, final_results, dataset_name, output_path, last_question=None, last_response=None):
        """Print evaluation summary"""
        stats = final_results["accuracy_statistics"]
        
        if dataset_name == "sample":
            print(f"\n=== Sample Generation Results ===")
            print(f">>> QUESTION: {last_question}")
            print(f">>> RESPONSE: {last_response}")
            return
        
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Dataset: {dataset_name}")
        print(f"Overall Accuracy: {stats['overall_accuracy']:.4f} ({stats['total_correct']}/{stats['total_questions']})")
        
        if stats["task_type_accuracy"]:
            print(f"\nAccuracy by Task Type:")
            for task_type, acc in stats["task_type_accuracy"].items():
                details = stats["task_type_details"][task_type]
                print(f"  {task_type}: {acc:.4f} ({details['correct']}/{details['total']})")
        
        if dataset_name == "videomme" and stats.get("duration_accuracy"):
            print(f"\nAccuracy by Duration:")
            for duration, acc in stats["duration_accuracy"].items():
                details = stats["duration_details"][duration]
                print(f"  {duration}: {acc:.4f} ({details['correct']}/{details['total']})")
        
        print(f"\nResults saved to: {output_path}")
    
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
        task_type_stats = {}
        duration_stats = {"short": {"correct": 0, "total": 0}, 
                         "medium": {"correct": 0, "total": 0}, 
                         "long": {"correct": 0, "total": 0}}
        last_question, last_response = None, None
        
        print(f"Starting evaluation on {dataset_name} with {len(dataset)} samples...")

        for idx, data_item in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}", dynamic_ncols=True)):
            is_first_sample = (idx == 0)
            question_text = format_question(data_item, dataset_name)
            inputs = self.prepare_video_input(data_item["video"], question_text, is_first_sample)

            # Choose and run inference method
            start_time = time.time()
            frame_count = inputs["video_grid_thw"][0, 0].item()
            use_block_processing = (self.block_size > 0 and frame_count > self.block_size and 
                                    self.compress_frame_num > 0)
            
            if is_first_sample:
                print(f"Frame count: {frame_count}, Block size: {self.block_size}, Compress frames: {self.compress_frame_num}")
                print(f"Using {'Block Processing' if use_block_processing else 'Standard Generation'}")
            
            response = self.block_process(inputs) if use_block_processing else self.generate(inputs)
            inference_time = time.time() - start_time
            last_question, last_response = question_text, response
            
            # Evaluate result
            pred_answer = extract_answer(response, dataset_name)
            ground_truth = data_item["answer"]
            is_correct = pred_answer == ground_truth
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Update task stats
            task_type = data_item.get("task_type", "unknown")
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {"correct": 0, "total": 0}
            task_type_stats[task_type]["total"] += 1
            if is_correct:
                task_type_stats[task_type]["correct"] += 1
            
            # Update duration stats (VideoMME)
            if dataset_name == "videomme" and "duration" in data_item:
                duration = data_item["duration"]
                if duration in duration_stats:
                    duration_stats[duration]["total"] += 1
                    if is_correct:
                        duration_stats[duration]["correct"] += 1
            
            # Store result
            result = {
                "video_name": data_item["video_name"],
                "question": data_item["questions"],
                "ground_truth": ground_truth,
                "predicted_answer": pred_answer,
                "model_response": response,
                "is_correct": is_correct,
                "task_type": task_type,
                "inference_time": inference_time,
                "video_path": data_item["video"]
            }
            if dataset_name == "videomme":
                result["duration"] = data_item.get("duration")
                result["choices"] = data_item.get("choices")
            results.append(result)
            
            if idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Build final results
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        task_accuracy = self._compute_accuracy(task_type_stats)
        
        final_results = {
            "results": results,
            "accuracy_statistics": {
                "overall_accuracy": overall_accuracy,
                "total_questions": total_count,
                "total_correct": correct_count,
                "task_type_accuracy": task_accuracy,
                "task_type_details": task_type_stats,
                "type_wise_summary": self._create_summary(task_type_stats)
            },
            "experiment_config": {
                "dataset": dataset_name,
                "model_path": self.model_path,
                "max_frames": self.max_frames_num,
                "block_size": self.block_size,
                "compress_frame_num": self.compress_frame_num,
                "compression_method": self.compression_method,
                "load_dumped": self.load_dumped,
                "per_frame": self.per_frame
            }
        }
        
        if dataset_name == "videomme":
            final_results["accuracy_statistics"]["duration_accuracy"] = self._compute_accuracy(duration_stats)
            final_results["accuracy_statistics"]["duration_details"] = duration_stats
            final_results["accuracy_statistics"]["duration_wise_summary"] = self._create_summary(duration_stats)
        
        # Save and print results
        output_path = self._save_results(final_results, dataset_name, output_dir, exp_tag)
        self._print_summary(final_results, dataset_name, output_path, last_question, last_response)
        
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
        per_frame=args.per_frame,
        verbose=args.verbose
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
            print(f"Compression method: {args.compression_method}")
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
