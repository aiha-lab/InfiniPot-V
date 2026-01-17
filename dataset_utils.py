"""
Dataset utilities for video understanding evaluation

This module contains:
- EvalDataset: Dataset class for loading various video QA benchmarks
- format_question: Format questions based on dataset type
- extract_answer: Extract answers from model responses
"""

import os
import json
import re
import torch
from datasets import load_dataset


class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for video understanding evaluation"""

    VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(self, data_path: str, dataset: str) -> None:
        super(EvalDataset, self).__init__()
        self.data_path = data_path
        self.dataset = dataset
        self.data = self._load_data()
    
    def _load_data(self):
        """Load dataset based on dataset name"""
        if self.dataset == "videomme":
            return self._load_videomme()
        elif "mlvu" in self.dataset or "sample" in self.dataset:
            return self._load_mlvu_format("json")
        elif "lvb" in self.dataset:
            return self._load_lvb()
        elif self.dataset == "egoschema":
            return self._load_egoschema()
        else:
            raise NotImplementedError(f"Dataset not available: {self.dataset}. Choose from {{videomme, mlvu, egoschema, lvb, sample}}")
    
    def _load_videomme(self):
        """Load VideoMME dataset"""
        data_list = load_dataset("lmms-lab/Video-MME")
        results = []
        
        for item in data_list:
            video_ytid = item["url"].split("watch?v=")[-1]
            video_path = os.path.join(self.data_path, "video", f"{video_ytid}.mp4")
            
            # Try different video formats
            for fmt in self.VIDEO_FORMATS:
                temp_path = os.path.join(self.data_path, "video", f"{video_ytid}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            
            results.append({
                "questions": item["question"],
                "video": video_path,
                "subtitle": os.path.join(self.data_path, "subtitle", f"{video_ytid}.srt"),
                "video_name": video_ytid,
                "answer": item["answer"],
                "duration": item["duration"],
                "task_type": item["task_type"],
                "choices": item["options"],
            })
        
        return results
    
    def _load_mlvu_format(self, json_name):
        """Load MLVU-format datasets (MLVU, sample)"""
        json_folder_path = os.path.join(self.data_path, json_name)
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
        
        data_list = {}
        for json_file in json_files:
            task_name = re.sub(r'^\d+_(.+)\.json$', r'\1', json_file)
            data_list[task_name] = (
                f"{json_name}/{json_file}",
                f"video/{json_file.replace('.json', '')}",
            )
        
        results = []
        for task_name, (json_path, video_folder) in data_list.items():
            with open(os.path.join(self.data_path, json_path), "r") as f:
                json_data = json.load(f)
            
            for data in json_data:
                question, answer = self._qa_template(data)
                results.append({
                    "task_type": task_name,
                    "video": os.path.join(self.data_path, video_folder, data["video"]),
                    "video_name": data["video"],
                    "questions": data["question"],
                    "prompt": question,
                    "answer": answer,
                    "duration": data["duration"],
                    "choices": data["candidates"]
                })
        
        return results
    
    def _load_lvb(self):
        """Load LongVideoBench dataset"""
        json_name = "wo_subtitle"
        json_folder_path = os.path.join(self.data_path, json_name)
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
        
        data_list = {}
        for json_file in json_files:
            task_name = re.sub(r'^\d+_(.+)\.json$', r'\1', json_file)
            data_list[task_name] = (
                f"{json_name}/{json_file}",
                f"video/{json_file.replace('.json', '')}",
            )
        
        results = []
        for task_name, (json_path, video_folder) in data_list.items():
            with open(os.path.join(self.data_path, json_path), "r") as f:
                json_data = json.load(f)
            
            for data in json_data:
                question, _ = self._qa_template(data, abcd=False)
                results.append({
                    "task_type": task_name[:-5],  # Remove suffix
                    "video": os.path.join(self.data_path, "video", data["video"]),
                    "video_name": data["video"],
                    "questions": data["question"],
                    "prompt": question,
                    "answer": data["answer"],
                    "duration": data["duration"],
                    "choices": data["candidates"]
                })
        
        return results
    
    def _load_egoschema(self):
        """Load EgoSchema dataset"""
        answer_list = json.load(open(os.path.join(self.data_path, "subset_answers.json"), "r"))
        questions_list = json.load(open(os.path.join(self.data_path, "questions.json"), "r"))
        
        questions_dict = {q["q_uid"]: q for q in questions_list}
        answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        
        results = []
        for key, answer in answer_list.items():
            data = questions_dict[key]
            options = [data[f"option {i}"] for i in range(5)]
            prompt = (
                f"Question: {data['question']}\nOptions:\n"
                f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\n"
                "Respond with only the letter (A, B, C, D or E) of the correct option."
            )
            
            results.append({
                "video": os.path.join(self.data_path, "video", f"{key}.mp4"),
                "video_name": key,
                "questions": data["question"],
                "prompt": prompt,
                "answer": answer_map.get(answer),
                "choices": ", ".join(f"{i}.{opt}" for i, opt in enumerate(options)),
                "duration": None,
                "task_type": None,
            })
        
        return results
    
    def _qa_template(self, data, abcd=True):
        """Generate QA template for MLVU-style datasets"""
        question = f"Question: {data['question']}\nOptions:\n"
        answer_idx = -1
        
        for idx, candidate in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {candidate}\n"
            if candidate == data["answer"]:
                answer_idx = idx
        
        if abcd:
            question += "Respond with only the letter (A, B, C or D) of the correct option.\n"
        else:
            question += "Please answer with the letter for the correct option.\n"
        
        return question.rstrip(), chr(ord('A') + answer_idx)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


def format_question(data_item, dataset_name):
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
        instruct = f"Question: {q}\nOptions:\n"
        for op in ops:
            instruct += f"{op}\n"
        instruct += "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        return instruct.rstrip()
    
    elif "mlvu" in dataset_name or dataset_name == "egoschema" or "lvb" in dataset_name:
        return data_item["prompt"]
    
    elif "sample" in dataset_name:
        return data_item["questions"] + " Respond with which option is the correct answer and explain why it is the correct answer."
    
    else:
        raise NotImplementedError(f"Question formatting not implemented for dataset: {dataset_name}")


def extract_answer(response, dataset_name):
    """
    Extract answer from model response based on dataset format
    
    Args:
        response: Raw model response
        dataset_name: Name of the dataset
        
    Returns:
        Extracted answer letter
    """
    response = response.replace("Answer", "")
    
    if "ego" in dataset_name or "lvb" in dataset_name:
        letters = ["A", "B", "C", "D", "E"]
        pred_answer = re.findall(r"[\(\ ]*[A-E][\)\ ]*", response)
    else:
        letters = ["A", "B", "C", "D"]
        pred_answer = re.findall(r"[\(\ \[]*([A-D])[\)\.\ \]]*", response)
    
    if len(pred_answer) >= 1:
        pred_answer = pred_answer[0].strip().strip("()")
        
    if pred_answer in letters:
        return letters[letters.index(pred_answer)]
    # else:
    #     print(f">>> No alphabet found!!! pred_answer: {pred_answer}, response: {response}", flush=True)
    #     return letters[2]  # Default to C


def get_default_data_path(dataset_name):
    """
    Get default data path for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Default data path string
    """
    default_paths = {
        "mlvu": "your MLVU directory",
        "ego": "your egoschema directory",
        "mme": "your videomme directory",
        "lvb": "your longvideobench directory",
        "sample": "sample",
    }
    
    for key, path in default_paths.items():
        if key in dataset_name:
            return path
    
    raise ValueError(f"Please provide data_path for dataset: {dataset_name}")
