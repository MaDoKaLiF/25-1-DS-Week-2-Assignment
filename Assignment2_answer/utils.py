import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import pickle
import logging
from safetensors.torch import load_file


def build_problem_id_mapping(args, dataset_name, tokenizer=None):
    """
    Build a mapping dictionary from problem IDs to their questions and answers

    Args:
        args: Arguments containing necessary paths and configurations
        dataset_name: Name of the dataset to build mapping for
        tokenizer: Optional tokenizer for datasets that need preprocessing

    Returns:
        dict: Dictionary mapping problem IDs to question/answer pairs
    """
    problem_mapping = {}

    try:
        # Handle different dataset sources and structures
        train_data = load_dataset("json", data_files="../datasets/CommonsenseQA/train_rand_split.jsonl")["train"]
        test_data = load_dataset("json", data_files="../datasets/CommonsenseQA/test.jsonl")["train"]

        # Process train set
        for idx, example in enumerate(train_data):
            question_id = str(example.get("idx", idx))
            problem_mapping[question_id] = {
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "split": "train"
            }

        # Process test set
        for idx, example in enumerate(test_data):
            question_id = str(example.get("idx", idx))
            problem_mapping[question_id] = {
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "split": "test"
            }
        
        print(f"Successfully built mapping for {len(problem_mapping)} problems from {dataset_name}")
        return problem_mapping

    except Exception as e:
        print(f"Error building problem mapping for {dataset_name}: {e}")
        return {}

def initialize_problem_tracking(args, dataset):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")

    # Check if file already exists
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        if args.rank == 0:
            print(f"Problem tracking data loaded from {tracking_file}")
    else:
        # Initialize new tracking data
        tracking_data = {}

        # Handle any dataset structure - always get train dataset
        if isinstance(dataset, dict) and "train" in dataset:
            # For dictionary-style datasets (like HF datasets)
            train_data = dataset["train"]
        else:
            # For single datasets without train/test split
            train_data = dataset

        # Process dataset
        for idx, example in enumerate(train_data):
            question_id = example.get("idx", idx)  # Use idx field if available, otherwise use index
            tracking_data[str(question_id)] = 0

        # Save to JSON file
        if args.rank == 0:
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            print(f"Problem tracking data initialized and saved to {tracking_file}")

    return tracking_data


def get_problem_tracking_stats(args, cur_iter=None, dataset=None):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")

    # Load tracking data
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)

    # Calculate how many problems were last solved at each iteration
    iter_counts = {}
    for problem_id, last_iter in tracking_data.items():
        iter_counts[last_iter] = iter_counts.get(last_iter, 0) + 1

    # Count never solved problems (iteration 0)
    never_solved = iter_counts.get(0, 0)

    # Basic statistics
    stats = {
        "total_problems": len(tracking_data),
        "solved_at_least_once": len(tracking_data) - never_solved,
        "never_solved": never_solved,
        "iteration_distribution": iter_counts
    }

    # If current iteration is provided, identify problems NOT consistently solved
    if cur_iter is not None and hasattr(args, 'sup_thresholds'):
        not_consistently_solved_ids = []

        for problem_id, last_iter in tracking_data.items():
            # Calculate iterations since last correct solution
            iterations_since_correct = cur_iter - last_iter if last_iter > 0 else cur_iter

            # Problem is not consistently solved if it was either:
            # 1. Never solved (last_iter == 0)
            # 2. Not solved for at least sup_threshold iterations
            if iterations_since_correct >= args.sup_thresholds:
                not_consistently_solved_ids.append(problem_id)

        stats["not_consistently_solved_count"] = len(not_consistently_solved_ids)
        stats["not_consistently_solved_ids"] = not_consistently_solved_ids

        # Log problems not consistently solved, including questions and answers
        log_not_consistently_solved_problems(args, not_consistently_solved_ids, cur_iter, dataset)

    return stats


def log_not_consistently_solved_problems(args, problem_ids, cur_iter, dataset=None):
    # Don't log if there are no inconsistently solved problems
    if not problem_ids:
        print(f"All problems are consistently solved (sup_threshold={args.sup_thresholds}) at iteration {cur_iter}")
        return

    # Create log file path
    log_dir = args.log_dir if hasattr(args, 'log_dir') else f"{args.task}/{args.experiment_name}"
    log_file = os.path.join(log_dir, f"not_consistently_solved_problems_iter{cur_iter}.json")

    # Extract problem details if dataset is provided
    problem_details = []
    if dataset is not None:
        # Handle any dataset structure - always get train dataset
        if isinstance(dataset, dict) and "train" in dataset:
            # For dictionary-style datasets (like HF datasets)
            train_data = dataset["train"]
        else:
            # For single datasets without train/test split
            train_data = dataset

        # Create mapping from problem ID to index in dataset
        id_to_index = {}
        for idx, example in enumerate(train_data):
            question_id = str(example.get("idx", idx))
            id_to_index[question_id] = idx

        # Extract question and answer for each problem ID
        for problem_id in problem_ids:
            if problem_id in id_to_index:
                idx = id_to_index[problem_id]
                example = train_data[idx]
                problem_details.append({
                    "id": problem_id,
                    "question": example.get("question", ""),
                    "answer": example.get("answer", "")
                })
            else:
                # If problem ID doesn't match any in dataset, just store the ID
                problem_details.append({"id": problem_id})
    else:
        # If no dataset is provided, just store the IDs
        problem_details = [{"id": pid} for pid in problem_ids]

    # Create log data
    log_data = {
        "iteration": cur_iter,
        "sup_threshold": args.sup_thresholds,
        "not_consistently_solved_count": len(problem_ids),
        "not_consistently_solved_problems": problem_details
    }

    # Save log file
    if args.rank == 0:  # Only rank 0 process logs in distributed training
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Logged {len(problem_ids)} not consistently solved problems to {log_file}")

def update_problem_tracking_stats(args, cur_iter, dataset=None):
    try:
        # Calculate current iteration statistics (including not consistently solved problems)
        updated_stats = get_problem_tracking_stats(args, cur_iter, dataset)

        # Save statistics file
        stats_file = os.path.join(f"{args.task}/{args.experiment_name}", f"problem_tracking_stats_iter{cur_iter}.json")
        if args.rank == 0:
            with open(stats_file, 'w') as f:
                json.dump(updated_stats, f, indent=2)
            print(f"Problem tracking stats updated for iteration {cur_iter}")

            # Report not consistently solved problems
            if "not_consistently_solved_count" in updated_stats:
                print(f"Found {updated_stats['not_consistently_solved_count']} problems NOT solved consistently "
                      f"(sup_threshold={args.sup_thresholds})")
    except Exception as e:
        print(f"Error updating problem tracking stats: {e}")

def update_problem_tracking(args, correct_problems, current_iter):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")
    # 현재 tracking 데이터 로드
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    # 맞춘 문제들에 대한 iteration 업데이트
    for problem_id in correct_problems:
        tracking_data[str(problem_id)] = current_iter
    # 업데이트된 데이터 저장
    if args.rank == 0:
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Problem tracking data updated for {len(correct_problems)} problems at iteration {current_iter}")
    return tracking_data

def fsdp_wrap(args, model, rank, cpu_offload=False):
    if args.precision == "bf16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  # Model parameters in bfloat16
            reduce_dtype=torch.bfloat16,  # Gradient reduction in bfloat16
            buffer_dtype=torch.bfloat16  # Buffers in bfloat16
            )
    elif args.precision == "fp16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,  # Model parameters in bfloat16
            reduce_dtype=torch.float16,  # Gradient reduction in bfloat16
            buffer_dtype=torch.float16  # Buffers in bfloat16
            )

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    args.cfg = cfg

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload,
        device_id=torch.device(rank),  # Specify the device
        )
    return model


def get_loaded_model_tokenizer(args, model_path, model_name, rank, cpu_offload=False, eval=False):
    if args.precision == "bf16":
        model_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if not args.lora:
        state_dict = torch.load(model_path, map_location="cpu")  # pt/bin 파일 로드
        model.load_state_dict(state_dict)
        if eval:
            model = model.to(rank)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = fsdp_wrap(args, model, rank, cpu_offload=cpu_offload)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora["lora_rank"],
            lora_alpha=args.lora["lora_alpha"],
            lora_dropout=args.lora["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model = model.to(model_dtype)
        if eval:
            model = model.to(rank)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = fsdp_wrap(args, model, rank)

        lora_state_dict = torch.load(model_path, map_location="cpu")
        model_state_dict = model.state_dict()
        updated_lora_state_dict = {}
        for key in lora_state_dict.keys():
            new_key = "module." + key if "module." + key in model_state_dict else key
            updated_lora_state_dict[new_key] = lora_state_dict[key]

        # Load LoRA parameters into the model
        model.load_state_dict(updated_lora_state_dict, strict=False)  # Allow missing keys
        dist.barrier()

        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.print_trainable_parameters()
            else:
                model.print_trainable_parameters()

    model = model.to(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    return model, tokenizer


def get_model_tokenizer(args, model_name, rank, eval=False):
    if args.precision == "bf16":
        model_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora["lora_rank"],
            lora_alpha=args.lora["lora_alpha"],
            lora_dropout=args.lora["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model = model.to(model_dtype)

        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.print_trainable_parameters()
            else:
                model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if eval:
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = fsdp_wrap(args, model, rank)
    return model, tokenizer


def get_optimizer_scheduler_step_based(args, model, train_loader):
    # Other tasks: Keep original warmup logic
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(current_step: int):
        if current_step < args.warm_up_steps:
            return float(current_step) / float(max(1, args.warm_up_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

def get_dataloader_STaR(args, tokenizer, rank, world_size):
    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset_train = pickle.load(f)

    if rank == 0:
        print(f"Dataset from {args.dataset_path} : total {len(dataset_train)} sequences")

    processed_data = []
    for item in dataset_train:
        processed_data.append({
            "tokens": item['tokens'],
            "idx": item['idx'],
            "original_position": len(processed_data)  # 원래 위치도 저장
        })

    # Create dataset
    dataset_train = Dataset.from_list(processed_data)

    # Add preprocessing
    dataset_train = dataset_train.map(
        lambda examples: preprocess_function_STaR(args, examples, tokenizer),
        batched=True
    )

    # Create sampler
    sampler_train = DistributedSampler(
        dataset_train,
        rank=rank,
        num_replicas=world_size,
        shuffle=False
    )

    # DataLoader kwargs
    train_kwargs = {
        'batch_size': args.batch_size,
        'sampler': sampler_train,
        'collate_fn': collate_fn_with_metadata
    }

    cuda_kwargs = {
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False
    }

    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)

    return train_loader, sampler_train


def preprocess_function_STaR(args, examples, tokenizer):
    """Enhanced preprocessing function that preserves metadata"""
    tokens_list = examples["tokens"]
    indices = examples["idx"]
    original_positions = examples["original_position"]

    tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "idx": [],
        "original_position": []
    }

    for tokens, idx, orig_pos in zip(tokens_list, indices, original_positions):
        padded_tokens = tokenizer.pad(
            {"input_ids": [tokens]},
            max_length=args.max_length,
            padding="max_length",
            return_attention_mask=True
        )

        tokenized["input_ids"].append(padded_tokens["input_ids"][0])
        tokenized["attention_mask"].append(padded_tokens["attention_mask"][0])
        tokenized["idx"].append(idx)
        tokenized["original_position"].append(orig_pos)

    return tokenized


def collate_fn_with_metadata(batch):
    """Enhanced collate function that preserves metadata"""
    collated = {
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'idx': [item['idx'] for item in batch],
        'original_position': [item['original_position'] for item in batch]
    }
    return collated

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return data

def get_wrong_examples_dataloader_STaR(args, wrong_examples, rank, world_size):
    dataset_test = Dataset.from_list(wrong_examples)

    # Create distributed samplers
    sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True)

    # Set DataLoader configurations
    train_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler_test, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)

    # Create DataLoaders
    test_loader = torch.utils.data.DataLoader(dataset_test, **train_kwargs)
    return test_loader, sampler_test


def preprocess_function_STaR(args, examples, tokenizer):
    examples = examples["tokens"]  # Extract tokenized sequences

    # Initialize a dictionary to hold results
    tokenized = {"input_ids": [], "attention_mask": [], "text": []}

    for tokens in examples:
        # Convert token sequence back to text
        original_text = tokenizer.decode(tokens, skip_special_tokens=True)

        # Pad token sequence to the left or right as required
        padded_tokens = tokenizer.pad(
            {"input_ids": [tokens]},  # Process each sequence individually
            max_length=args.max_length,  # Pad/truncate to max length
            padding="max_length",  # Always pad to max_length
            return_attention_mask=True  # Include attention mask
        )

        # Store results
        tokenized["input_ids"].append(padded_tokens["input_ids"][0])
        tokenized["attention_mask"].append(padded_tokens["attention_mask"][0])
        tokenized["text"].append(original_text)

    return tokenized

def add_idx_to_batch(batch, indices):
    # indices는 배치 내 각 예제에 대한 인덱스 리스트입니다.
    batch["idx"] = indices
    return batch

def get_dataloader(args, tokenizer, rank, world_size):
    dataset, dataset_train, dataset_test = None, None, None
    
    dataset_train = load_dataset("json", data_files="CommonsenseQA/train_rand_split.jsonl")["train"]
    dataset_test = load_dataset("json", data_files="CommonsenseQA/test.jsonl")["train"]

    # Preprocess datasets
    dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
    dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    dataset_train = dataset_train.map(
    add_idx_to_batch,
    with_indices=True,
    batched=True
    )

    dataset_test = dataset_test.map(
        add_idx_to_batch,
        with_indices=True,
        batched=True
    )
    # # For debugging purposes, limit the number of examples
    # dataset_train = dataset_train.select(range(min(128, len(dataset_train))))
    # dataset_test = dataset_test.select(range(min(128, len(dataset_test))))

    sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler_train, 'collate_fn': collate_fn}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler_test, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    return train_loader, sampler_train, test_loader, sampler_test

def preprocess_function(args, examples, tokenizer, split):
    combined_texts = []
    # question_components = examples["question"]

    for text, ans in zip(examples["question"], examples["answerKey"]):
        q = text['stem']
        choices = text['choices']
        options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
        if split == "train":
            combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
        else:
            combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

    tokenized = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length
    )

    tokenized["question"] = examples["question"]
    tokenized["answer"] = examples["answerKey"]

    log_truncation_warnings(args, combined_texts, tokenizer, log_point="Simple data load")

    return tokenized

def collate_fn(batch):
    collated_batch = {}

    # Convert input_ids and attention_mask to tensors
    for key in batch[0]:
        if key in ["input_ids", "attention_mask", "wrong_input_ids", "wrong_attention_mask"]:
            collated_batch[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        else:
            # For non-tensor data (e.g., strings, lists of strings), keep as a list
            collated_batch[key] = [item[key] for item in batch]

    return collated_batch



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def log_args(output_file, **kwargs):
    # Load existing log data or initialize an empty dictionary
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as json_file:
                logs = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
    else:
        logs = []

    logs.append(kwargs)
    with open(output_file, "w") as json_file:
        json.dump(logs, json_file, indent=4)

def log_truncation_warnings(args, prompts, tokenizer, log_point):
    # Configure logging (global configuration)
    logging.basicConfig(
        level=logging.WARNING,  # Set the default log level to WARNING
        format="%(levelname)s - %(message)s",  # Define the log message format
        handlers=[
            logging.FileHandler(f"{args.log_dir}/truncation_warnings.log")  # Save logs to a file
        ]
    )
    # Compare the original lengths of each input to the maximum length
    original_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    truncated = [original > args.max_length for original in original_lengths]

    if any(truncated):
        truncated_count = sum(truncated)
        logging.warning(f"{truncated_count} input(s) exceeded max_length and were truncated.")
        for idx, (prompt, length, is_truncated) in enumerate(zip(prompts, original_lengths, truncated)):
            if is_truncated:
                logging.warning(f"Iteration: {args.exp_iter}, Log point: {log_point}\n--Truncated Prompt {idx+1} (Token Length {length} > Max Length {args.max_length})--\n{prompt}")

def gather_and_merge_list(data_list, dst=0):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered_data = [None] * world_size  # List to hold gathered data

    # Gather data from all GPUs
    dist.all_gather_object(gathered_data, data_list)

    if rank == dst:
        all_dicts = [d for sublist in gathered_data for d in sublist]

        return all_dicts
    else:
        return None  # Other GPUs don't need to return anything

def gather_and_merge_dicts(local_dict, dst=0):
    """
    Gather a Python dictionary `local_dict` from each rank,
    and merge them on the destination rank `dst`.

    Returns:
      - merged_dict on rank=dst
      - None on all other ranks
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Prepare a list to receive all dicts on the destination rank
    # On other ranks, gather_list can be empty
    gather_list = [None] * world_size if rank == dst else []

    # Gather each local_dict into gather_list on rank=dst
    dist.gather_object(obj=local_dict, object_gather_list=gather_list, dst=dst)

    # Merge all dictionaries on the destination rank
    if rank == dst:
        merged_dict = {}
        for d in gather_list:
            for k, v in d.items():
                # Customize merging logic as needed
                # Example: just overwrite or store as a list of values
                if k not in merged_dict:
                    merged_dict[k] = v
        return merged_dict

    else:
        # Non-dst ranks return None
        return None
