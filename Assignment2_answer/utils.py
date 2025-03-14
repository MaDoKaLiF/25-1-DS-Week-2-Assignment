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

def add_idx_to_batch(batch, indices):
    # indices는 배치 내 각 예제에 대한 인덱스 리스트입니다.
    batch["idx"] = indices
    return batch

def get_dataloader(args, tokenizer, rank, world_size):
    dataset, dataset_train = None, None
    
    dataset_train = load_dataset("json", data_files="CommonsenseQA/train_rand_split.jsonl")["train"]

    # Preprocess datasets
    dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)

    dataset_train = dataset_train.map(
    add_idx_to_batch,
    with_indices=True,
    batched=True
    )

    sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler_train, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    return train_loader, sampler_train 

def preprocess_function(args, examples, tokenizer, split):
    combined_texts = []

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
