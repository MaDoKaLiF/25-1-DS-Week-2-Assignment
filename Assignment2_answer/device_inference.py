import argparse
import json
import torch
import pprint

from sympy import false
from tqdm import tqdm
import os
import torch.distributed as dist
import re
from itertools import chain
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp

from utils import get_model_tokenizer, get_wrong_examples_dataloader_STaR, get_dataloader, setup, cleanup, log_args, log_truncation_warnings

pp = pprint.PrettyPrinter(indent=2).pprint


def write_new_data(args, target_save, pred, data, endoftext):
    text = data["question"]
    q = text['stem']
    choices = text['choices']
    options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
    new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext

    new_example_no_answer = q
    with open(args.idx_save + f"/{args.split}_corr_idx_{args.exp_iter}.txt", 'a+') as new_idx_f:
        print(f"idx: {data['idx']}\nQ: {new_example_no_answer}", file=new_idx_f, end="\n\n")

    with open(target_save, 'a+') as new_train_f:
        print(new_example, file=new_train_f, end="\n\n")

    return new_example
    


def test_metric_STaR(args, predictions, datas, target_save, tokenizer, hint):
    wrong_examples = []
    correct, total = 0, 0

    log_file = f"{args.log_dir}/prediction_log.jsonl"    

    try:
        for idx, (pred, data) in enumerate(zip(predictions, datas), 1):
            try:
                cur_correct = False
                answer = data.get("answer")  # Use get() to safely access
                if answer is None:
                    print(f"Warning: Missing answer for index {idx}")
                    continue

                q_start_idx = pred.find("Q: ")
                if q_start_idx != -1:
                    pred = pred[:q_start_idx]

                if "####" in pred:
                    parts = pred.split("####")
                    if len(parts) > 1 and len(parts[1].split()) > 0:
                        # Only attempt to access the first word if it exists
                        pred = parts[0] + "#### " + parts[1].split()[0]
                    else:
                        # Handle case where there's nothing after ####
                        pred = parts[0] + "#### "

                # Get predicted answer based on task type
                pred_answer = None
                try:
                    matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
                    pred_answer = matches[-1].group(1) if matches else None
                    
                except IndexError as e:
                    print(f"Warning: Failed to extract answer from prediction at index {idx}: {e}")
                    continue

                # Verify answers match
                if  pred_answer and pred_answer == answer:
                    cur_correct = True
                

                # Log prediction details
                try:
                    log_entry = {
                        "index": idx,
                        "task": args.task,
                        "question": data.get("question", ""),
                        "prediction": pred,
                        "predicted_answer": pred_answer,
                        "true_answer": answer,
                        "correct": cur_correct
                    }

                    with open(log_file, 'a') as f:
                        json.dump(log_entry, f)
                        f.write('\n')
                except Exception as e:
                    print(f"Warning: Failed to log prediction at index {idx}: {e}")

                # Handle correct/incorrect cases
                if cur_correct:
                    correct += 1
                    try:
                        if args.split == "train":
                            write_new_data(args, target_save + "/correct_data.txt", pred, data, tokenizer.eos_token)
                        else:
                            write_new_data(args, target_save, pred, data, tokenizer.eos_token)
                    except Exception as e:
                        print(f"Warning: Failed to write new data at index {idx}: {e}")
                else:
                    if not hint:
                        wrong_examples.append(data)
                total += 1

            except Exception as e:
                print(f"Warning: Error processing prediction at index {idx}: {e}")
                continue

    except Exception as e:
        print(f"Critical error in test_metric_STaR: {e}")

    return wrong_examples, correct, total

def eval_examples(args, model, rank, train_loader, tokenizer, gen_length, n_shot_prompts, hint=False):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate

    eval_progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"{'Hint' if hint else 'No Hint'} Eval [Rank {rank}]",
        position=rank + 1,
        leave=False,
        disable=(rank != 0),
    )

    correctsum = 0
    totalsum = 0
    wrong_datasets = []

    with torch.no_grad():
        for batch_idx, data in eval_progress_bar:
            try:
                tokenized = prompt_preprocess(args, data, tokenizer, n_shot_prompts, hint=hint)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                try:
                    outputs = generate_fn(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + gen_length,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=args.inference_temp
                    )
                         
                except Exception as e:
                    print(f"Warning: Generation failed for batch {batch_idx}: {e}")
                    continue

                try:
                    generated_tokens = outputs[:, input_ids.shape[-1]:]
                    predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    eos_token_id = tokenizer.eos_token_id
                    actual_lengths = 0
                    actual_c=0
                    for seq in generated_tokens:
                        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                        if eos_positions.numel() > 0:
                            actual_length = eos_positions[0].item() + 1
                        else:
                            actual_length = seq.size(0)
                        actual_lengths+=actual_length
                        actual_c+=1

                    actual_gen_len = actual_lengths
                    
                except Exception as e:
                    print(f"Warning: Decoding failed for batch {batch_idx}: {e}")
                    continue

                all_predictions = [None for _ in range(dist.get_world_size())]
                all_data = [None for _ in range(dist.get_world_size())]

                try:
                    dist.all_gather_object(all_predictions, predictions)
                    dist.all_gather_object(all_data, data)
                except Exception as e:
                    print(f"Warning: All-gather failed for batch {batch_idx}: {e}")
                    continue

                if rank == 0:
                    try:
                        all_predictions = list(chain.from_iterable(all_predictions))
                        merged_data = []
                        for rank_data in all_data:
                            for i in range(len(rank_data["question"])):
                                single_data = {}
                                for key in rank_data.keys():
                                    single_data[key] = rank_data[key][i]
                                merged_data.append(single_data)

                        wrong_examples, correct, total = test_metric_STaR(
                            args, all_predictions, merged_data,
                            args.target_save, tokenizer, hint=hint
                        )
                        correctsum += correct
                        totalsum += total
                        if not hint:
                            wrong_datasets.extend(wrong_examples)
                    except Exception as e:
                        print(f"Warning: Processing results failed for batch {batch_idx}: {e}")
                        continue

                dist.barrier()

            except Exception as e:
                print(f"Warning: Failed to process batch {batch_idx}: {e}")
                continue
    
    if rank == 0:
        if totalsum > 0:
            if hint:
                print(f"Hint Correct: {correctsum}, Accuracy: {correctsum / totalsum:.4f}")
            else:
                print(f"No hint Correct: {correctsum}, Accuracy: {correctsum / totalsum:.4f}")
        else:
            print("Warning: No valid examples were processed")
        
    return wrong_datasets, correctsum, totalsum 

def broadcast_list(data, src_rank):
    object_list = [data if dist.get_rank() == src_rank else None]  # Source rank provides the data
    dist.broadcast_object_list(object_list, src=src_rank)
    return object_list[0]

def prompt_preprocess(args, examples, tokenizer, prompt, hint):
    combined_texts = []
    for text, ans in zip(examples["question"], examples["answerKey"]):
        q = text['stem']
        choices = text['choices']
        options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
        if hint:
            combined_texts.append(f"{prompt}\nQ: {q} ({ans})\nOptions:\n{options_text}\nA: ")
        else:
            combined_texts.append(f"{prompt}\nQ: {q}\nOptions:\n{options_text}\nA: ")
                
    # Tokenize the combined texts
    tokenized = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding="max_length",  # Pad to max length
        truncation=True,  # Truncate if exceeding max length
        max_length=args.max_length,  # Adjust max length as needed
    )

    log_truncation_warnings(args, combined_texts, tokenizer, log_point="Data generation")
    
    return tokenized

def evaluate(args, model, rank, world_size, train_loader, tokenizer, gen_length, target_save, n_shot_prompts, n_shot_prompts_hint):
    # Clean existing logs before starting evaluation
    model.eval()

    if args.split == "train": # Wrong example generation only for training (Not evaluation)
        wrong_datasets, correctsum, totalsum = eval_examples(args, model, rank, train_loader, tokenizer, gen_length, n_shot_prompts, hint=False)
        wrong_datasets = broadcast_list(wrong_datasets, src_rank=0)
        wrong_train_loader, sampler_wrong_train = get_wrong_examples_dataloader_STaR(args, wrong_datasets, rank, world_size)
        if args.no_hint:
            wrong_datasets, correctsum2, totalsum2 = eval_examples(args, model, rank, wrong_train_loader, tokenizer, gen_length, n_shot_prompts, hint=False)
            correctsum += correctsum2
            correctsum_hint, totalsum_hint = "_", "_"
        else:
            wrong_wrong_datasets, correctsum_hint, totalsum_hint = eval_examples(args, model, rank, wrong_train_loader, tokenizer, gen_length, n_shot_prompts_hint, hint=True)

    dist.barrier()

    return correctsum, totalsum, correctsum_hint, totalsum_hint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument('--rationalize', action='store_true', help="Whether to use rationalization")
    parser.add_argument('--no_prompt', type=bool, default=False, help="Whether to remove prompts during eval")
    parser.add_argument('--show_hint_prompt', action='store_true', help="Whether a hint prompt will be necessary")
    parser.add_argument("--split", type=str, default="dev", help="Split")
    parser.add_argument("--task", type=str, default="cqa", help="Which dataset to run on")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Which checkpoint to eval. -1 means the final one")
    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")


    return parser.parse_args()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for this rank

    # prompt setup
    n_shot_prompts = ""
    n_shot_prompts_hint = ""
    prompt_file_path = f"./n_shot_prompts/{args.task}.json"
    prompt_hint_file_path = f"./n_shot_prompts/{args.task}_hint.json"
    with open(prompt_file_path, "r") as f:
        data = json.load(f)
    with open(prompt_hint_file_path, "r") as f:
        data_hint = json.load(f) 
    n_shot_prompts = [item["prompt"] for item in data["n_shot_prompts"]]
    n_shot_prompts_hint = [item["prompt"] for item in data_hint["n_shot_prompts"]]
    n_shot_prompts = "\n".join(n_shot_prompts)
    n_shot_prompts_hint = "\n".join(n_shot_prompts_hint)

    model, tokenizer = get_model_tokenizer(args, args.model_name, rank, eval=True)

    tokenized_prompt = tokenizer(n_shot_prompts, return_tensors="pt")
    prompt_tokenized_len = tokenized_prompt["input_ids"].shape[1]
    tokenized_prompt_hint = tokenizer(n_shot_prompts_hint, return_tensors="pt")
    prompt_tokenized_len_hint = tokenized_prompt_hint["input_ids"].shape[1]
    args.max_length += max(prompt_tokenized_len, prompt_tokenized_len_hint)

    args.batch_size = args.test_batch_size # for inference task on train set

    train_loader, sampler_train = get_dataloader(args, tokenizer, rank, world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    if args.split == "train":
        init_start_event.record()
        correct, total, correct_hint, total_hint = evaluate(args, model, rank, world_size, train_loader, tokenizer, args.gen_length, args.target_save, n_shot_prompts, n_shot_prompts_hint)
        init_end_event.record()
        if rank == 0:
            log_args(f"{args.log_dir}/elapsed_time_log.json", iter=args.exp_iter, log_point="gen_train_data", time=init_start_event.elapsed_time(init_end_event) / 1000)

    if rank == 0:
        accuracy = correct / total
        if args.split == "train":
            if not args.no_hint:
                hint_accuracy = correct_hint / total_hint
            else:
                hint_accuracy = "_"
            print(f"{args.split}, {args.task}, accuracy: {accuracy}, hint_accuracy: {hint_accuracy}")
            log_args(f"{args.log_dir}/eval_log.json", iter=args.exp_iter, split=args.split, accuracy=accuracy, hint_accuracy=hint_accuracy)

        # else:
        #     print(f"{args.split}, {args.task}, {accuracy}")
        #     log_args(f"{args.log_dir}/eval_log.json", iter=args.exp_iter, split=args.split, accuracy=accuracy)
    
    cleanup()

if __name__ == "__main__":
    # TODO implement prompt inference when ready

    args = parse_args()
    print(args)
    split = args.split
    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.model_name = params["model_name"]
    args.precision = params["precision"]
    args.max_length = params["max_length"]
    args.gen_length = params["gen_length"]
    args.n_shot = params["n_shot"]
    args.self_consistency = params["self_consistency"]
    args.delete_model_after_loading = params["delete_model_after_loading"]
    args.accumulate = params["accumulate"]
    args.lora = params.get("lora", None)
    args.inference_temp = params["inference_temp"]
    args.no_hint = params["no_hint"]

    # STaR specific
    args.name = params["name"]
    args.idx_save = params["target_save"] 
    args.target_save = params["target_save"] 
    args.model_dir = params["model_dir"]
    try: # load from trained model
        args.total_steps = params["total_steps"]
    except: # Use base model
        args.total_steps = 0
    
    args.method = params["method"]

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)


