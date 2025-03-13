import os
import sys
import json
import shutil
import argparse
import torch
import time
import glob
from utils import log_args


def record_folder(cur_iter):
    return f"{task}/{experiment_name}/{experiment_name}_{cur_iter}"

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Specific training setups (not used)
    parser.add_argument("--steady_grow", action='store_true', help="Whether to use a fixed number of epochs")
    parser.add_argument("--add_steps", type=float, default=20., help="Steps to add each iteration")

    # Default training setup from paper
    parser.add_argument("--start_steps", type=float, default=40, help="Steps for the first iteration")
    parser.add_argument("--exponential_grow", type=bool, default=True, help="Whether to use a fixed number of epochs")
    parser.add_argument("--grow_steps", type=float, default=1.2, help="Steps to add each iteration")

    # Rationalization parameters
    parser.add_argument("--p_rationalization", type=float, default=1., help="Percent of wrong examples to rationalize")
    parser.add_argument("--p_show_hint_save", type=float, default=0., help="Percent of rationalization hints to save")
    parser.add_argument('--rationalize', type=bool, default=True, help="Whether to use rationalization")

    parser.add_argument("--start_iter", type=int, default=1, help="Starting iteration")
    parser.add_argument("--n_iters", type=int, default=1, help="Upper limit on outer loop iterations")

    parser.add_argument("--copy_n", type=int, default=0, help="Number of files to copy each iteration")

    parser.add_argument("--config", type=str, required=True, help="Base config file location")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--method", type=str, default="vanilla", help="training method (vanilla, dpo)")

    parser.add_argument('--dry_run', action='store_true', help="Whether to do a quick run to visualize output")
    parser.add_argument('--skip_eval', action='store_true', help="Whether to skip evaluation (e.g. arithmetic)")

    args = parser.parse_args()
    return args


def gen_train():
    if args.method == "vanilla":
        train_cmd = f"python3 device_inference.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"

    print(f"Generating training set {cur_iter} using model {cur_iter - 1}: {train_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        if args.method =="vanilla" and (cur_iter == 1) and os.path.exists(
                record_folder(0) + f"/correct_data.txt"):
            print("First file cached")
        else:
            os.system(train_cmd)


def copy_files():
    all_files = sorted(os.listdir(record_folder(cur_iter - 1)), key=lambda x: int(x.split('.')[0].split("_")[-1]))
    relevant_files = all_files[-args.copy_n:]
    for cur_file in relevant_files:
        shutil.copy(f"{record_folder(cur_iter - 1)}/{cur_file}", record_folder(cur_iter))


def make_first_config():
    if args.method != "vanilla":
        with open(f'configs_method/{args.method}.json', 'r') as method_json_file:
            method_json = json.load(method_json_file)
    else:
        method_json = {}

    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)

        # args loaded from config
        args.batch_size = new_json["batch_size"]
        args.grad_accumulation = new_json["grad_accumulation"]
        args.model_name = new_json["model_name"]
        args.n_shot = new_json["n_shot"]
        args.base_model_location = new_json["model_dir"]
        args.gen_length = new_json["gen_length"]
        args.delete_model_after_loading = new_json["delete_model_after_loading"]
        args.accumulate = new_json["accumulate"]
        args.max_length = new_json["max_length"]
        args.warm_up_steps = new_json["warm_up_steps"]
        args.task = new_json["task"]
        global task
        task = args.task

        os.makedirs(record_folder(0), exist_ok=True)
        new_json["p_rationalization"] = args.p_rationalization
        new_json["target_save"] = record_folder(0)
        new_json["name"] = f"{experiment_name}_0"
        new_json["method"] = args.method
        for key, value in method_json.items():
            new_json[key] = value

    with open(prev_config, "w", encoding='utf-8') as base_json_file:
        json.dump(new_json, base_json_file, indent=2)
    return new_json


if __name__ == "__main__":
    args = parse_args()
    print(args)
    experiment_name = "_".join([args.config.split("/")[-1].split(".")[0], args.method, str(args.seed)])
    experiment_name = ''.join(ch for ch in experiment_name if ch.isalnum() or ch == "_")

    os.makedirs(f"configs/{experiment_name}", exist_ok=True)
    shutil.copy(args.config, f"configs/{experiment_name}/base.json")

    # 초기 설정
    prev_config = f"configs/{experiment_name}/base.json"
    new_json = make_first_config()
    task = args.task

    log_gen =[]

    os.makedirs(f'data/{experiment_name}', exist_ok=True)
    os.makedirs(f'{task}/{experiment_name}', exist_ok=True)


    # 메인 학습 루프
    for cur_iter in range(1, args.n_iters + 1):
        exp_iteration = f"{experiment_name}_{cur_iter}"
        gen_train()
