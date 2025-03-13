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
    parser.add_argument("--n_iters", type=int, default=46, help="Upper limit on outer loop iterations")

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

    if args.method == "no_beta_cap":
        train_cmd = f"python3 device_inference.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"

    elif args.method in {"nobeta_k_sample", "k_sample"}:
        train_cmd = f"python3 device_inference_ksample.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"

    elif args.method == "adastar":
        train_cmd = f"python3 device_inference_adastar.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
        train_cmd += f" --cur_total_steps={get_n_steps()}"

    elif args.method == "adastar_new":
        train_cmd = f"python3 device_inference_adastar_new.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
        train_cmd += f" --cur_total_steps={get_n_steps()}"

    elif args.method == "k_sample_adastar_new":
        train_cmd = f"python3 device_inference_ksample_adastar_new.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
        train_cmd += f" --cur_total_steps={get_n_steps()}"

    elif args.method == "irpo":
        train_cmd = f"python3 answer_generation_irpo.py --config={prev_config} --seed={args.seed} --task={task} --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"

    print(f"Generating training set {cur_iter} using model {cur_iter - 1}: {train_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        if args.method in {"vanilla", "k_sample", "adastar", "adastar_new", "k_sample_adastar_new"} and (cur_iter == 1) and os.path.exists(
                record_folder(0) + f"/correct_data.txt"):
            print("First file cached")
        elif args.method == "irpo" and (cur_iter == 1) and os.path.exists(record_folder(0) + f"/irpo_data.jsonl"):
            print("First file cached")
        else:
            os.system(train_cmd)


def gen_records():
    gen_cmd = f'python3 create_finetune_tfrecords.py {record_folder(cur_iter - 1)} {record_folder(cur_iter - 1)}  --model_name={args.model_name} --seed={args.seed}'
    gen_cmd += f' --max-length={args.max_length}'
    gen_cmd += f' --idx_save={record_folder(cur_iter - 1)}'
    gen_cmd += f' --split=train'
    gen_cmd += f' --exp_iter={cur_iter}'

    print(f"Creating records for finetuning {cur_iter}: {gen_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        if args.method in {"vanilla", "no_beta_cap", "adastar", "adastar_new", "k_sample", "k_sample_adastar_new", "nobeta_k_sample"}:
            os.system(gen_cmd)
        elif args.method == "irpo":
            pass

    train_set = f"{experiment_name}/{exp_iteration}.index"

    if args.method in {"vanilla", "no_beta_cap", "adastar", "adastar_new", "k_sample", "k_sample_adastar_new", "nobeta_k_sample"}:
        with open(f"data/{train_set}", "w") as new_data_file:
            new_data_file.write(f"{record_folder(cur_iter - 1)}.pt")
    if args.method == "irpo":
        with open(f"data/{train_set}", "w") as new_data_file:
            new_data_file.write(f"{record_folder(cur_iter - 1)}/irpo_data.jsonl")
    return

def get_no_beta_steps():
    log_file = os.path.join(record_folder(cur_iter-1), "step.json")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        last_step = logs[-1]["step"]
        print("Last step:", last_step)
        
        return last_step
 

def get_n_steps():
    if args.method == "irpo":
        return args.start_steps
    if args.steady_grow:
        return int(args.start_steps + args.add_steps * (cur_iter - 1))
    elif args.exponential_grow:  # Default setup from STaR paper
        return int(args.start_steps * (args.grow_steps ** (cur_iter - 1)))
    else:
        # Count data points
        total_count = 0
        for cur_file in sorted(os.listdir(record_folder(cur_iter - 1)),
                               key=lambda x: int(x.split('.')[0].split("_")[-1])):
            with open(f"{record_folder(cur_iter - 1)}/{cur_file}", encoding='utf-8') as train_file:
                train_file_text = train_file.read()
                total_count += len(train_file_text.split("\n\n"))
                print(len(train_file_text.split("\n\n")))
        train_epochs = args.base_epochs + args.add_epochs * (cur_iter - 1)
        cur_steps = int(
            total_count * train_epochs // (args.batch_size * args.grad_accumulation * torch.cuda.device_count()))
        return cur_steps  # cur_steps


def gen_config():
    print(f"Creating new config file {cur_iter}")
    config_name = f'configs/{experiment_name}/{exp_iteration}.json'
    os.makedirs(record_folder(cur_iter), exist_ok=True)
    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)
        new_json["model_dir"] = f"{args.base_model_location}/{exp_iteration}"
        new_json["target_save"] = record_folder(cur_iter)
        new_json["total_steps"] = get_n_steps()
        new_json["name"] = exp_iteration
        new_json["p_rationalization"] = args.p_rationalization
        new_json["grad_accumulation"] = args.grad_accumulation
    with open(config_name, "w", encoding='utf-8') as new_json_file:
        json.dump(new_json, new_json_file, indent=2)
    os.makedirs(new_json["model_dir"], exist_ok=True)
    return config_name

def gen_no_beta_config():
    print(f"Creating new config file {cur_iter}")
    config_name = f'configs/{experiment_name}/{exp_iteration}.json'
    os.makedirs(record_folder(cur_iter), exist_ok=True)
    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)
        new_json["model_dir"] = f"{args.base_model_location}/{exp_iteration}"
        new_json["target_save"] = record_folder(cur_iter)
        new_json["total_steps"] = get_no_beta_steps()
        new_json["name"] = exp_iteration
        new_json["p_rationalization"] = args.p_rationalization
        new_json["grad_accumulation"] = args.grad_accumulation
    with open(config_name, "w", encoding='utf-8') as new_json_file:
        json.dump(new_json, new_json_file, indent=2)
    os.makedirs(new_json["model_dir"], exist_ok=True)
    return config_name


def train_model():
    if args.method in {"vanilla", "adastar", "adastar_new", "k_sample", "k_sample_adastar_new"}:
        model_cmd = f"python device_train.py --config {config_name} --tune-model-path={args.base_model_location} --exp_iter={cur_iter} --seed={args.seed} --log_dir={record_folder(cur_iter - 1)} --data_dir=data/{experiment_name}/{exp_iteration}.index"
        if len(log_gen) >1:  
            delete_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 2}/step_{log_gen[0]}/lm.pt"
            model_cmd += f" --delete_path={delete_path}"
        if args.accumulate == True and cur_iter != 1:
            model_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 1}/step_{log_gen[-1]}/lm.pt"
            model_cmd += f" --model_path={model_path}"
    elif args.method in {"nobeta_k_sample", "no_beta_cap"}:
        model_cmd = f"python device_train_no_beta_cap.py --config {config_name} --tune-model-path={args.base_model_location} --exp_iter={cur_iter} --seed={args.seed} --log_dir={record_folder(cur_iter - 1)} --data_dir=data/{experiment_name}/{exp_iteration}.index"
        if len(log_gen) >1:
            delete_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 2}/step_{log_gen[0]}/lm.pt"
            model_cmd += f" --delete_path={delete_path}"
        if args.accumulate == True and cur_iter != 1:
            model_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 1}/step_{log_gen[-1]}/lm.pt"
            model_cmd += f" --model_path={model_path}"

    elif args.method == "irpo":
        model_cmd = f"python device_train_irpo.py --config {config_name} --tune-model-path={args.base_model_location} --exp_iter={cur_iter} --seed={args.seed} --log_dir={record_folder(cur_iter - 1)} --data_dir=data/{experiment_name}/{exp_iteration}.index"

    print(f"Train model {cur_iter}: {model_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        os.system(model_cmd)


def eval_model():
    eval_cmd = f"python3 device_inference.py --config={config_name} --split=dev --seed={args.seed}"
    eval_cmd += f" --task={task} "
    eval_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
    eval_cmd += f" --flops_dir={record_folder(cur_iter-1)}"

    print(f"Eval model {cur_iter}: {eval_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter) and not args.skip_eval:
        os.system(eval_cmd)


def copy_files():
    all_files = sorted(os.listdir(record_folder(cur_iter - 1)), key=lambda x: int(x.split('.')[0].split("_")[-1]))
    relevant_files = all_files[-args.copy_n:]
    for cur_file in relevant_files:
        shutil.copy(f"{record_folder(cur_iter - 1)}/{cur_file}", record_folder(cur_iter))


def make_first_config():
    if args.method != "vanilla" and args.method != "no_beta_cap":
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


def find_last_completed_iteration(experiment_name, n_iters):
    log_file_path = f"{task}/{experiment_name}/eval_log.json"

    if not os.path.exists(log_file_path):
        return 0, None

    try:
        with open(log_file_path, 'r') as log_file:
            logs = json.load(log_file)

            if isinstance(logs, list):
                # 각 iteration의 train/dev 상태를 추적하는 딕셔너리
                completed_iters = {}

                # 모든 로그 엔트리 처리
                for entry in logs:
                    if not isinstance(entry, dict) or 'iter' not in entry:
                        continue

                    iter_num = entry['iter']
                    split = entry.get('split')

                    if iter_num not in completed_iters:
                        completed_iters[iter_num] = {'train': False, 'dev': False}

                    if split == 'train':
                        completed_iters[iter_num]['train'] = True
                    elif split == 'dev':
                        completed_iters[iter_num]['dev'] = True

                # 완전히 완료된 마지막 iteration 찾기
                fully_completed_iters = [
                    iter_num for iter_num, status in completed_iters.items()
                    if status['train'] and status['dev']
                ]

                if not fully_completed_iters:
                    return 0, None

                last_full_iter = max(fully_completed_iters)

                # 다음 iteration이 train만 완료되었는지 확인
                # next_iter = last_full_iter + 1
                # if next_iter in completed_iters:
                #     if completed_iters[next_iter]['train'] and not completed_iters[next_iter]['dev']:
                #         return last_full_iter, 'eval'

                return last_full_iter, None

            elif isinstance(logs, dict) and 'iter' in logs:
                iter_num = logs['iter']
                return iter_num - 1, None

    except json.JSONDecodeError:
        return 0, None

    return 0, None


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

    # 마지막으로 완료된 iteration과 재시작 지점 찾기
    last_completed_iter, restart_point = find_last_completed_iteration(experiment_name, args.n_iters)

    # 다음 iteration 설정
    args.start_iter = last_completed_iter + 1
    log_gen =[]

    # 첫 iteration이 아닌 경우 처리
    if last_completed_iter > 0:
        prev_config = f'configs/{experiment_name}/{experiment_name}_{last_completed_iter}.json'
        if not os.path.exists(prev_config):
            print(f"Warning: Could not find config file for last completed iteration: {prev_config}")
            print("Falling back to base config")
            prev_config = f"configs/{experiment_name}/base.json"
        else:
            print(f"Starting iteration {args.start_iter} using config from iteration {last_completed_iter}")

        # flops 로그 파일 삭제
        flops_log_files = glob.glob(record_folder(last_completed_iter) + f"/flops_log*.json")
        for file in flops_log_files:
            os.remove(file)
        indice_log_files = glob.glob(record_folder(last_completed_iter) + f"/*indices_log.json")
        for file in indice_log_files:
            os.remove(file)
        file_path=record_folder(last_completed_iter) + f"/final_indices_stats.json"
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path=record_folder(last_completed_iter) + f"/train_log.json"
        if os.path.exists(file_path):
            os.remove(file_path)
            # 이전 iteration의 beta 구하기
        for cur_iter in range(args.start_iter - 2,args.start_iter):
            exp_iteration = f"{experiment_name}_{cur_iter}"
            log_gen.append(get_n_steps())

    os.makedirs(f'data/{experiment_name}', exist_ok=True)
    os.makedirs(f'{task}/{experiment_name}', exist_ok=True)


    # 메인 학습 루프
    for cur_iter in range(args.start_iter, args.n_iters + 1):
        exp_iteration = f"{experiment_name}_{cur_iter}"

        # #restart_point가 'eval'이고 첫 iteration인 경우, train 과정 건너뛰기
        # if restart_point == 'eval' and cur_iter == args.start_iter:
        #     config_name = f'configs/{experiment_name}/{experiment_name}_{cur_iter}.json'
        #     eval_model()
        # else:
        gen_train()

        start_time = time.time()
        gen_records()
        elapsed_record = time.time() - start_time
        file = f"{task}/{experiment_name}/elapsed_time_log.json"
        log_args(file, iter=cur_iter, log_point="tfrecord", time=elapsed_record)

        config_name = gen_config()
        train_model()
        if args.method in {"no_beta_cap", "nobeta_k_sample"}:
            gen_no_beta_config()
        eval_model()

        prev_config = config_name
        if args.method in {"no_beta_cap", "nobeta_k_sample"}:
            log_gen.append(get_no_beta_steps())
        else:
            log_gen.append(get_n_steps())
        if len(log_gen) > 2:
            log_gen.pop(0)

        if args.copy_n > 0:
            copy_files()
