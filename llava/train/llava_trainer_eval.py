import json
import subprocess

from llava.train.llava_trainer import LLaVATrainer


class LLaVAEvalTrainer(LLaVATrainer):
    def evaluate(self, evaluate_args):
        cmd = f"accelerate launch --num_processes {evaluate_args.eval_num_processes} -m lmms_eval \
                --model {evaluate_args.model} \
                --model_args {evaluate_args.model_args} \
                --tasks {evaluate_args.task_names} \
                --batch_size {evaluate_args.batch_size} \
                --log_samples_suffix {evaluate_args.log_samples_suffix} \
                --output_path {evaluate_args.output_path}"
        if evaluate_args.limit:
            cmd += f" --limit {evaluate_args.limit}"
        if evaluate_args.num_fewshot:
            cmd += f" --num_fewshot {evaluate_args.num_fewshot}"
        if evaluate_args.gen_kwargs != "":
            cmd += f" --gen_kwargs {evaluate_args.gen_kwargs}"
        if evaluate_args.log_samples:
            cmd += f" --log_samples"
        else:
            assert False, "Please log samples so that the result can be parsed"
        results = subprocess.run([cmd], shell=True, capture_output=True, text=True)
        try:
            result_file_index_start = results.stdout.index("Saved samples to ")
            result_file_index_end = results.stdout.index(f".json")
            result_file_index_start += len("Saved samples to ")
            file = results.stdout[result_file_index_start:result_file_index_end]
        except:
            result_file_index_start = results.stderr.index("Saved samples to ")
            result_file_index_end = results.stderr.index(f".json")
            result_file_index_start += len("Saved samples to ")
            file = results.stderr[result_file_index_start:result_file_index_end]
        file = file.split("/")[:-1]
        file = "/".join(file) + "/results.json"
        with open(file, "r") as f:
            lmms_eval_results = json.load(f)
        result_dict = {}
        tasks_list = evaluate_args.task_names.split(",")
        for task in tasks_list:
            task_results = lmms_eval_results["results"][task]
            for k, v in task_results.items():
                if k != "alias" and "stderr" not in k:
                    metric = k.split(",")[0]
                    result_dict[f"{task}_{metric}"] = v
        return result_dict

    """def evaluate(self, evaluate_args):
        initialize_tasks()
        tasks_list = evaluate_args.task_names.split(",")
        result_dict = {}
        results = evaluator.simple_evaluate(
            model=evaluate_args.model,
            model_args=evaluate_args.model_args,
            tasks=tasks_list,
            num_fewshot=evaluate_args.num_fewshot,
            batch_size=evaluate_args.batch_size,
            device=evaluate_args.device,
            limit=evaluate_args.limit,
            check_integrity=evaluate_args.check_integrity,
            show_task_to_terminal=evaluate_args.show_task_to_terminal,
            log_samples=evaluate_args.log_samples,
            gen_kwargs=evaluate_args.gen_kwargs,
            cli_args=evaluate_args,
        )
        for task in tasks_list:
            task_results = results["results"][task]
            for k,v in task_results.items():
                if k != "alias" and "stderr" not in k:
                    metric = k.split(",")[0]
                    result_dict[f"{task}_{metric}"] = v
            
        return result_dict"""
