import json
import numpy as np



def micro_average(paths, ground_truth, n=5):
    C = [0] * n
    total = 0

    for _, path in paths.items():
            total += len(path)

    for k in range(n):
        for episode_id, path in paths.items():
            if len(path) < k + 1:
                pass
            else:
                gt = ground_truth[episode_id]
                if int(path[k]) == (gt[k]):
                    C[k] += 1
    return sum(C) / total

def accuracy_mean(paths, ground_truth, n=5):
    C = [0] * n
    for k in range(n):
        for episode_id, path in paths.items():
            if len(path) < k + 1:
                pass
            else:
                gt = ground_truth[episode_id]
                if int(path[k]) == (gt[k]):
                    C[k] += 1
    for j in range(n):
        count = 0
        for _ , path in paths.items():
            if len(path) >= j + 1:
                count += 1
        C[j] = C[j] / count
    C = np.array(C)
    return np.mean(C)

def accuracy_mean_0_15(paths, ground_truth, n=5):
    C = [0] * n
    for k in range(n):
        for episode_id, path in paths.items():
            if len(path) < k + 1:
                pass
            else:
                gt = ground_truth[episode_id]
                if int(path[k]) == (gt[k]):
                    C[k] += 1
    for j in range(n):
        count = 0
        for _ , path in paths.items():
            if len(path) >= j + 1:
                count += 1
        C[j] = C[j] / count
    C = np.array(C)
    C = C[:15]
    return np.mean(C)

def accuracy_mean_15_end(paths, ground_truth, n=5):
    C = [0] * n
    for k in range(n):
        for episode_id, path in paths.items():
            if len(path) < k + 1:
                pass
            else:
                gt = ground_truth[episode_id]
                if int(path[k]) == (gt[k]):
                    C[k] += 1
    for j in range(n):
        count = 0
        for _ , path in paths.items():
            if len(path) >= j + 1:
                count += 1
        C[j] = C[j] / count
    C = np.array(C)
    C = C[15:]
    return np.mean(C)

def accuracy_mean_0_143(paths, ground_truth, n=5):
    C = [0] * n
    for k in range(n):
        for episode_id, path in paths.items():
            if len(path) < k + 1:
                pass
            else:
                gt = ground_truth[episode_id]
                if int(path[k]) == (gt[k]):
                    C[k] += 1
    for j in range(n):
        count = 0
        for _ , path in paths.items():
            if len(path) >= j + 1:
                count += 1
        C[j] = C[j] / count
    C = np.array(C)
    C = C[:143]
    return np.mean(C)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    for eval_type in ["qwen_val_seen", "qwen_val_seen_uniform", "qwen_val_seen_exp"]:
        print(f"Evaluating {eval_type}")
        
        paths_file = f"/auto/home/hakobtam/LLaVA-NeXT-AirVLN/evals/outputs/{eval_type}_results.json"
        gt_file = f"/auto/home/hakobtam/LLaVA-NeXT-AirVLN/evals/outputs/{eval_type}_gt.json"

        paths = load_json(paths_file)
        ground_truth = load_json(gt_file)
        max = 0

        for path in paths.items():
            if max < len(path[1]):
                max = len(path[1])

        n_steps = max
        micro_average_ = micro_average(paths, ground_truth, n=n_steps)
        accuracy_mean_ = accuracy_mean(paths, ground_truth, n=n_steps)
        accuracy_mean_0_15_ = accuracy_mean_0_15(paths, ground_truth, n=n_steps)
        accuracy_mean_15_end_ = accuracy_mean_15_end(paths, ground_truth, n=n_steps)
        accuracy_mean_0_143_ = accuracy_mean_0_143(paths, ground_truth, n=n_steps)

        print(eval_type)
        # Print the accuracy per step
        #for i, acc in enumerate(accuracy_per_step):
        #    print(f"Step {i+1}: Accuracy = {acc:.4f}")

        print(f"{eval_type} with 15 steps")
        print(f"Micro_Average = {micro_average_}")
        print(f"Accuracy_Mean = {accuracy_mean_}")
        print(f"Accuracy_Mean_0_15 = {accuracy_mean_0_15_}")
        print(f"Accuracy_Mean_15_END = {accuracy_mean_15_end_}")
        print(f"Accuracy_Mean_0_143 = {accuracy_mean_0_143_}")
        print("-" * 20)
