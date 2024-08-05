import json
import re

json_path = "/mnt/bn/vl-research/workspace/boli01/projects/sft_data_workspace/vlfeedback_80k.jsonl"

with open(json_path, "r") as f:
    data = f.readlines()

data = [json.loads(d) for d in data]


def convert_format(original_data, dimension="Visual Faithfulness"):
    converted_data = []
    for item in original_data:
        # Assuming the best response is the one with the highest helpfulness rating
        best_completion = max(item["completions"], key=lambda x: int(x["annotations"]["Helpfulness"]["Rating"]))
        best_response = best_completion["response"]
        best_model = best_completion["model"]

        if "†source" in best_response:
            print(best_response)
            # Regex pattern to match the pattern 【digit†source】
            pattern = r"【\d+†source】"
            # Replace the matched patterns with an empty string
            cleaned_text = re.sub(pattern, "", best_response)
            best_response = cleaned_text
            print(f"*****************************************")
            print(best_response)

        # Assuming the worst response is the one with the lowest helpfulness rating
        worst_completion = min(item["completions"], key=lambda x: int(x["annotations"]["Helpfulness"]["Rating"]))
        worst_response = worst_completion["response"]

        if "†source" in worst_response:
            print(worst_response)
            # Regex pattern to match the pattern ��digit†source】
            pattern = r"【\d+†source】"
            # Replace the matched patterns with an empty string
            cleaned_text = re.sub(pattern, "", worst_response)
            worst_response = cleaned_text
            print(f"*****************************************")
            print(worst_response)

        # Extract scores
        best_score = int(best_completion["annotations"][dimension]["Rating"])
        worst_score = int(worst_completion["annotations"][dimension]["Rating"])

        # Construct the new format
        new_item = {
            "id": item["id"],
            "prompt": item["prompt"],
            "answer": "",
            "image": f"silkie_dpo/{item['id']}.jpg",  # Assuming the video ID is the last part of the original ID
            "chosen": best_response,
            "rejected": worst_response,
            "chosen_score": best_score,
            "rejected_score": worst_score,
        }
        converted_data.append(new_item)

    return converted_data


for dimension in ["Visual Faithfulness", "Helpfulness", "Ethical Considerations"]:
    converted_data = convert_format(data, dimension=dimension)
    with open(f"/mnt/bn/vl-research/data/llava_instruct/dpo_data/silkie_dpo_data_{dimension.replace(' ', '_').lower()}_{len(converted_data)}.json", "w") as f:
        json.dump(converted_data, f, indent=4)
