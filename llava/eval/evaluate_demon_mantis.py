import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


mantis = ["NLVR2", "QBench", "Mantis", "BLINK"]


class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip("'")
        answer = answer.strip('"')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self, preds):
        rouge = Rouge()
        acc = {"f": []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res["sample_id"]
            # print(sample_id)
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            assert gt_ans != ""
            if pred_ans == "":
                s = 0
            else:
                if len(pred_ans) > 512:
                    pred_ans = pred_ans[0:512]
                s = rouge.get_scores(pred_ans, gt_ans)[0]["rouge-l"]["f"]
            acc["f"].append(s)
            eval_list.append({"id": str(sample_id), "score": str(round(s, 3))})
        results = {"Rouge-L f": np.mean(acc["f"])}
        return results, eval_list

    def judge_multi_choice(self, sample):
        sample_id = sample["sample_id"]
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]
        choice_list = sample["choice_list"]
        # if gt_ans not in choice_list:
        #     print(gt_ans)
        #     print(choice_list)
        # assert gt_ans in choice_list
        # print(pred_ans, gt_ans)
        # try:
        #     vectorizer = TfidfVectorizer()
        #     texts = [pred_ans] + choice_list
        #     tfidf_matrix = vectorizer.fit_transform(texts)
        #     cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        #     most_similar_index = cosine_similarities.argmax()
        #     if choice_list[most_similar_index] == gt_ans:
        #         return 1
        #     else:
        #         return 0
        # except:
        pred_ans = pred_ans.split(":")[0]
        if pred_ans == gt_ans:
            return 1
        else:
            return 0

    def process_sample(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        for i in range(len(sample["choice_list"])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            if "choice_list" not in sample.keys():
                print(sample)
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample["sample_id"]
            sample["result"] = score
            eval_list.append({"id": str(sample_id), "score": str(score)})
            correct += score
        return {"Accuracy": correct / len(preditions)}, eval_list

    def evaluate_multi_choice_image(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            sample_id = sample["sample_id"]
            choice_list = ["image a", "image b", "image c", "image d"]
            # if gt_ans[:7] == pred_ans[:7]:
            #     score = 1
            # else:
            #     score = 0
            # count = 0
            # for choice in choice_list:
            #     if choice in pred_ans:
            #         count += 1
            # if count > 1:
            #     score = 0
            pred_ans = pred_ans.split(":")[0]
            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
            sample_id = sample["sample_id"]
            sample["result"] = score
            eval_list.append({"id": str(sample_id), "score": str(score)})
            correct += score
        return {"Accuracy": correct / len(preditions)}, eval_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)

    args = parser.parse_args()

    result_file = os.path.join(args.result_dir, "result.jsonl")

    if not os.path.exists(result_file):
        print("No prediction file found")
        exit(0)
    with open(result_file, "r") as f:
        preds_all = [json.loads(line) for line in f]

    preds_all_dict = dict()
    for pred in preds_all:
        if pred["dataset"] not in preds_all_dict:
            preds_all_dict[pred["dataset"]] = list()
        preds_all_dict[pred["dataset"]].append(pred)

    image_choice_dataset_list = ["recipeqa-RecipeQA_VisualCloze", "RecipeQA_ImageCoherence", "COMICS_Panel"]
    E = Eval()

    eval_result_list = dict()
    eval_result_list_detail = dict()

    for dataset in preds_all_dict:
        # if dataset == "MMCoQA":
        #     continue
        preds = preds_all_dict[dataset]
        question_type = preds[0]["question_type"]

        if question_type == "open-ended":
            eval_result, eval_list = E.evaluate_rouge(preds)

        elif question_type == "multi-choice":
            if dataset in image_choice_dataset_list:
                eval_result, eval_list = E.evaluate_multi_choice_image(preds)
            else:
                eval_result, eval_list = E.evaluate_multichoice(preds)

        else:
            eval_result = "Dataset not supported"
            print("Dataset not supported")
            exit(0)

        print(dataset, end=":  ")
        print(eval_result)

        eval_result_list[dataset] = eval_result
        eval_result_list_detail[dataset] = eval_list

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "eval_dataset.json"), "w") as f:
        json.dump(f, indent=4)

    with open(os.path.join(args.result_dir, "eval_dataset_details.json"), "w") as f:
        json.dump(eval_result_list_detail, f, indent=4)

    eval_cat_list = dict()
    print()

    # mantis
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in mantis:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    score /= count
    eval_cat_list["mantis"] = score
    print("mantis", end=":  ")
    print("{:.2f}".format(100 * score))

    with open(os.path.join(args.result_dir, "eval_cat.json"), "w") as f:
        json.dump(eval_cat_list, f, indent=4)
