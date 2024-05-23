import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


multi_modal_dialogue = ["MMCoQA", "ALFRED"]
visual_story_telling_list = ["AESOP", "DiDeMoSV", "FlintstonesSV", "PororoSV", "VIST"]
visual_relation_inference = ["IEdit", "Spot-the-Diff", "Birds-to-Words", "CLEVR-Change"]
multi_modal_cloze = ["COMICS_Dialogue", "COMICS_Panel", "RecipeQA_VisualCloze", "RecipeQA_TextCloze"]
knowledge_grounded_qa = ["WebQA", "TQA", "MultiModalQA"]
text_rich_images_qa = ["SlideVQA", "OCR-VQA", "DocVQA"]
multi_image_reasoning = ["Fashion200K", "NLVR2", "nuscenes", "VizWiz", "MIT-States_StateCoherence", "MIT-States_PropertyCoherence", "VISION", "RecipeQA_ImageCoherence"]

mantis = ["NLVR2_Mantis", "QBench", "Mantis", "BLINK", "MVBench"]
ood = ["RAVEN", "BAPPS", "HQ-Edit"]


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
        answer = answer.strip(")")
        answer = answer.strip("(")
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
            # assert gt_ans != ''
            if gt_ans == "":
                print("wrong gt_ans", gt_ans)
                s = 0
            elif pred_ans == "":
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

        # if ":" in pred_ans:
        #     a1, a2 = pred_ans.split(":")
        #     a1, a2 = a1.strip(), a2.strip()
        #     if len(a1) == 1 and a1[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        #         pred_ans = a1
        #     if len(a2) == 1 and a2[0] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        #         pred_ans = a2
        if ":" in pred_ans:
            a_list = pred_ans.split(":")
            a_list = [a.strip() for a in a_list]
            for a in a_list:
                if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    pred_ans = a

        if pred_ans == gt_ans:
            return 1
        else:
            return 0

    def process_sample(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        # for i in range(len(sample['choice_list'])):
        #     sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            # if 'choice_list' not in sample.keys():
            #     print(sample)
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

            # if ":" in pred_ans:
            #     a1, a2 = pred_ans.split(":")
            #     a1, a2 = a1.strip(), a2.strip()
            #     if len(a1) == 1 and a1[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            #         pred_ans = a1
            #     if len(a2) == 1 and a2[0] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            #         pred_ans = a2

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a

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

        preds = preds_all_dict[dataset]
        question_type = preds[0]["question_type"]

        if question_type == "open-ended":
            eval_result, eval_list = E.evaluate_rouge(preds)

        elif question_type == "multi-choice" or dataset == "Mantis":
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
        json.dump(eval_result_list, f, indent=4)

    with open(os.path.join(args.result_dir, "eval_dataset_details.json"), "w") as f:
        json.dump(eval_result_list_detail, f, indent=4)

    eval_cat_list = dict()
    print()

    # multi_modal_dialogue
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in multi_modal_dialogue:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["multi_modal_dialogue"] = score
        print("multi_modal_dialogue", end=":  ")
        print("{:.2f}".format(100 * score))

    # visual_story_telling_list
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in visual_story_telling_list:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["visual_story_telling_list"] = score
        print("visual_story_telling_list", end=":  ")
        print("{:.2f}".format(100 * score))

    # visual_relation_inference
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in visual_relation_inference:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["visual_relation_inference"] = score
        print("visual_relation_inference", end=":  ")
        print("{:.2f}".format(100 * score))

    # multi_modal_cloze
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in multi_modal_cloze:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["multi_modal_cloze"] = score
        print("multi_modal_cloze", end=":  ")
        print("{:.2f}".format(100 * score))

    # knowledge_grounded_qa
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in knowledge_grounded_qa:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["knowledge_grounded_qa"] = score
        print("knowledge_grounded_qa", end=":  ")
        print("{:.2f}".format(100 * score))

    # text_rich_images_qa
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in text_rich_images_qa:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["text_rich_images_qa"] = score
        print("text_rich_images_qa", end=":  ")
        print("{:.2f}".format(100 * score))

    # multi_image_reasoning
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in multi_image_reasoning:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["multi_image_reasoning"] = score
        print("multi_image_reasoning", end=":  ")
        print("{:.2f}".format(100 * score))

    # mantis
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in mantis:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["mantis"] = score
        print("mantis", end=":  ")
        print("{:.2f}".format(100 * score))

    # ood
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in ood:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["ood"] = score
        print("ood", end=":  ")
        print("{:.2f}".format(100 * score))

    with open(os.path.join(args.result_dir, "eval_cat.json"), "w") as f:
        json.dump(eval_cat_list, f, indent=4)
