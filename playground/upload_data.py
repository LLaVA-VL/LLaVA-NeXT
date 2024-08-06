from datasets import Dataset, Features, Value, ClassLabel, Sequence, Image
import json
import PIL.Image as pil_image
from io import BytesIO
from tqdm import tqdm

json_paths = [
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/mavis_math_metagen_87358.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/mavis_math_rule_geo_100000.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/k12_printing_train_256646.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/iiit5k_annotations_2000.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/hme100k_train_clean_74502.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/ai2d_azuregpt_detailed_understanding_4874.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/infographic_vqa_4404.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/infographic_azuregpt4v_1992.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/lrv_chart_1787.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/lrv_normal_gpt4v_filtered_10500.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/scienceqa_nona_context_19218.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/allava_instruct_vflan4v_20000.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/allava_instruct_laion4v_50000.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/textocr_gpt4v_train_converted_25114.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/ai2d_train_internvl_single_12413.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/textcaps_train_21952.json",
    # "/mnt/bn/vl-research/data/llava_instruct/ureader_new/ureader_qa_sft.json",
    # "/mnt/bn/vl-research/data/llava_instruct/ureader_new/ureader_cap_sft.json",
    # "/mnt/bn/vl-research/data/llava_instruct/ureader_new/ureader_ie_sft.json",
    # "/mnt/bn/vl-research/data/llava_instruct/ureader_new/ureader_kg_sft.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/vision_flan_filtered_186070.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/mathqa_29837.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/geo3k_2101.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/geo170k_qa_converted_67833.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/geo170k_align_converted_60252.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sharegpt4v-coco-50k.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sharegpt4v-knowledge-2k.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sharegpt4v-llava-30k.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sharegpt4v-sam-20k.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_CLEVR-Math_5290.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_FigureQA_17597.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_Geometry3K_9734.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_GeoQA+_17172.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_GEOS_508.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_IconQA_22599.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_MapQA_5235.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_PMC-VQA_35958.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_Super-CLEVR_8652.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_TabMWP_22462.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_UniGeo_11959.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/MathV360K_VizWiz_6614.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/magpie_pro_qwen2_72b_st_300000_sp_token_fltd_299992.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/magpie_pro_l3_80b_st_300000.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/magpie_pro_l3_80b_mt_300000_sp_token_fltd_299998.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/image_textualization_dataset_filtered.json",
    # "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/cambrian_filtered_gpt4vo_sp_token_fltd_max10k.json",
    "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sharegpt4o_dataset.jsonl",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/ai2d_llava_format_2434.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/aokvqa_16539_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/chart2text_26961.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/chartqa_18265_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/clevr_70000_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/diagram_image_to_text_300.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/dvqa_200000_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/figureqa_100000_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/geomverse_9303.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/hateful_memes_8500_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/hitab_2500_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/iam_5663.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/raven_42000.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/iconqa_llava_format_27307.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/infographic_vqa_2118_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/intergps_1280_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/mapqa_37417_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/multihiertt_7619.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/rendered_text_10000.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/robut_sqa_8514.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/robut_wikisql_74989.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/robut_wtq_38246_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/screen2words_15730.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/scienceqa_llava_format_4976.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/tabmwp_22722.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/tallyqa_98680_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/st_vqa_17247_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/tqa_llava_format_27307.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/visual7w_llava_format_14366.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/visualmrc_3027.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/vqarad_313_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/vsr_2157_llava_format.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/vistext_9969.json",
    "/mnt/bn/vl-research/data/llava_instruct/cauldron/websight_10000.json"
]

short_names = [
    # "mavis_math_metagen",
    # "mavis_math_rule_geo",
    # "k12_printing",
    # "iiit5k",
    # "hme100k",
    # "ai2d(gpt4v)",
    # "infographic_vqa",
    # "infographic(gpt4v)",
    # "lrv_chart",
    # "lrv_normal(filtered)",
    # "scienceqa(nona_context)",
    # "allava_instruct_vflan4v",
    # "allava_instruct_laion4v",
    # "textocr(gpt4v)",
    # "ai2d(internvl)",
    # "textcaps",
    # "ureader_qa", # need to re-upload
    # "ureader_cap", # need to re-upload
    # "ureader_ie", # need to re-upload
    # "ureader_kg", # need to re-upload
    # "vision_flan(filtered)",
    # "mathqa",
    # "geo3k",
    # "geo170k(qa)",
    # "geo170k(align)",
    # "sharegpt4v(coco)",
    # "sharegpt4v(knowledge)",
    # "sharegpt4v(llava)",
    # "sharegpt4v(sam)",
    # "CLEVR-Math(MathV360K)",
    # "FigureQA(MathV360K)",
    # "Geometry3K(MathV360K)",
    # "GeoQA+(MathV360K)",
    # "GEOS(MathV360K)",
    # "IconQA(MathV360K)",
    # "MapQA(MathV360K)",
    # "PMC-VQA(MathV360K)",
    # "Super-CLEVR(MathV360K)",
    # "TabMWP(MathV360K)",
    # "UniGeo(MathV360K)",
    # "VizWiz(MathV360K)",
    # "magpie_pro(qwen2_72b_st)",
    # "magpie_pro(l3_80b_st)",
    # "magpie_pro(l3_80b_mt)",
    # "image_textualization(filtered)",
    # "cambrian(filtered_gpt4vo)", # need to re-upload
    "sharegpt4o",
    "ai2d(cauldron,llava_format)",
    "aokvqa(cauldron,llava_format)",
    "chart2text(cauldron)",
    "chartqa(cauldron,llava_format)",
    "clevr(cauldron,llava_format)",
    "diagram_image_to_text(cauldron)",
    "dvqa(cauldron,llava_format)",
    "figureqa(cauldron,llava_format)",
    "geomverse(cauldron)",
    "hateful_memes(cauldron,llava_format)",
    "hitab(cauldron,llava_format)",
    "iam(cauldron)",
    "raven(cauldron)",
    "iconqa(cauldron,llava_format)",
    "infographic_vqa_llava_format",
    "intergps(cauldron,llava_format)",
    "mapqa(cauldron,llava_format)",
    "multihiertt(cauldron)",
    "rendered_text(cauldron)",
    "robut_sqa(cauldron)",
    "robut_wikisql(cauldron)",
    "robut_wtq(cauldron,llava_format)",
    "screen2words(cauldron)",
    "scienceqa(cauldron,llava_format)",
    "tabmwp(cauldron)",
    "tallyqa(cauldron,llava_format)",
    "st_vqa(cauldron,llava_format)",
    "tqa(cauldron,llava_format)",
    "visual7w(cauldron,llava_format)",
    "visualmrc(cauldron)",
    "vqarad(cauldron,llava_format)",
    "vsr(cauldron,llava_format)",
    "vistext(cauldron)",
    "websight(cauldron)"
]

def upload_data(json_path, short_name):
    def gen():
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                data = [json.loads(line) for line in f]
        else:
            with open(json_path, "r") as f:
                data = json.load(f)

        preview_index = 5
        idx = 0
        for item in tqdm(data):
            if preview_index > 0:
                preview_index -= 1
                print(item)
                continue

            try:
                if "image" in item:
                    image_path = f"/mnt/bn/vl-research/data/llava_data/{item['image']}"
                    try:
                        with open(image_path, "rb") as img_file:
                            image = pil_image.open(BytesIO(img_file.read()))
                    except:
                        print(f"Failed to load image {item['image']}")
                        continue
                else:
                    image = None

                item_id = item["id"] if "id" in item else f"{idx:06d}"
                yield {"id": item_id, "image": image, "conversations": item["conversations"], "data_source": short_name}
                idx += 1
                
            except Exception as e:
                print(e)
                continue


    hf_dataset = Dataset.from_generator(generator=gen, num_proc=32)
    hf_dataset.push_to_hub("lmms-lab/LLaVA-OneVision-Data", config_name=short_name, split="train")

for json_path, short_name in zip(json_paths, short_names):
    upload_data(json_path, short_name)