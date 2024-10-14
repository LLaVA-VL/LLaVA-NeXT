import json

with open('/Users/zhangyuanhan/Downloads/videomme_72B_mix_32f.json') as f:
    data = json.load(f)

counter = 0
save_json = {}
for _ in data["logs"]:

    videomme_percetion_score = _["videomme_percetion_score"]
    question = _["doc"]["question"] 
    options = _["doc"]["options"]
    url = _["doc"]["url"]   
    duration = _["doc"]["duration"]
    doc_id = _["doc_id"] 
    if duration != "short":
        continue
    
    # import pdb; pdb.set_trace()
    if videomme_percetion_score["answer"] != videomme_percetion_score["pred_answer"]:
        # if counter < 112:
        #     counter += 1
        #     continue
        print("question:", question)
        print("options:", options)
        print("answer:", videomme_percetion_score["answer"])
        print("pred_answer:", videomme_percetion_score["pred_answer"])
        print("url:", url)
        print("doc_id:", doc_id)    
        # if doc_id == 508:
        #     break
        counter += 1
        save_json[doc_id] = {"question": question, "options": "\n".join(options), "gt_answer": videomme_percetion_score["answer"], "pred_answer": videomme_percetion_score["pred_answer"], "url": url}


print(counter)

import pandas as pd

# Load the Excel file and fetch all sheet names
excel_path = '/Users/zhangyuanhan/Downloads/错题.xlsx'  # Update with the correct path
xls = pd.ExcelFile(excel_path)

# Prepare a dictionary to store dataframes for each sheet
updated_sheets = {}

# Iterate over all sheets in the Excel file
for sheet_name in xls.sheet_names:
    print(f"Processing sheet: {sheet_name}")
    if sheet_name in ["总计", "其他"]:
        continue
    
    # Load the sheet into a dataframe
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Iterate over rows in the dataframe and update metadata
    for index, row in df.iterrows():
        doc_id = row['doc_id']  # Assuming there's a 'doc_id' column in the Excel file

        if doc_id in save_json:
            df.at[index, 'question'] = save_json[doc_id]["question"]
            df.at[index, 'options'] = save_json[doc_id]["options"]
            df.at[index, 'gt_answer'] = save_json[doc_id]["gt_answer"]
            df.at[index, 'pred_answer'] = save_json[doc_id]["pred_answer"]
            df.at[index, 'url'] = save_json[doc_id]["url"]

    # Store the updated dataframe in the dictionary
    updated_sheets[sheet_name] = df

# Now, write all the updated sheets to a new Excel file
output_path = '/Users/zhangyuanhan/Downloads/updated_excel_file.xlsx'  # Update with the correct path

with pd.ExcelWriter(output_path, mode='w') as writer:
    for sheet_name, df in updated_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel file has been updated and saved to: {output_path}")





