import pandas as pd
import json
import numpy as np

# Parquet 파일 읽기
df = pd.read_parquet("d.parquet")

# NumPy 타입을 파이썬 기본 타입으로 변환
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# JSONL 파일로 저장
with open("output_file.jsonl", "w", encoding="utf-8") as f:
    for record in df.to_dict(orient="records"):
        json_record = json.dumps(record, default=convert_numpy_types, ensure_ascii=False)
        f.write(json_record + "\n")

import json

input_file = "output_file.jsonl"
output_file = "merged_file.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        # instruction과 input 합치기
        merged_prompt = record["instruction"]
        if record.get("input"):
            merged_prompt += "\n" + record["input"]
        
        new_record = {
            "input": merged_prompt,
            "output": record["output"],
            "meta": {"형태": "질의응답_및_일상대화"}
        }
        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
