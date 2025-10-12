import pandas as pd
import json

# CSV 파일 읽기
df = pd.read_csv("input_file.csv")  # input_file.csv에 input, output 컬럼 있다고 가정

# JSONL 파일로 저장
with open("output_file1.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = {
            "input": row["input"],   # CSV의 input 컬럼을 prompt로
            "output": row["output"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
