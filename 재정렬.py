import json

# 1. 기존 JSON 파일 읽기
with open("yuchan_dict.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 알파벳 순으로 정렬
sorted_data = dict(sorted(data.items()))

# 3. 정렬된 내용을 같은 파일에 저장
with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)
