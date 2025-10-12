import json

# JSON 파일 불러오기
with open("nukil_dict.json", "r", encoding="utf-8") as f:
    nukil_dict = json.load(f)

# 단어 정렬
root_words = sorted(nukil_dict.keys())

# txt로 저장
with open("nukil_root_words.txt", "w", encoding="utf-8") as f:
    for word in root_words:
        pos = nukil_dict[word].get("품사", "알수없음")
        meaning = nukil_dict[word].get("뜻", "")
        f.write(f"{word} ({pos}): {meaning}\n")

print(f"총 {len(root_words)}개의 단어와 뜻이 nukil_root_words.txt에 저장되었습니다.")
