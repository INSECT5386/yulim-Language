import json

try:
    with open("nukil_dict.json", "r", encoding="utf-8") as f:
        nukil_dict = json.load(f)
except json.JSONDecodeError as e:
    print("JSON 파싱 오류:", e)

# 단어 정렬
root_words = sorted(nukil_dict.keys())

# txt로 저장
with open("nukil_full_words.txt", "w", encoding="utf-8") as f:
    for word in root_words:
        pos = nukil_dict[word].get("품사", "알수없음")
        meaning = nukil_dict[word].get("뜻", "")
        f.write(f"{word} ({pos}): {meaning}\n")
        
        # 변형 저장
        variants = nukil_dict[word].get("변형", {})
        if variants:
            for v_key, v_value in variants.items():
                f.write(f"    [{v_key}]: {v_value}\n")
        
        # 형용사/부사 명사형/부사형 저장
        forms = nukil_dict[word].get("형태", {})
        if forms:
            for f_key, f_value in forms.items():
                f.write(f"    [{f_key}]: {f_value}\n")

print(f"총 {len(root_words)}개의 단어와 변형/형태가 nukil_full_words.txt에 저장되었습니다.")

print("단어 수:", len(nukil_dict))
print("샘플 키 몇 개:", list(nukil_dict.keys())[:5])
