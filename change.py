import json
import pandas as pd

# 1. JSON 파일 읽기
with open("nukil_dict_re.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 단어별로 표 형태로 변환
rows = []
for word, info in data.items():
    품사 = info.get("품사", "")
    뜻 = info.get("뜻", "")
    설명 = info.get("설명", "")
    예문 = "\n".join(info.get("예문", []))
    
    # 동사일 경우 변형 표시
    변형 = ""
    if "변형" in info:
        변형_list = [f"{k}: {v}" for k, v in info["변형"].items()]
        변형 = "\n".join(변형_list)
    
    # 형용사일 경우 명사형/부사형 표시
    if "형태" in info:
        변형_list = [f"{k}: {v}" for k, v in info["형태"].items()]
        변형 = "\n".join(변형_list)
    
    rows.append([word, 품사, 뜻, 설명, 변형, 예문])

# 3. 데이터프레임 생성
df = pd.DataFrame(rows, columns=["단어", "품사", "뜻", "설명", "변형/형태", "예문"])

# ✅ 최신 pandas에서는 이렇게
df.to_excel("nukil_dictionary.xlsx", index=False)
print("✅ nukile_dictionary.xlsx 파일 생성 완료!")
