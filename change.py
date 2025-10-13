import json
import pandas as pd

# 1. JSON 파일 읽기
with open("nukil_dict_re.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 단어별로 표 형태로 변환 (변형/형태 제외)
rows = []
for word, info in data.items():
    품사 = info.get("품사", "")
    뜻 = info.get("뜻", "")
    설명 = info.get("설명", "")
    예문 = "\n".join(info.get("예문", [])) if "예문" in info else ""
    
    # 변형/형태는 완전히 무시하고 포함하지 않음
    rows.append([word, 품사, 뜻, 설명, 예문])

# 3. 데이터프레임 생성
df = pd.DataFrame(rows, columns=["단어", "품사", "뜻", "설명", "예문"])

# 4. 엑셀로 저장
df.to_excel("nukil_dictionary.xlsx", index=False)
print("✅ nukil_dictionary.xlsx 파일 생성 완료!")
