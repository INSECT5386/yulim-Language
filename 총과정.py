import requests
import json
import pandas as pd

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/verbs.json?download=true', 'verbs.json')
download_file('https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/others.json?download=true', 'others.json')
download_file('https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/nouns.json?download=true', 'nouns.json')
download_file('https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/adjectives.json?download=true', 'adjectives.json')

CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def get_chosung(word):
    result = ""
    for char in word:
        if '가' <= char <= '힣':
            code = ord(char) - 0xAC00
            cho = code // 588
            result += CHOSUNG[cho]
        else:
            result += char
    return result

# JSON 불러오기
with open("verbs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("verbs_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)




with open("others.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("others_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)


with open("nouns.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("nouns_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)


with open("adjectives.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("adjectives_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)


# 1. JSON 파일 읽기
with open("adjectives_re.json", "r", encoding="utf-8") as f:
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
df.to_excel("adjectives.xlsx", index=False)


with open("nouns_re.json", "r", encoding="utf-8") as f:
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
df.to_excel("nouns.xlsx", index=False)


with open("others_re.json", "r", encoding="utf-8") as f:
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
df.to_excel("others.xlsx", index=False)

with open("verbs_re.json", "r", encoding="utf-8") as f:
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
df.to_excel("verbs.xlsx", index=False)
