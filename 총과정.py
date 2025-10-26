import requests
import json

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

with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)




with open("others.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)


with open("nouns.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)


with open("adjectives.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)
