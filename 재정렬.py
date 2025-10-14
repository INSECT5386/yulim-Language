import json

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
with open("yuchan_dict.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}

# 뜻 있는 항목 ㄱㄴㄷ 순 정렬
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))

# 뜻 없는 항목 뒤로 붙이기
sorted_data = {**sorted_with_meaning, **without_meaning}

# 저장
with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)
