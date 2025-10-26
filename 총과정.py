import requests

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
