import os
import dropbox
import pandas as pd
import spacy
import memcache
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

DROPBOX_API_TOKEN = os.getenv("DROPBOX_API_TOKEN")

if not DROPBOX_API_TOKEN:
    raise ValueError("Dropbox API 토큰이 환경 변수에 설정되지 않았습니다.")

dbx = dropbox.Dropbox(DROPBOX_API_TOKEN)

CACHE_SERVER = '127.0.0.1:11211'
CACHE = memcache.Client([CACHE_SERVER], debug=0)

nlp = spacy.load('en_core_web_sm')

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def get_account_info():
    try:
        account_info = dbx.users_get_current_account()
        return account_info
    except dropbox.exceptions.AuthError:
        print("인증 오류: 액세스 토큰을 확인하세요.")
        return None

def collect_folder_list(selected_folder: str) -> List[str]:
    try:
        result = dbx.files_list_folder(selected_folder, recursive=True)
        files = [entry.path_lower for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)]
        print(f"수집된 파일 수: {len(files)}")
        return files
    except Exception as e:
        print(f"폴더 목록 수집 중 오류 발생: {e}")
        return []

def read_file_content(file_path: str) -> str:
    try:
        metadata, res = dbx.files_download(file_path)
        content = res.content.decode('utf-8')
        return content
    except Exception as e:
        print(f"파일 읽기 중 오류 발생 ({file_path}): {e}")
        return ""

def clean_text(text: str) -> str:
    return text.lower()

def split_text(text: str) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def remove_duplicates(text_list: List[str]) -> List[str]:
    return list(set(text_list))

def preprocess_files(file_paths: List[str]) -> List[str]:
    preprocessed_data = []
    for file_path in file_paths:
        cached = CACHE.get(file_path)
        if cached:
            content = cached
            print(f"캐시에서 가져온 파일: {file_path}")
        else:
            content = read_file_content(file_path)
            CACHE.set(file_path, content)
            print(f"새로 읽은 파일: {file_path}")
        cleaned_text = clean_text(content)
        split_texts = split_text(cleaned_text)
        preprocessed_data.extend(split_texts)
    unique_data = remove_duplicates(preprocessed_data)
    print(f"중복 제거 후 텍스트 수: {len(unique_data)}")
    return unique_data

def load_preprocessed_data(file_path: str) -> List[str]:
    try:
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()
        return texts
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return []

def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def ensure_data_directory():
    if not os.path.exists('data'):
        os.makedirs('data')
        print("data 디렉토리가 생성되었습니다.")
    else:
        print("data 디렉토리가 이미 존재합니다.")

def main():
    ensure_data_directory()
    account_info = get_account_info()
    if account_info:
        print(f"사용자 이름: {account_info.name.display_name}")
    else:
        print("Dropbox 계정 정보를 가져오는데 실패했습니다.")
        return
    selected_folder = "/gwxx"
    file_paths = collect_folder_list(selected_folder)
    if not file_paths:
        print("수집된 파일이 없습니다. 폴더 경로를 확인하세요.")
        return
    preprocessed_texts = preprocess_files(file_paths)
    preprocessed_file = 'data/preprocessed_data.csv'
    df = pd.DataFrame({'text': preprocessed_texts})
    df.to_csv(preprocessed_file, index=False)
    print(f"Preprocessing 완료, 총 텍스트 수: {len(preprocessed_texts)}")
    texts = load_preprocessed_data(preprocessed_file)
    if not texts:
        print("전처리된 텍스트가 없습니다. 전처리 스크립트를 먼저 실행하세요.")
        return
    embeddings = embed_texts(texts)
    embeddings_file = 'data/embeddings.npy'
    np.save(embeddings_file, embeddings)
    print(f"Embedding 완료. 총 벡터 수: {embeddings.shape[0]}")

if __name__ == "__main__":
    main()