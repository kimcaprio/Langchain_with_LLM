#from langchain_community.document_loaders import TextLoader
#
#loader = TextLoader("./README.md")
#loader.load()



# 필요한 라이브러리와 Langchain 컴포넌트를 임포트합니다.
import fitz  # PyMuPDF
import os
from langchain.llms import TextLoader, TextSplitter  # Langchain 컴포넌트 (예시)
from milvus import Milvus, DataType  # Milvus 클라이언트



# PyMuPDF를 사용하여 PDF 파일에서 텍스트를 추출하는 함수
def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Milvus 클라이언트 초기화 및 연결
milvus_client = Milvus(host='localhost', port='19530')

# 벡터 데이터를 Milvus에 저장하는 함수
def save_vectors_to_milvus(vectors, collection_name="pdf_vector_collection"):
    # 컬렉션 생성 (이미 존재하는 경우 생략 가능)
    collection_params = {
        "fields": [
            {"name": "embeddings", "type": DataType.FLOAT_VECTOR, "params": {"dim": 768}},
        ],
        "segment_row_limit": 4096,
        "auto_id": True
    }
    if not milvus_client.has_collection(collection_name):
        milvus_client.create_collection(collection_name, collection_params)
    
    # 벡터 데이터 저장
    milvus_client.insert(collection_name, records=vectors)
    milvus_client.flush([collection_name])  # 데이터를 디스크에 강제로 저장
  
# 특정 디렉토리 내의 모든 .pdf 파일을 찾아서 처리하는 함수를 정의합니다.
def process_pdf_files(directory_path):
    # 디렉토리 내의 모든 파일을 순회합니다.
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            
            # PDF 파일을 텍스트로 변환합니다.
            text = convert_pdf_to_text(pdf_path)
            
            # TextLoader를 사용해 텍스트를 로드합니다.
            text_loader = TextLoader()
            loaded_text = text_loader.load(text)
            
            # TextSplitter를 사용해 텍스트를 토큰으로 나눕니다.
            text_splitter = TextSplitter()
            tokens = text_splitter.split(loaded_text)
            
            # Embedder를 사용해 토큰을 벡터로 변환합니다.
            embedder = Embedder()
            vectors = embedder.embed(tokens)
            
            # VectorDB에 벡터를 저장합니다.
            vector_db = VectorDB()
            vector_db.save(vectors)

            
