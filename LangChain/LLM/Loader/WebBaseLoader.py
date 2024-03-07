# !pip install bs4
# !pip install chromadb
# !pip install pysqlite3-binary
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load blog post
loader = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)
