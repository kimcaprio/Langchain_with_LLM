#https://colab.research.google.com/drive/1PALBOJ-vXgKe3LOIbKoqeD8kuZYx_Lnr#scrollTo=ltRw8HCmmECS

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

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



question = "삼성전자 갤럭시 S24는 어떨 예정이야?"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#model_id = "yanolja/Bookworm-10.7B-v0.4-DPO"
#model_id = "Raphael21/Raphael21-SOLAR-10.7B"
model_id = "ENERGY-DRINK-LOVE/eeve_dpo-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=True,
    max_new_tokens=300,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)

unique_docs