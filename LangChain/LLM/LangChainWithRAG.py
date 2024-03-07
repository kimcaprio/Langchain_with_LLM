# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install pypdf
# !pip install sentence-transformers
# !pip install faiss-gpu
  
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "yanolja/Bookworm-10.7B-v0.4-DPO"

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

prompt_template = """
### [INST]
Instruction: Answer the question based on your knowledge.
Here is context to help:

{context}

### QUESTION:
{question}

[/INST]
 """

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=llm, prompt=prompt)




# embedding & insert vector DB
loaders = [
    PyPDFLoader("/home/cdsw/LangChain/data/1.디지털자산+동향+보고서_202201.pdf"),
    PyPDFLoader("/home/cdsw/LangChain/data/비트코인의 현황 및 시사점.pdf"),
]

docs = []
for loader in loaders:
  docs.extend(loader.load_and_split())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(texts, hf)
retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k': 3}
                        )


rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

import warnings
warnings.filterwarnings('ignore')

result = rag_chain.invoke("암호화 화폐 성장 가능성에 대해 알려주세요.")

for i in result['context']:
    print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")

print(f"\n답변: {result['text']}")