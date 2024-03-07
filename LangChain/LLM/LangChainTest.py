# pip install transformers
# pip install langchain
# pip install torch
# pip install optimum
# pip install auto-gptq
# and restart session
# add config.json file as "disable_exllama": true if you have only single gpu.

import os
os.environ['HF_HOME'] = '/home/cdsw/models'

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GPTQConfig

# HuggingFace Model ID
model_id = 'TheBloke/Mistral-7B-OpenOrca-GPTQ'
access_token = "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"

quantization_config = GPTQConfig(bits=4, disable_exllama=True)

# Load the model stored in models/llm-model
print(f"Starting to load the LLM model")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False, device_map="auto", revision="main", token=access_token)

print(f"Starting to load the LLM tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=access_token)

print(f"Finished loading the model and tokenizer")

print(f"Creating generator with transformers pipeline")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=500)

print(f"Creating HuggingFacePipeline")
mistral_llm = HuggingFacePipeline(pipeline=generator)


# 템플릿
template = """질문: {question}

답변: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# LLM Chain 객체 생성
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

question = "캐나다의 수도와 대한민국의 수도까지의 거리는 어떻게 돼?"
print(llm_chain.run(question=question))