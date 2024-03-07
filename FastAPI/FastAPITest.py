# !pip install fastapi
# !pip install uvicorn
# !uvicorn /home/cdsw/LangChain/FastAPI/FastAPITest:app --reload --host 0.0.0.0 --port 8001

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
#async def root():
def root():
  return {"message" : "Hello World"}

@app.get("/home")
def home():
  return {"message" : "home"}