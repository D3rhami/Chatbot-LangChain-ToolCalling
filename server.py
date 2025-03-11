# Standard library imports
import os
import random

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
model_name = "gpt-4o-mini"

llm = ChatOpenAI(model=model_name, base_url="https://api.avalai.ir/v1", api_key=API_KEY)

class Query(BaseModel):
    prompt: str

@app.get("/user_id")
async def get_user_id():
    user_id = random.randint(1000, 9999)
    print("user id is", user_id)
    return {"user_id": user_id}

@app.post("/process")
async def process_query(query: Query):
    print("query is", query)
    try:
        response = llm.invoke(query.prompt).content
        return {"response": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
