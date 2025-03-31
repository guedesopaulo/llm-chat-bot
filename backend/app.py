from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_chain import answer_question, get_memory, get_prompt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Libera requisições vindas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ou ["*"] para permitir tudo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(message: Message):
    answer = answer_question(message.text)
    return {"response": answer}

@app.post("/debug_prompt")
def debug_prompt(message: Message):
    prompt = get_prompt(message.text)
    return {"prompt": prompt}

@app.get("/memory")
def memory():
    return {"history": get_memory()}
