from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_chain import answer_question, get_memory

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(message: Message):
    answer = answer_question(message.text)
    return {"response": answer}

@app.get("/chat/history")
def history():
    return get_memory()
