from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag.rag_chain import answer_question

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(message: Message):
    answer = answer_question(message.text)
    return {"response": answer}
