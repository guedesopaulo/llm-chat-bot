from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from .models.gemma import load_gemma_chat

import torch

# Carrega o processor e o modelo LLM
processor, model = load_gemma_chat()

# Monta o prompt estruturado no estilo chat
def build_messages(context: str, question: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{context}\n\nQuestion: {question}"}]
        }
    ]

# Busca os documentos relevantes
def retrieve_context(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vector_store/index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in docs)

def generate_response(context: str, question: str):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{context}\n\nQuestion: {question}"}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            generated = output[0][input_len:]
            decoded = processor.decode(generated, skip_special_tokens=True)
            print("🧾 Resposta do modelo:", decoded)
            return decoded.strip()
        except Exception as e:
            print("⚠️ Erro ao gerar resposta:", e)
            return "Houve um erro ao gerar a resposta."


# Endpoint principal chamado pelo app.py
def answer_question(query: str):
    context = retrieve_context(query)
    return generate_response(context, query)
