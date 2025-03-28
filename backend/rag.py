from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from .models.bedrock import load_bedrock_chat
# ou use: from .models.gemma import load_gemma_chat

# === SETUP ===
processor, model = load_bedrock_chat()
memory = ConversationBufferMemory(return_messages=True)

# === RETRIEVAL ===
def retrieve_context(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    db = FAISS.load_local("vector_store/index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in docs)

# === GERAÇÃO COM MEMÓRIA ===
def generate_response(context: str, question: str):
    # Adiciona a pergunta atual na memória
    memory.chat_memory.add_user_message(f"{context}\n\nQuestion: {question}")

    # Pega histórico formatado
    chat_history = memory.chat_memory.messages

    # Monta o prompt no formato usado pelo modelo
    messages = [{"role": "system", "content": [{"type": "text", "text": "Você é um assistente de vendas da nossa empresa, seja gentil e responda como um humano. Foque em responder sobre a pergunta. Evite alucinações fora da pergunta"}]}]
    for m in chat_history:
        role = "user" if m.type == "human" else "assistant"
        messages.append({
            "role": role,
            "content": [{"type": "text", "text": m.content}]
        })

    prompt = processor(messages)

    try:
        response = model.invoke(prompt)
        memory.chat_memory.add_ai_message(response)
        print("🧾 Resposta do modelo:", response)
        return response.strip()
    except Exception as e:
        print("⚠️ Erro ao gerar resposta:", e)
        return "Houve um erro ao gerar a resposta."

# === FUNÇÃO PRINCIPAL DO FASTAPI ===
def answer_question(query: str):
    context = retrieve_context(query)
    return generate_response(context, query)
