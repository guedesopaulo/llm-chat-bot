from langchain_core.messages import HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from backend.models.bedrock import load_bedrock_chat

# Inicializa o modelo Bedrock e o "processor" de prompt
processor, model = load_bedrock_chat()

# Memória da conversa
memory = ConversationBufferMemory(return_messages=True)

# Função de recuperação de contexto via FAISS

def retrieve_context(query: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vector_store/index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in docs)

# Monta o prompt com contexto moderno

def build_prompt(history, context, query):
    prompt = """### INSTRUÇÃO
Você é um assistente de vendas. Responda com clareza e detalhes, baseando-se nos documentos abaixo.
Se não souber, diga que não sabe. Seja claro, educado e útil.

### CONVERSA
"""
    for msg in history:
        if isinstance(msg, HumanMessage):
            prompt += f"Usuário: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt += f"Assistente: {msg.content}\n"

    prompt += f"""
### CONTEXTO
As informações a seguir vêm de documentos internos em formato de perguntas e respostas. Use-as como base confiável para responder.
{context}

### PERGUNTA
Usuário: {query}

### RESPOSTA
Assistente:"""
    return prompt

# Gera resposta do modelo

def generate_response(context: str, query: str):
    memory.chat_memory.add_user_message(query)
    history = memory.chat_memory.messages
    prompt = build_prompt(history[:-1], context, query)

    try:
        response = model.invoke(prompt)  # sem stop
        memory.chat_memory.add_ai_message(response)
        print("\U0001f4be Resposta do modelo:", response)
        return response.strip()
    except Exception as e:
        print("⚠️ Erro ao gerar resposta:", e)
        return "Houve um erro ao gerar a resposta."

# Função principal

def answer_question(query: str):
    context = retrieve_context(query)
    return generate_response(context, query)

# Opcional: expor histórico para depuração ou front

def get_chat_history():
    return [
        {"type": m.type, "content": m.content}
        for m in memory.chat_memory.messages
    ]
