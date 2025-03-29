from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings
import faiss

# 1. Carregar os documentos da sua loja de cafés
documents = SimpleDirectoryReader("backend/data/docs").load_data()

# 2. Configurar o modelo de linguagem (LLM) usando o Ollama com o modelo Llama 3.1
llm = Ollama(model="llama3", request_timeout=30.0)

# 3. Configurar o modelo de embeddings usando o HuggingFace
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Criar o índice FAISS para armazenamento vetorial
dimension = 384  # Dimensão do modelo de embeddings
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 5. Configurar o contexto global com a LLM e os embeddings
Settings.llm = llm
Settings.embed_model = embedding_model

# 6. Criar o índice de documentos com o armazenamento vetorial
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# 7. Configurar a memória de conversação com limite de tokens
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

# 8. Criar o mecanismo de chat com memória e prompt personalizado
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "Você é um assistente especializado em cafés especiais da loja Serenatto. "
        "Responda às perguntas dos clientes com base nas informações fornecidas nos documentos. "
        "Se não souber a resposta, admita que não sabe. Seja claro, educado e útil."
    ),
)

# 9. Função para responder às perguntas dos clientes
def answer_question(question: str) -> str:
    response = chat_engine.chat(question)
    return response.response

# Função opcional para expor o histórico da memória

def get_memory():
    return memory.get_all()