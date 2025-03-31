from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

# 1. Carregar os documentos da sua loja de cafés
documents = SimpleDirectoryReader("backend/data/docs").load_data()

# 2. Configurar o modelo de linguagem (LLM) usando o Ollama com o modelo Llama 3.1
llm = Ollama(model="gemma3:12b", request_timeout=30.0)

# 3. Configurar o modelo de embeddings usando o HuggingFace
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Criar o índice FAISS e o vetor store
dimension = 384  # Dimensão do modelo de embeddings
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 5. Criar contexto de armazenamento e indexar os documentos
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, embed_model=embedding_model, storage_context=storage_context
)

# 6. Salvar o índice completo (inclui textos e índice FAISS)
index.storage_context.persist("backend/data/index")

# 7. Configurar o contexto global
Settings.llm = llm
Settings.embed_model = embedding_model

# 8. Configurar memória de conversação
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

# 9. Criar o chat engine com prompt customizado
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "Você é um agente especializado em cafés especiais da loja Serenatto.\n\n"
        "=== CONTEXTO ===\n"
        "As informações que você tem acesso vêm de documentos da empresa.\n"
        "Use-as para responder de forma precisa, educada e profissional.\n"
        "Se não souber a resposta, diga que não sabe.\n"
        "Perguntas sobre temas fora do assunto ou sensíveis devem ser respondidas como 'isso foge ao assunto da loja'.\n\n"
        "=== HISTÓRICO DA CONVERSA ===\n"
        "Você verá a seguir o histórico da conversa com o cliente. Use isso para manter o contexto da conversa, "
        "mas sempre responda à nova pergunta diretamente.\n\n"
        "=== NOVA PERGUNTA ===\n"
        "Responda de forma objetiva, educada e clara com base no histórico e nos documentos disponíveis."
    )
)

# 10. Funções principais de resposta e memória
def answer_question(question: str) -> str:
    response = chat_engine.chat(question)
    return response.response

def get_memory():
    return memory.get_all()
