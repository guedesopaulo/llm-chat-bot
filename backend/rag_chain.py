from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
import faiss

# Configurações personalizáveis
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 512
TOP_K = 4
INDEX_PATH = "backend/data/index"

splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = SimpleDirectoryReader("backend/data/docs").load_data()
nodes = splitter.get_nodes_from_documents(documents)

llm = Ollama(model="gemma3:12b", request_timeout=30.0)

embedding_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

dimension = 1024  
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, embed_model=embedding_model, storage_context=storage_context)

index.storage_context.persist(INDEX_PATH)

Settings.llm = llm
Settings.embed_model = embedding_model

# 8. Configurar memória de conversação
memory = ChatMemoryBuffer.from_defaults(token_limit=512)

SYSTEM_PROMPT = (
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
    "Responda de forma objetiva, educada e clara com base no histórico e nos documentos disponíveis, parece um ser humano."
)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=SYSTEM_PROMPT,
    retriever_kwargs={"similarity_top_k": TOP_K}
)

def answer_question(question: str) -> str:
    response = chat_engine.chat(question)
    return response.response

def get_memory():
    return memory.get_all()

def get_prompt(question: str) -> str:
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    retrieved_nodes = retriever.retrieve(question)
    context_str = "\n\n".join([node.get_content() for node in retrieved_nodes])

    chat_history = memory.get_all()
    history_str = "\n".join(
        [f"user: {msg.content}" if msg.role == "user" else f"assistant: {msg.content}"
         for msg in chat_history]
    )

    full_prompt = f"""
    {SYSTEM_PROMPT}

    === CONTEXTO ===
    {context_str}

    === HISTÓRICO DA CONVERSA ===
    {history_str}

    === NOVA PERGUNTA ===
    {question}
    """.strip()

    return full_prompt