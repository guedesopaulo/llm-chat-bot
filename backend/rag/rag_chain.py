from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# LLM via Ollama
llm = Ollama(model="llama3")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Carrega o vector store FAISS local
vectorstore = FAISS.load_local("vector_store/index", embeddings, allow_dangerous_deserialization=True)

# Prompt customizado
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é um assistente de cafés especiais. Responda com base no contexto.\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta: {question}\n\n"
        "Resposta:"
    )
)

# Cadeia de recuperação e geração
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Função principal de resposta
def answer_question(question: str) -> str:
    return qa_chain.run(question)
