from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Caminho para o PDF
PDF_PATH = Path("backend/data/docs/serenatto_cafes_especiais.pdf")
VECTORSTORE_PATH = "vector_store/index"

# 1. Carregar PDF com o loader do LangChain
loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

# 2. Dividir o conteúdo em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Criar modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Criar e salvar o FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTORSTORE_PATH)

print(f"✅ Vetor store salvo com sucesso em {VECTORSTORE_PATH}")
