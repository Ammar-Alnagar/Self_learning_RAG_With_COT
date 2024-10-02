from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma





def index_conversations():
    # Load documents from a PDF file
    loader = DirectoryLoader("./Data", glob="**/*.pdf")
    print("Loading PDF documents...")
    documents = loader.load()
    print(f"{len(documents)} documents loaded.")

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    # Create Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=300,
        add_start_index=True,
    )

    # Split documents into chunks
    print("Splitting documents into chunks...")
    texts = text_splitter.split_documents(documents)

    # Create and persist the vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./db-mawared"
    )

    print("Vectorstore created and persisted.")

if __name__ == "__main__":
    index_conversations()
