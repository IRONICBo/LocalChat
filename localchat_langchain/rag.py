import gradio as gr
from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import Ollama
import chromadb
import os

# Initialize ChromaDB with persistence
db_path = "./chroma_db"
db_client = chromadb.PersistentClient(path=db_path)
collection_name = "knowledge_base"
chroma_collection = db_client.get_or_create_collection(collection_name)

# Initialize embedding model and Ollama LLM
embed_args = {
    "model_name": "maidalun1020/bce-embedding-base_v1",
    "max_length": 512,
    "embed_batch_size": 32,
    "device": "cpu",
}
embed_model = HuggingFaceEmbedding(**embed_args)
llm = Ollama(model="qwen2:0.5b")  # Your Ollama LLM instance

# Set up Chroma vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize or load the VectorStoreIndex
if chroma_collection.count() > 0:
    # Load existing index from ChromaDB
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
else:
    # Create a new index
    index = VectorStoreIndex(storage_context=storage_context, embed_model=embed_model)


# Function to upload and index documents
def upload_and_index(file):
    # Read and parse the document
    documents = SimpleDirectoryReader(input_files=[file.name]).load_data()

    # Update the index with new documents
    index.insert_documents(documents)
    return f"Document '{file.name}' successfully uploaded and indexed into the knowledge base!"


# Function to answer a query
def answer_question(query):
    if chroma_collection.count() == 0:
        return "The knowledge base is empty. Please upload a document first!"

    # Set up query engine
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

    # Execute the query
    response = query_engine.query(query)
    return response.response


# Function to fetch documents stored in ChromaDB
def fetch_documents():
    data = chroma_collection.get()
    docs = [meta["text"] for meta in data["metadatas"]]
    return docs


# Gradio UI design
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            "<h1><center>Document Knowledge Base with Chroma and Ollama</center></h1>"
        )

        # Tab for document upload
        with gr.Tab("Upload Documents"):
            file_input = gr.File(
                label="Upload Document", file_types=[".txt", ".md", ".pdf"]
            )
            upload_button = gr.Button("Upload and Index")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            upload_button.click(
                upload_and_index, inputs=[file_input], outputs=[upload_status]
            )

        # Tab for querying knowledge base
        with gr.Tab("Ask a Question"):
            question_input = gr.Textbox(label="Your Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            ask_button = gr.Button("Get Answer")
            ask_button.click(
                answer_question, inputs=[question_input], outputs=[answer_output]
            )

        # Tab to fetch indexed documents
        with gr.Tab("View Indexed Documents"):
            docs_output = gr.Textbox(
                label="Indexed Documents", lines=10, interactive=False
            )
            fetch_button = gr.Button("Fetch Documents")
            fetch_button.click(fetch_documents, outputs=[docs_output])

    demo.launch()


if __name__ == "__main__":
    gradio_ui()
