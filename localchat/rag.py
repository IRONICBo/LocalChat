import gradio as gr
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index import ServiceContext
import pandas as pd
import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma")
collection = client.create_collection("rag_collection")

# Initialize embedding model
embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cpu'}
embed_model = HuggingFaceEmbedding(**embed_args)

# Initialize language model
llm = Ollama(model="qwen2:0.5b")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Function to upload documents and add embeddings to ChromaDB
def upload_and_index(file):
    # Read and process the file
    content = file.read().decode('utf-8')
    chunks = [content[i:i+400] for i in range(0, len(content), 400)]

    # Generate embeddings and add to ChromaDB
    for i, chunk in enumerate(chunks):
        embedding = embed_model.get_text_embedding(chunk)
        collection.upsert(ids=[f"doc_{file.name}_{i}"], embeddings=[embedding], metadatas=[{"text": chunk}])
    return f"Document '{file.name}' uploaded and indexed successfully."

# Function to retrieve relevant chunks and generate answers
def answer_question(query):
    # Generate query embedding
    query_embedding = embed_model.get_text_embedding(query)

    # Retrieve relevant chunks from ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_chunks = [meta["text"] for meta in results["metadatas"][0]]

    # Use LLM to generate an answer based on the retrieved chunks
    context = "\n".join(retrieved_chunks)
    response = llm.complete(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")

    return response

# ChromaDB Management Functions
def fetch_data(page_number, page_size):
    all_data = collection.get()
    ids, embeddings, metadatas = all_data['ids'], all_data['embeddings'], all_data['metadatas']

    data = pd.DataFrame({
        "ID": ids,
        "Metadata": [meta["text"] for meta in metadatas]
    })

    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)

def upsert_data(record_id, embedding, metadata):
    embedding_vector = [float(i) for i in embedding.split(",")]
    collection.upsert(ids=[record_id], embeddings=[embedding_vector], metadatas=[{"text": metadata}])
    return f"Data with ID '{record_id}' upserted successfully."

def delete_data(record_id):
    collection.delete(ids=[record_id])
    return f"Data with ID '{record_id}' deleted successfully."

def search_data(query_embedding, page_number, page_size):
    query_vector = [float(i) for i in query_embedding.split(",")]
    results = collection.query(query_embeddings=[query_vector], n_results=page_size * page_number)

    ids, scores, metadatas = results['ids'][0], results['distances'][0], results['metadatas'][0]
    data = pd.DataFrame({
        "ID": ids,
        "Score": scores,
        "Metadata": [meta["text"] for meta in metadatas]
    })

    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)

# Gradio UI
def chromadb_manager_tab():
    gr.Markdown("## ChromaDB Manager")

    # Upsert Section
    record_id_input = gr.Textbox(label="Record ID")
    embedding_input = gr.Textbox(label="Embedding (comma-separated values)")
    metadata_input = gr.Textbox(label="Metadata")
    upsert_button = gr.Button("Upsert Data")
    upsert_result = gr.Textbox(label="Upsert Result")

    # Deletion Section
    delete_id_input = gr.Textbox(label="Record ID to Delete")
    delete_button = gr.Button("Delete Data")
    delete_result = gr.Textbox(label="Delete Result")

    # Query Section
    page_number_input = gr.Number(label="Page Number", value=1, precision=0)
    page_size_input = gr.Number(label="Page Size", value=10, precision=0)
    query_result_table = gr.DataFrame(headers=["ID", "Metadata"], datatype=["str", "str"], label="Query Results", height=500)
    fetch_data_button = gr.Button("Fetch Data")

    # Search Section
    search_embedding_input = gr.Textbox(label="Search Embedding (comma-separated values)")
    search_page_number_input = gr.Number(label="Search Page Number", value=1, precision=0)
    search_page_size_input = gr.Number(label="Search Page Size", value=10, precision=0)
    search_result_table = gr.DataFrame(headers=["ID", "Score", "Metadata"], datatype=["str", "number", "str"], label="Search Results", height=500)
    search_button = gr.Button("Search Data")

    # Button Click Events
    upsert_button.click(fn=upsert_data, inputs=[record_id_input, embedding_input, metadata_input], outputs=upsert_result)
    delete_button.click(fn=delete_data, inputs=delete_id_input, outputs=delete_result)
    fetch_data_button.click(fn=fetch_data, inputs=[page_number_input, page_size_input], outputs=query_result_table)
    search_button.click(fn=search_data, inputs=[search_embedding_input, search_page_number_input, search_page_size_input], outputs=search_result_table)

if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>RAG and ChromaDB Management System</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="Document Upload & Q&A"):
                gr.Markdown("## Document Upload & Question Answering")
                file_input = gr.File(label="Upload Document", file_types=[".txt", ".md", ".pdf"])
                upload_button = gr.Button("Upload and Index")
                upload_status = gr.Textbox(label="Upload Status")
                question_input = gr.Textbox(label="Enter Your Question")
                answer_output = gr.Textbox(label="Answer", interactive=False)
                ask_button = gr.Button("Get Answer")

                upload_button.click(upload_and_index, inputs=[file_input], outputs=[upload_status])
                ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

            with gr.Tab(label="ChromaDB Manager"):
                chromadb_manager_tab()

    main_block.launch()
