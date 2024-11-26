import gradio as gr
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import OpenAI
import chromadb
import pandas as pd

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma")
collection_name = "llamaindex_chroma"
chromadb_store = ChromaVectorStore(client=client, collection_name=collection_name)

# Initialize embedding model and LLM
embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cpu'}
embed_model = HuggingFaceEmbedding(**embed_args)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Create a global index
global_index = VectorStoreIndex(vector_store=chromadb_store, service_context=service_context)

# Function to upload and index documents
def upload_and_index(file):
    # Read the document content
    documents = SimpleDirectoryReader(input_files=[file.name]).load_data()

    # Parse the document into smaller chunks (nodes)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
    nodes = node_parser.get_nodes_from_documents(documents)

    # Insert nodes into the ChromaDB index
    global_index.insert_nodes(nodes)
    return f"Document '{file.name}' successfully uploaded and added to the knowledge base!"

# Function to answer questions using the indexed knowledge base
def answer_question(query):
    if global_index is None:
        return "The knowledge base is empty. Please upload a document first!"

    # Use LlamaIndex QueryEngine to retrieve relevant information and generate an answer
    query_engine = global_index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)

    # Return the response
    return response.response

# Function to fetch paginated data from ChromaDB
def fetch_data(page_number, page_size):
    all_data = chromadb_store.client.get(collection_name)
    ids = all_data["ids"]
    embeddings = all_data["embeddings"]
    metadatas = all_data["metadatas"]

    data = pd.DataFrame({
        "ID": ids,
        "Metadata": [meta.get("text", "") for meta in metadatas]
    })

    # Paginate the data
    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)

# Function to insert or update data in ChromaDB
def upsert_data(record_id, embedding, metadata):
    embedding_vector = [float(i) for i in embedding.split(",")]
    chromadb_store.client.upsert(collection_name, ids=[record_id], embeddings=[embedding_vector], metadatas=[{"text": metadata}])
    return f"Data with ID '{record_id}' successfully inserted/updated!"

# Function to delete data from ChromaDB
def delete_data(record_id):
    chromadb_store.client.delete(collection_name, ids=[record_id])
    return f"Data with ID '{record_id}' successfully deleted!"

# Function to search for similar data in ChromaDB using embeddings
def search_data(query_embedding, page_number, page_size):
    query_vector = [float(i) for i in query_embedding.split(",")]
    results = chromadb_store.client.query(collection_name, query_embeddings=[query_vector], n_results=page_size * page_number)

    ids = results["ids"][0]
    scores = results["distances"][0]
    metadatas = results["metadatas"][0]

    data = pd.DataFrame({
        "ID": ids,
        "Score": scores,
        "Metadata": [meta.get("text", "") for meta in metadatas]
    })

    # Paginate the search results
    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)

# Gradio UI design for ChromaDB management
def chromadb_manager_tab():
    gr.Markdown("## ChromaDB Manager")

    # Section for inserting/updating data
    record_id_input = gr.Textbox(label="Record ID")
    embedding_input = gr.Textbox(label="Embedding (comma-separated values)")
    metadata_input = gr.Textbox(label="Metadata")
    upsert_button = gr.Button("Insert/Update Data")
    upsert_result = gr.Textbox(label="Operation Result")

    # Section for deleting data
    delete_id_input = gr.Textbox(label="Record ID (to delete)")
    delete_button = gr.Button("Delete Data")
    delete_result = gr.Textbox(label="Delete Result")

    # Section for paginated data fetching
    page_number_input = gr.Number(label="Page Number", value=1, precision=0)
    page_size_input = gr.Number(label="Page Size", value=10, precision=0)
    query_result_table = gr.DataFrame(headers=["ID", "Metadata"], datatype=["str", "str"], label="Query Results", height=500)
    fetch_data_button = gr.Button("Fetch Data")

    # Section for searching data
    search_embedding_input = gr.Textbox(label="Search Embedding (comma-separated values)")
    search_page_number_input = gr.Number(label="Search Page Number", value=1, precision=0)
    search_page_size_input = gr.Number(label="Search Page Size", value=10, precision=0)
    search_result_table = gr.DataFrame(headers=["ID", "Score", "Metadata"], datatype=["str", "number", "str"], label="Search Results", height=500)
    search_button = gr.Button("Search Data")

    # Connect the buttons to their respective functions
    upsert_button.click(fn=upsert_data, inputs=[record_id_input, embedding_input, metadata_input], outputs=upsert_result)
    delete_button.click(fn=delete_data, inputs=[delete_id_input], outputs=delete_result)
    fetch_data_button.click(fn=fetch_data, inputs=[page_number_input, page_size_input], outputs=query_result_table)
    search_button.click(fn=search_data, inputs=[search_embedding_input, search_page_number_input, search_page_size_input], outputs=search_result_table)

if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LlamaIndex + ChromaDB Management System</center></h1>")

        with gr.Tabs():
            # Tab for document upload and question answering
            with gr.Tab(label="Document Upload & Q&A"):
                gr.Markdown("## Upload Documents and Ask Questions")
                file_input = gr.File(label="Upload Document", file_types=[".txt", ".md", ".pdf"])
                upload_button = gr.Button("Upload and Index")
                upload_status = gr.Textbox(label="Upload Status")
                question_input = gr.Textbox(label="Enter Your Question")
                answer_output = gr.Textbox(label="Answer", interactive=False)
                ask_button = gr.Button("Generate Answer")

                upload_button.click(upload_and_index, inputs=[file_input], outputs=[upload_status])
                ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

            # Tab for managing ChromaDB
            with gr.Tab(label="ChromaDB Manager"):
                chromadb_manager_tab()

    main_block.launch()
