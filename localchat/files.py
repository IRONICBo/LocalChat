import os
import shutil
import chromadb
from llama_index import Document, SimpleDirectoryReader
import pandas as pd
import gradio as gr
from llama_index.node_parser import SentenceSplitter
from llama_index.ingestion.pipeline import IngestionPipeline
from llama_index.embeddings import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Define file upload directory
upload_path = "uploads"
ollama_embedding = OllamaEmbedding(model_name="nomic-embed-text:latest")

db_path = "./chroma_db"
db_client = chromadb.PersistentClient(path=db_path)
collection_name = "default"
chroma_collection = db_client.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Ensure the directory exists
if not os.path.exists(upload_path):
    os.makedirs(upload_path)


# Fetch paginated file list with sorting
def fetch_file_list(
    page_number, page_size, sort_by="name", reverse=False, search_query=""
):
    files = os.listdir(upload_path)
    files_info = [
        {
            "Name": f,
            "Size (bytes)": os.path.getsize(os.path.join(upload_path, f)),
            "Modified At": os.path.getmtime(os.path.join(upload_path, f)),
        }
        for f in files
        if search_query.lower() in f.lower()
    ]

    # Sort files
    if sort_by == "name":
        files_info.sort(key=lambda x: x["Name"], reverse=reverse)
    elif sort_by == "size":
        files_info.sort(key=lambda x: x["Size (bytes)"], reverse=reverse)
    elif sort_by == "modified":
        files_info.sort(key=lambda x: x["Modified At"], reverse=reverse)

    # Convert to DataFrame and apply pagination
    df = pd.DataFrame(files_info)
    start = (page_number - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].reset_index(drop=True)

# Get documents from file
def get_documents_from_file(file) -> Document:
    with open(file, "r") as f:
        text = f.read()
    return Document(text=text,metadata={"file": file})


# Upload file function
def upload_file(raw_file_path):
    print(f"Uploading file: {raw_file_path}")
    file_name = os.path.basename(raw_file_path)
    file_path = os.path.join(upload_path, file_name)
    # copy file to upload directory
    shutil.copyfile(raw_file_path, file_path)

    # Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            ollama_embedding,
        ],
        vector_store=vector_store,
    )

    documents = SimpleDirectoryReader("/Users/asklv/Projects/AO.space/LocalLLM/LocalChat/localchat/uploads/aaa").load_data()

    pipeline.run(
        documents=documents,
        show_progress=True,
        num_workers=4,
    )

    return f"File '{file_name}' uploaded successfully!"


# Delete file function
def delete_file(file_name):
    file_path = os.path.join(upload_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return f"File '{file_name}' deleted successfully!"
    else:
        return "File not found!"


# Download file function
def download_file(file_name):
    file_path = os.path.join(upload_path, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        return "File not found!"


# Local files tab UI function
def files_tab():
    gr.Markdown("## File Management")

    # Input elements for pagination, sorting, and search
    with gr.Row():
        page_number_input = gr.Number(label="Page Number", value=1, precision=0)
        page_size_input = gr.Number(label="Page Size", value=10, precision=0)
        sort_by_input = gr.Dropdown(
            ["name", "size", "modified"], label="Sort By", value="name"
        )
        reverse_sort_input = gr.Checkbox(label="Sort Descending")
        search_query_input = gr.Textbox(label="Search File Name")

    # Display file list as a DataFrame
    file_list_table = gr.DataFrame(
        headers=["Name", "Size (bytes)", "Modified At"],
        datatype=["str", "number", "str"],
        label="Files",
        height=500,
    )

    # Fetch button
    fetch_files_button = gr.Button("Fetch File List")
    fetch_files_button.click(
        fn=fetch_file_list,
        inputs=[
            page_number_input,
            page_size_input,
            sort_by_input,
            reverse_sort_input,
            search_query_input,
        ],
        outputs=file_list_table,
    )

    # Upload elements
    with gr.Row():
        file_input = gr.File(label="Upload File")
        upload_button = gr.Button("Upload")
        upload_result = gr.Textbox(label="Upload Result")
        upload_button.click(fn=upload_file, inputs=file_input, outputs=upload_result)

    # Delete elements
    with gr.Row():
        delete_file_name_input = gr.Textbox(label="File Name to Delete")
        delete_button = gr.Button("Delete")
        delete_result = gr.Textbox(label="Delete Result")
        delete_button.click(
            fn=delete_file, inputs=delete_file_name_input, outputs=delete_result
        )

    # Download elements
    with gr.Row():
        download_file_name_input = gr.Textbox(label="File Name to Download")
        download_button = gr.Button("Download")
        download_output = gr.File(label="Download File")
        download_button.click(
            fn=download_file, inputs=download_file_name_input, outputs=download_output
        )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Files Management System</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="Files"):
                files_tab()

    # main_block.queue()
    main_block.launch()
