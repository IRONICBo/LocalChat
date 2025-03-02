import os
import shutil

import chromadb
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from tqdm import tqdm

from document_manager import get_document
from model_manager import DEFAULT_PAGE_NUM, DEFAULT_PAGE_SIZE
from utils.alert import show_warning
from models import DocumentLibrary, FileMetadata, SessionLocal, get_db


# Helper function to fetch all document libraries
def fetch_document_pairs():
    db = next(get_db())
    try:
        libraries = db.query(DocumentLibrary).all()
        data = [(lib.name, lib.id) for lib in libraries]
        return data
    finally:
        db.close()

def refresh_file_list(document_id=0):
    db = next(get_db())
    try:
        file_metadatas = db.query(FileMetadata).filter(FileMetadata.document_id == document_id).all()
        data = [
            (file_metadata.name, file_metadata.uuid) for file_metadata in file_metadatas
        ]
        return data
    except Exception as e:
        show_warning(f"Error fetching document libraries: {e}")
    finally:
        db.close()

def fetch_file_chunk_list(page_number_input, page_size_input, document_id, uuid):
    """
    collection_name is the document name in chroma db.
    source is the generated filename for specific file.
    """
    db = next(get_db())
    current_document = get_document(db, document_id)

    chromadb_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chromadb_client.get_or_create_collection(current_document.name)
    chunks = chroma_collection.get(
        include=["documents"],
        where={'uuid': uuid}
    )
    print(chunks["documents"])
    return [chunks["documents"]]

def update_model_dropdown(document_id=0):
    """Update Dropdown with file names."""
    file_list = refresh_file_list(document_id)
    print(file_list)
    return gr.update(choices=file_list, value=None)


# Chunk upload and document retrieval UI function
def chunk_manager_tab():
    gr.Markdown("## Manage File")

    with gr.Row():
        with gr.Column(scale=1):
            document_pairs = fetch_document_pairs()
            print(document_pairs)
            knowledge_base_choice = gr.Dropdown(
                choices=document_pairs,
                # Tips: default value is a tuple (default, 1)
                label="Choose Knowledge Base",
            )
            file_choice = gr.Dropdown(
                choices=["Please Refresh..."],
                # Tips: default value is a tuple (default, 1)
                label="Choose File",
            )
            process_button = gr.Button("Refresh File List")
            process_button.click(update_model_dropdown, inputs=[knowledge_base_choice], outputs=file_choice)

            page_number_input = gr.Number(
                label="Page Number", value=DEFAULT_PAGE_NUM, precision=0
            )
            page_size_input = gr.Slider(
                label="Page Size",
                value=DEFAULT_PAGE_SIZE,
                minimum=1,
                maximum=10,
                step=1,
            )
            fetch_files_metadata_button = gr.Button("Refresh Chunk Metadata")

        with gr.Column(scale=3):
            chunk_metadata_list = gr.Dataframe(
                label="Chunk Metadata",
                headers=[
                    "Chunk",
                ],  # Specify the headers
                interactive=False,
            )

    fetch_files_metadata_button.click(
        fn=fetch_file_chunk_list,
        inputs=[page_number_input, page_size_input, knowledge_base_choice, file_choice],
        outputs=chunk_metadata_list,
    )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChat Chunk Manager</center></h1>")
        chunk_manager_tab()

    main_block.queue()
    main_block.launch()
