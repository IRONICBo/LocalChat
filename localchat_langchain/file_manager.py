import os
import shutil

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from tqdm import tqdm

from models import get_db, SessionLocal, FileMetadata
from utils.alert import show_info, show_warning
from utils.model_helper import fetch_model_names
from settings import DEFAULT_ROOT_FILE_PATH

# Directory for storing files and vectorstore
os.makedirs(DEFAULT_ROOT_FILE_PATH, exist_ok=True)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)


def process_files(raw_file_paths):
    raw_file_paths = (
        raw_file_paths if isinstance(raw_file_paths, list) else [raw_file_paths]
    )
    for raw_file_path in tqdm(raw_file_paths):
        _process_file(raw_file_path)


# Function to process uploaded file and add to vectorstore
def _process_file(raw_file_path):
    original_file_name = os.path.basename(raw_file_path)
    file_uuid = str(uuid4())
    folder_name = file_uuid[:2]  # First two characters as folder name
    file_name = file_uuid[2:]  # Rest of the UUID as the file name

    # Create directory and save the uploaded file
    file_dir = os.path.join(DEFAULT_ROOT_FILE_PATH, folder_name)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, file_name)

    shutil.copy(raw_file_path, file_path)

    # Read and process the saved file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    file_hash = str(hash(content))
    file_size = len(content)
    ori_type = os.path.splitext(original_file_name)[1]
    if ori_type is None:
        ori_type = "unknown"

    # TODO: current step we do not need to convert txt files.
    new_type = ori_type

    # TODO: Default chunk size and overlap can be adjusted based on your needs
    # Split the content into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_text(content)

    # Convert to LangChain Documents with metadata
    langchain_documents = [
        Document(page_content=doc, metadata={"source": file_name, "uuid": file_uuid})
        for doc in documents
    ]

    # Generate UUIDs for the documents
    uuids = [str(uuid4()) for _ in langchain_documents]

    # Add documents to vectorstore
    vectorstore.add_documents(documents=langchain_documents, ids=uuids)

    submit_file_metadata(
        uuid=file_uuid,
        name=original_file_name,
        file_hash=file_hash,
        location=file_path,
        size=file_size,
        type=new_type,
        ori_type=ori_type,
    )

    return f"File processed and added to vectorstore. UUID: {file_uuid}"


# Function to get retrieved documents
def get_retrieved_documents(question):
    results = retriever.get_relevant_documents(question)
    return results


def get_all_file_metadatas(db):
    return db.query(FileMetadata).all()


def submit_file_metadata(
    uuid,
    name,
    file_hash,
    location,
    size,
    type,
    ori_type,
):
    try:
        db = next(get_db())
        db_filemetadata = FileMetadata(
            uuid=uuid,
            name=name,
            file_hash=file_hash,
            location=location,
            size=size,
            type=type,
            ori_type=ori_type,
        )
        db.add(db_filemetadata)
        db.commit()
        db.refresh(db_filemetadata)
        show_info(f"File {name} metadata with UUID {uuid} added successfully.")
    except Exception as e:
        show_warning(f"Error adding file metadata: {str(e)}")

    return db_filemetadata


def fetch_file_metadata_list():
    db = next(get_db())
    file_metadatas = get_all_file_metadatas(db)
    file_metadata_list = [
        [
            file_metadata.id,
            file_metadata.uuid,
            file_metadata.name,
            file_metadata.file_hash,
            file_metadata.location,
            file_metadata.size,
            file_metadata.type,
            file_metadata.ori_type,
            file_metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            file_metadata.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for file_metadata in file_metadatas
    ]

    if not file_metadata_list:
        show_warning("No file metadata available.")

    return file_metadata_list


# File upload and document retrieval UI function
def file_manager_tab():
    gr.Markdown("## Manage File")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload text files",
                file_types=[".txt"],
                type="filepath",
                file_count="multiple",
            )
            process_button = gr.Button("Process File")
            process_output = gr.Textbox(
                label="File Processing Status", interactive=False
            )

            question_input = gr.Textbox(
                label="Enter your question",
                placeholder="Type your question here...",
                lines=2,
            )
            submit_button = gr.Button("Submit")

        with gr.Column(scale=3):
            # Use a Dataframe for displaying the answer
            file_metadata_list = gr.Dataframe(
                label="File Metadata",
                headers=[
                    "ID",
                    "UUID",
                    "Name",
                    "Hash",
                    "Location",
                    "Size",
                    "Type",
                    "Origional Type",
                    "Create At",
                    "Updated At",
                ],  # Specify the headers
                interactive=False,
            )

    process_button.click(process_files, inputs=[file_input], outputs=[process_output])
    submit_button.click(
        fetch_file_metadata_list, inputs=[], outputs=[file_metadata_list]
    )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChat File Manager</center></h1>")
        file_manager_tab()

    main_block.queue()
    main_block.launch()
