import gradio as gr
from pyzotero import zotero
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import os
import requests
from typing import List

# Initialize Zotero
ZOTERO_USER_ID = "9062826"  # Replace with your Zotero user ID

zot = zotero.Zotero(ZOTERO_USER_ID, "user", local=True)

# Create embeddings and vector store
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)

def fetch_zotero_items(start: int, limit: int) -> List[list]:
    # Fetch items from Zotero API
    items = zot.items(start=start, limit=limit)
    print(f"Fetched {len(items)} items from Zotero. items: {items}")
    formatted_items = []

    for item in items:
        # Extract relevant fields for display in DataFrame
        item_id = item.get("key", "N/A")
        title = item.get("title", "No Title")
        creators = ", ".join([creator.get("lastName", "") for creator in item.get("creators", [])])
        tags = ", ".join(item.get("tags", []))
        date = item.get("date", "No Date")

        formatted_items.append([item_id, title, creators, tags, date])

    return formatted_items

def fetch_items_by_tag(tag: str, limit: int = 100) -> List[list]:
    items = zot.items(tags=tag, limit=limit)
    formatted_items = []

    for item in items:
        # Extract relevant fields for display in DataFrame
        item_id = item.get("key", "N/A")
        title = item.get("title", "No Title")
        creators = ", ".join([creator.get("lastName", "") for creator in item.get("creators", [])])
        tags = ", ".join(item.get("tags", []))
        date = item.get("date", "No Date")

        formatted_items.append([item_id, title, creators, tags, date])

    return formatted_items

# Function to process PDF files and add to vector store
def process_pdf(raw_file_path: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(raw_file_path))
    with open(raw_file_path, "rb") as f:
        content = f.read()

    # Save the uploaded file content to the specified directory
    with open(file_path, "wb") as f:
        f.write(content)

    # Assume we have text extraction from PDF (you need to use PDF text extraction logic here)
    content = "Dummy extracted text from PDF"

    # Split the content into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_text(content)

    # Convert to LangChain Documents with metadata
    langchain_documents = [
        Document(page_content=doc, metadata={"source": os.path.basename(raw_file_path)})
        for doc in documents
    ]

    # Generate UUIDs for the documents
    uuids = [str(uuid4()) for _ in langchain_documents]

    # Add documents to vectorstore
    vectorstore.add_documents(documents=langchain_documents, ids=uuids)
    return "PDF processed and added to vectorstore."

# Gradio UI Function for Zotero Manager Tab
def zotero_manager_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Zotero Library Management")

            # Pagination controls
            start_input = gr.Number(label="Start", value=0, precision=0)
            limit_input = gr.Number(label="Limit", value=10, precision=0)
            fetch_button = gr.Button("Fetch Zotero Items")
            items_output = gr.Dataframe(
                label="Zotero Items",
                headers=["ID", "Title", "Creators", "Tags", "Date"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )

            # Search by tag
            tag_input = gr.Textbox(label="Enter Tag", placeholder="e.g., deep learning")
            tag_search_button = gr.Button("Fetch Items by Tag")
            items_by_tag_output = gr.Dataframe(
                label="Items with Tag",
                headers=["ID", "Title", "Creators", "Tags", "Date"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )

            # File upload and processing
            pdf_input = gr.File(label="Upload a PDF File", file_types=[".pdf"], type="filepath")
            pdf_process_button = gr.Button("Process PDF")
            pdf_process_output = gr.Textbox(label="PDF Processing Status", interactive=False)

        with gr.Column():
            gr.Markdown("## Zotero Vector Database Operations")
            question_input = gr.Textbox(
                label="Enter your question", placeholder="Type your question here...", lines=2
            )
            submit_button = gr.Button("Submit Query")

            answer_output = gr.Dataframe(
                label="Answer",
                headers=["Content", "Metadata"],
                datatype=["str", "str"],
                interactive=False,
            )

    # Handle fetching items from Zotero
    fetch_button.click(fetch_zotero_items, inputs=[start_input, limit_input], outputs=[items_output])

    # Handle fetching items by tag
    tag_search_button.click(fetch_items_by_tag, inputs=[tag_input], outputs=[items_by_tag_output])

    # Handle PDF processing
    pdf_process_button.click(process_pdf, inputs=[pdf_input], outputs=[pdf_process_output])

    # Handle user query and search in vector store
    submit_button.click(
        lambda question: retriever.get_relevant_documents(question),
        inputs=[question_input],
        outputs=[answer_output],
    )


# Main Gradio App
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Zotero Management Interface</center></h1>")
        with gr.Tabs():
            with gr.Tab(label="Zotero Manager"):
                zotero_manager_tab()

    main_block.launch()
