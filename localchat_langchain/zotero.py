import gradio as gr
from urllib.parse import unquote
from pyzotero import zotero
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import os
from typing import List

from langchain.document_loaders import PyPDFLoader

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


def get_local_file_path(item):
    if "links" in item and "enclosure" in item["links"]:
        enclosure_link = item["links"]["enclosure"].get("href", "No Path")
        if enclosure_link and enclosure_link.startswith("file://"):
            raw_path = enclosure_link.replace("file://", "")
            # Unquote the path to handle special characters
            return unquote(raw_path)
    return None


def format_zotero_items(items):
    """
    Format Zotero items for display in a Dataframe-compatible way.
    """
    formatted_items = []
    for item in items:
        # Extract metadata for display
        item_id = item.get("key", "N/A")
        title = item["data"].get("title", "No Title")
        creators = ", ".join([
            f'{creator.get("firstName", "")} {creator.get("lastName", "")}'.strip()
            for creator in item["data"].get("creators", [])
        ])
        tags = ", ".join([tag.get("tag", "") for tag in item["data"].get("tags", [])])
        date = item["data"].get("date", "No Date")
        url = item["data"].get("url", "No URL")
        local_path = get_local_file_path(item)
        # Append as a list (not a dictionary)
        formatted_items.append([item_id, title, creators, tags, date, url, local_path])
    return formatted_items


def fetch_zotero_items(start: int, limit: int) -> List[list]:
    """
    Fetch items from Zotero API and format them for display in Dataframe.
    """
    items = zot.items(start=start, limit=limit)
    return format_zotero_items(items)


def process_files(items):
    """
    Process valid file paths extracted from items_output.
    :param items: Dataframe content (list of rows).
    :return: Processing status message.
    """
    print(f"Processing {len(items)} items")

    valid_paths = []
    for i, row in enumerate(items['Path']):
        print(f"Row {i}: {row}")
        path = row

        # Check if the path exists
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Path does not exist: {path}")

    print(f"Found {len(valid_paths)} valid files for processing.")

    if len(valid_paths) == 0:
        return "No valid files found for processing."

    # Process each valid file
    processed_count = 0
    for path in valid_paths:
        file_name = path
        print(f"Processing file: {file_name}")

        # Simulated PDF processing logic
        loader = PyPDFLoader(file_path=file_name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        langchain_documents = loader.load_and_split(text_splitter=text_splitter)

        uuids = [str(uuid4()) for _ in langchain_documents]
        vectorstore.add_documents(documents=langchain_documents, ids=uuids)

        processed_count += 1

    return f"Processed {processed_count} valid files out of {len(valid_paths)} total paths."


def zotero_manager_tab():
    """
    Create Gradio UI for Zotero management.
    """
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Zotero Library Management")

            # Pagination controls
            start_input = gr.Number(label="Start", value=0, precision=0)
            limit_input = gr.Number(label="Limit", value=10, precision=0)
            fetch_button = gr.Button("Fetch Zotero Items")
            items_output = gr.Dataframe(
                label="Zotero Items",
                headers=["ID", "Title", "Creators", "Tags", "Date", "URL", "Path"],  # Column headers
                datatype=["str", "str", "str", "str", "str", "str", "str"],         # Data types
                interactive=False,
            )

            # Process files
            process_button = gr.Button("Process Selected Files")
            process_output = gr.Textbox(label="Processing Status", interactive=False)

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

    # Fetch Zotero items
    fetch_button.click(fetch_zotero_items, inputs=[start_input, limit_input], outputs=[items_output])

    # Process selected files
    process_button.click(
        lambda items: process_files(items),  # Ensure items are valid
        inputs=[items_output],
        outputs=[process_output],
    )

    # Submit question to vectorstore
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
