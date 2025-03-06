import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import os

from document_manager import get_db, get_document

# File upload directory
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


# Function to process uploaded file and add to vectorstore
def process_file(raw_file_path):
    print(f"Processing file: {raw_file_path}")
    file_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(raw_file_path))
    with open(raw_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Save the uploaded file content to the specified directory
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Read and process the saved file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

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
    return "File processed and added to vectorstore."


# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)


def get_retrieved_documents_with_collection(question, collection_id):
    db = next(get_db())
    current_document = get_document(db, collection_id)
    vectorstore = Chroma(
        collection_name=current_document.name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", k=2)
    results = retriever.get_relevant_documents(question)
    return results


def get_retrieved_documents(question):
    results = retriever.get_relevant_documents(question)
    return results


def answer_question(question):
    try:
        # Perform similarity search
        results = retriever.get_relevant_documents(question)

        # Format results as a list of lists for the Dataframe
        response = [
            [
                doc.page_content,
                str(doc.metadata),
            ]  # Convert each document into a row (list)
            for doc in results
        ]

        # Return the 2D list for Dataframe
        return response if response else [["No relevant documents found.", ""]]
    except Exception as e:
        return [["An error occurred:", str(e)]]


# Local files tab UI function
def retrival_tab():
    gr.Markdown("# LangChain-Powered File Upload & Retrieval Interface")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload a text file", file_types=[".txt"], type="filepath"
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

        with gr.Column():
            # Use a Dataframe for displaying the answer
            answer_output = gr.Dataframe(
                label="Answer",
                headers=["Content", "Metadata"],  # Specify the headers
                datatype=["str", "str"],  # Both columns are strings
                interactive=False,
            )

    process_button.click(process_file, inputs=[file_input], outputs=[process_output])
    submit_button.click(
        answer_question, inputs=[question_input], outputs=[answer_output]
    )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Retrival Management System</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="Retrival"):
                retrival_tab()

    # main_block.queue()
    main_block.launch()
