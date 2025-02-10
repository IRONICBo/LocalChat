from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from langchain_community.document_loaders import Docx2txtLoader

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import os

import uvicorn

# File upload directory
UPLOAD_DIRECTORY = "./uploaded_files_tmp"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


app = FastAPI(
    title="Keywords extract for search API",
    description="""
    This is a text processing and search service based on FastAPI and ChromaDB:
    - Search for relevant content using `/search`.
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_folder(files):
    for file in files:
        file_path = file
        if file.endswith(".docx"):  # Only process .txt files
            print(f"Processing file: {file_path}")

            loader = Docx2txtLoader(file_path=file_path)
            data = loader.load()
            content = data[0].page_content
            print(f"Content of file: {content}")

            # Get core keywords
            tr4w = TextRank4Keyword()
            tr4w.analyze(content, lower=True, window=2)
            keywords = []
            for item in tr4w.get_keywords(100, word_min_len=2):
                keywords.append(item.word)
            print(f"Keywords of file: {keywords}")

            # Convert to LangChain Documents with metadata
            langchain_documents = [
                Document(page_content=doc, metadata={"source": file_path})
                for doc in keywords
            ]

            # Generate UUIDs for the documents
            uuids = [str(uuid4()) for _ in langchain_documents]

            # Add documents to vectorstore
            vectorstore.add_documents(documents=langchain_documents, ids=uuids)

    return f"Folder processed and files added to vectorstore, {len(files)} files processed."


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
    persist_directory="./chroma_db_tmp",
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@app.get(
    "/search",
    summary="search similarity",
    description="Search for similar documents in the vectorstore.",
)
async def search(query: str, top_k: int = 2):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        results = retriever.get_relevant_documents(query)
        response = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
        return JSONResponse(content={"results": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")


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
                label="Upload a text file",
                file_types=[".txt", ".docx"],
                type="filepath",
                file_count="directory",
                interactive=True,
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

    # process_button.click(process_file, inputs=[file_input], outputs=[process_output])
    process_button.click(process_folder, inputs=[file_input], outputs=[process_output])
    submit_button.click(
        answer_question, inputs=[question_input], outputs=[answer_output]
    )


# Main Gradio app
if __name__ == "__main__":
    import threading

    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Retrival Management System</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="Retrival"):
                retrival_tab()

    # main_block.queue()
    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8082)

    threading.Thread(target=run_uvicorn).start()

    main_block.launch()
