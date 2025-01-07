from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from uuid import uuid4
import os
from langchain_ollama import OllamaEmbeddings

app = FastAPI(
    title="ChromaDB Text Upload and Search API",
    description="""
    This is a text processing and search service based on FastAPI and ChromaDB:
    - Upload text files to ChromaDB using `/upload`.
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

UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

@app.post("/upload", summary="Upload content", description="Update a content and save it to ChromaDB and file system.")
async def upload_content(file: UploadFile):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_text(text)

    langchain_documents = [
        Document(page_content=doc, metadata={"source": file.filename})
        for doc in documents
    ]

    uuids = [str(uuid4()) for _ in langchain_documents]
    vectorstore.add_documents(documents=langchain_documents, ids=uuids)

    return JSONResponse(content={"message": "File processed and added to vectorstore."})


@app.get(
    "/search",
    summary="search similarity",
    description="Search for similar documents in the vectorstore.",
)
async def search(query: str, top_k: int = 2):
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", k=top_k)
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


# @app.on_event("shutdown")
# def shutdown_event():
#     vectorstore.persist()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=True,
    )