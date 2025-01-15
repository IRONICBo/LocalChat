from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from uuid import uuid4
from trafilatura import extract
import os
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

from models import HtmlMetadata, SessionLocal

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


class UploadData(BaseModel):
    content: str
    url: str


@app.post(
    "/upload",
    summary="Upload json",
    description="Upload a JSON file containing content, save it, and add it to the vectorstore.",
)
async def upload_json(data: UploadData):
    print(data.content)
    if not data.content:
        raise HTTPException(
            status_code=400, detail="Content field is required in the JSON data."
        )

    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}.txt")

    # with open(file_path, "w", encoding="utf-8") as f:
    #     f.write(data.content)

    markdown_content = extract(
        data.content
    )
    print(111, data.content)
    print(222, markdown_content)

    if markdown_content is None:
        raise HTTPException(
            status_code=400, detail="Failed to extract content from the HTML."
        )

    # 5. Save the converted Markdown content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_text(markdown_content)

    langchain_documents = [
        Document(page_content=doc, metadata={"source": f"{file_id}.txt"})
        for doc in documents
    ]

    uuids = [str(uuid4()) for _ in langchain_documents]
    vectorstore.add_documents(documents=langchain_documents, ids=uuids)

    db = SessionLocal()
    try:
        metadata = HtmlMetadata(url=data.url, filepath=file_path)
        db.add(metadata)
        db.commit()
        db.refresh(metadata)

        return JSONResponse(
            content={
                "message": "HTML file downloaded, converted to Markdown, and metadata saved.",
                "file_id": file_id,
                "metadata": {
                    "url": data.url,
                    "filepath": file_path,
                    "id": metadata.id,
                    "created_at": metadata.created_at,
                },
            }
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    finally:
        db.close()


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
        port=8080,
        # reload=True,
    )
