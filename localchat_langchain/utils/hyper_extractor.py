from typing import Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from openai import BaseModel

import logger


class TextChunk(BaseModel):
    content: str
    metadata: Optional[Dict] = None


def extract_docx_with_headings(file_path, chunk_size=1000, chunk_overlap=0):
    try:
        doc = Document(file_path)

        headings = []
        current_content = []
        chunks = []

        # Split current data
        for paragraph in doc.paragraphs:
            if paragraph.style.name.startswith("Heading"):
                if current_content:
                    chunk_text = "\n".join(current_content)
                    chunks.append(
                        {
                            "content": chunk_text,
                            "metadata": {
                                "headings": (
                                    " > ".join(headings) if headings else "No Heading"
                                )
                            },
                        }
                    )
                    current_content = []

                heading_level = int(paragraph.style.name.split()[-1])
                headings = headings[: heading_level - 1] + [paragraph.text]
            else:
                if paragraph.text.strip():
                    current_content.append(paragraph.text)

        if current_content:
            chunk_text = "\n".join(current_content)
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "headings": " > ".join(headings) if headings else "No Heading"
                    },
                }
            )

        # Split longer chunks
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for chunk in chunks:
            if len(chunk["content"]) > chunk_size:
                sub_chunks = text_splitter.split_text(chunk["content"])
                for sub_chunk in sub_chunks:
                    final_chunks.append(
                        TextChunk(
                            content=sub_chunk,
                            metadata={"headings": chunk["metadata"]["headings"]},
                        )
                    )
            else:
                final_chunks.append(
                    TextChunk(
                        content=chunk["content"],
                        metadata={"headings": chunk["metadata"]["headings"]},
                    )
                )
    except Exception as e:
        logger.error(f"Error processing Word document {file_path}: {e}")
        raise
