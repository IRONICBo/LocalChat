import gradio as gr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from utils.alert import show_info, show_warning
from models import SessionLocal, DocumentLibrary


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_document(db, name, description):
    db_document = DocumentLibrary(name=name, description=description)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document(db, document_id):
    return db.query(DocumentLibrary).filter(DocumentLibrary.id == document_id).first()

def get_all_documents(db):
    return db.query(DocumentLibrary).all()


def submit_document(name, description):
    db = next(get_db())
    document = create_document(db, name, description)
    show_info(f"Document '{document.name}' added successfully.")


def load_documents():
    db = next(get_db())
    documents = get_all_documents(db)
    document_list = [
        f"ID: {doc.id} | Name: {doc.name} | Description: {doc.description}"
        for doc in documents
    ]
    # return "\n".join(document_list) if document_list else "No documents available."
    document_list = [
        [
            doc.id,
            doc.name,
            doc.description,
            doc.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for doc in documents
    ]

    if not document_list:
        show_warning("No documents available.")

    return document_list


def delete_document(db, document_id):
    document = (
        db.query(DocumentLibrary).filter(DocumentLibrary.id == document_id).first()
    )
    if document:
        db.delete(document)
        db.commit()
        return f"Document '{document.name}' deleted successfully."
    else:
        return "Document not found."


def remove_document(document_id):
    db = next(get_db())
    result = delete_document(db, document_id)
    return result


# Gradio Interface
def document_manager_tab():
    with gr.Row():
        # Left side - Document submission form
        with gr.Column(scale=1):
            gr.Markdown("## Knowledge Operation")

            # Document submission form
            name_input = gr.Textbox(label="Knowledge Name")
            description_input = gr.Textbox(label="Document Description")
            submit_button = gr.Button("Add Knowledge")
            submit_button.click(submit_document, inputs=[name_input, description_input])

            gr.Markdown("---")

            # Add delete buttons for each document item in the list
            document_id_input = gr.Number(label="Knowledge ID to Delete", precision=0)
            delete_button = gr.Button("Delete Knowledge")
            delete_button.click(remove_document, inputs=document_id_input)

        # Right side - Document list display with delete support
        with gr.Column(scale=2):
            document_list = load_documents()
            document_list_output = gr.Dataframe(
                headers=["ID", "Name", "Description", "CreateTime"],
                value=document_list,
            )
            load_button = gr.Button("Refresh Documents")
            load_button.click(load_documents, outputs=document_list_output)


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChat Document Manager</center></h1>")
        document_manager_tab()

    main_block.queue()
    main_block.launch()
