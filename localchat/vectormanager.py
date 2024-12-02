import gradio as gr
import chromadb
import pandas as pd
from models import SessionLocal, DocumentLibrary

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma")

# Helper function to create a new document library in SQLite
def create_document_library(name):
    db = SessionLocal()
    try:
        new_library = DocumentLibrary(name=name)
        db.add(new_library)
        db.commit()
        return f"Library '{name}' created successfully."
    except Exception as e:
        db.rollback()
        return f"Error: {str(e)}"
    finally:
        db.close()

# Helper function to fetch all document libraries
def fetch_document_libraries():
    db = SessionLocal()
    try:
        libraries = db.query(DocumentLibrary).all()
        data = [{"ID": lib.id, "Name": lib.name, "Created At": lib.created_at} for lib in libraries]
        return pd.DataFrame(data)
    finally:
        db.close()

# Helper function to switch ChromaDB collection
def switch_collection(library_name):
    try:
        global collection
        collection = client.get_or_create_collection(library_name)
        return f"Switched to collection '{library_name}'."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
def document_library_manager_tab():
    gr.Markdown("## Document Library Manager")

    # Create Library Section
    library_name_input = gr.Textbox(label="Library Name")
    create_library_button = gr.Button("Create Library")
    create_library_result = gr.Textbox(label="Create Library Result")

    # View Libraries Section
    view_libraries_button = gr.Button("View Libraries")
    libraries_table = gr.DataFrame(headers=["ID", "Name", "Created At"], label="Available Libraries", height=300)

    # Switch Collection Section (Dropdown for library selection)
    library_options_dropdown = gr.Dropdown(label="Select Library to Switch", choices=["test"], multiselect=False)
    switch_library_button = gr.Button("Switch Library")
    switch_library_result_popup = gr.Textbox(label="Switch Library Result (Popup Message)", visible=False)

    # Button Click Events
    def update_library_dropdown():
        """Fetch library names for the dropdown."""
        libraries = fetch_document_libraries()
        # return libraries["Name"].tolist()
        return gr.Dropdown.input(choices=libraries["Name"].tolist())

    def show_popup_message(message):
        """Show a message dynamically."""
        return gr.update(value=message, visible=True)

    # Add change event
    library_options_dropdown.change(fn=update_library_dropdown, inputs=view_libraries_button, outputs=library_options_dropdown)

    # Update dropdown options
    view_libraries_button.click(fn=fetch_document_libraries, outputs=libraries_table)
    view_libraries_button.click(fn=update_library_dropdown, outputs=library_options_dropdown)

    # Handle library switching
    switch_library_button.click(fn=switch_collection, inputs=library_options_dropdown, outputs=switch_library_result_popup)
    switch_library_button.click(fn=show_popup_message, inputs=switch_library_result_popup, outputs=switch_library_result_popup)

    # Create library logic
    create_library_button.click(fn=create_document_library, inputs=library_name_input, outputs=create_library_result)
    create_library_button.click(fn=show_popup_message, inputs=create_library_result, outputs=create_library_result)

# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>ChromaDB and Document Library Management</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="Document Library Manager"):
                document_library_manager_tab()

    main_block.queue()
    main_block.launch()