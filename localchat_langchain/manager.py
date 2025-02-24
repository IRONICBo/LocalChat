import gradio as gr

from document_manager import document_manager_tab
from model_manager import model_manager_tab


def manager_tab():
    with gr.Tab(label="Model Manager"):
        model_manager_tab()

    with gr.Tab(label="Knowledge Manager"):
        document_manager_tab()


if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChat Manager</center></h1>")
        manager_tab()

    main_block.queue()
    main_block.launch()
