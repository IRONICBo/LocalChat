import gradio as gr


def show_warning(message):
    return gr.Warning(message)


def show_info(message):
    return gr.Info(message)
