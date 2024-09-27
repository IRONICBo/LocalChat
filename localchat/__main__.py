import os

import gradio as gr
from openai import OpenAI


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history):
    history[-1][1] = ""

    history_openai_format = []
    for human, assistant in history[:-1]:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": history[-1][0]})

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:11434/v1",
    )
    completion = client.chat.completions.create(
        model="qwen:0.5b",
        messages=history_openai_format,
        temperature=0.1,
        stream=True,
    )
    for chunk in completion:
        history[-1][1] += chunk.choices[0].delta.content
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=[],
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

demo.queue()
demo.launch()
