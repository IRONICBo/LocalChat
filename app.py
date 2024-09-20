import gradio as gr
import requests
import json
import time
import psutil

# 调用本地API
def call_local_api(prompt, model):
    url = "http://localhost:11434/v1/completions"  # 本地API端点
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,  # 根据选择的模型进行调用
        "prompt": prompt
    }

    start_time = time.time()  # 开始计时
    response = requests.post(url, headers=headers, data=json.dumps(data))
    end_time = time.time()  # 结束计时
    response_time = end_time - start_time  # 响应时间

    if response.status_code == 200:
        result = response.json()

        # 检查返回的数据结构，确保解析正确
        completion_text = result['choices'][0]['text'] if 'choices' in result else "No text found"
        prompt_tokens = result['usage'].get('prompt_tokens', 0)
        completion_tokens = result['usage'].get('completion_tokens', 0)
        total_tokens = result['usage'].get('total_tokens', 0)

        return completion_text, prompt_tokens, completion_tokens, total_tokens, response_time
    else:
        return f"Error: {response.status_code}, {response.text}", 0, 0, 0, 0

# 获取当前进程的内存占用
def get_memory_usage():
    process = psutil.Process()  # 获取当前进程
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # 将内存占用量转换为MB

# ChatGPT模型功能
def chatgpt_clone(input, history, model):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)

    # 调用API，获取结果和tokens数据
    output, prompt_tokens, completion_tokens, total_tokens, response_time = call_local_api(inp, model)

    # 获取当前内存占用
    memory_usage = get_memory_usage()

    # 更新聊天记录
    history.append((input, output))

    # 生成Markdown表格
    table_md = f"""
    | prompt_tokens | completion_tokens | total_tokens | response_time (s) | memory_usage (MB) |
    |---------------|-------------------|--------------|-------------------|------------------|
    | {prompt_tokens}        | {completion_tokens}            | {total_tokens}       | {round(response_time, 3)}             | {round(memory_usage, 2)} MB        |
    """

    return history, history, table_md

# Gradio界面设计
block = gr.Blocks()

with block:
    gr.Markdown("<h1><center>Build Your Own Chatbot with Local LLM Model</center></h1>")

    with gr.Row():
        # 聊天框
        chatbot = gr.Chatbot(label="Chatbot")
        # 表格，使用Markdown显示
        table = gr.Markdown()

    # 输入框，提示用户输入内容
    message = gr.Textbox(placeholder="Ask anything to the AI assistant...", label="Your Prompt")

    # 模型选择框
    model_choice = gr.Dropdown(choices=["qwen:0.5b", "qwen:1.8b", "qwen:4b"], value="qwen:0.5b", label="Choose Model")

    # 状态保存聊天记录
    state = gr.State()

    # 发送按钮
    submit = gr.Button("SEND")

    # 点击按钮后调用 chatgpt_clone 函数
    submit.click(chatgpt_clone, inputs=[message, state, model_choice], outputs=[chatbot, state, table])

block.launch(debug=True)
