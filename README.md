# LocalChat
A personal LLM-driven chat application for engaging with local documents.

[EN](README.md) | [中文](README_ZH.md)

## Pre-requisites

- Python 3.8 or higher
- Docker Desktop 4.9.1 (81317) Version: 20.10.16
- Go 1.22.5
- Conda 22.11.1

## Run with docker

### start backend

```bash
docker run -d -v YOUR_LOCAL_OLLAMA_PATH:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### start frontend

```bash
pip install -r requirements.txt
python app.py
```

## Run local llm

### start backend

If you do have pretrained model files, you can download this file from `https://ollama.com/library/qwen` or donwload by ollama command.


For Mac:

Clone ollama from `https://github.com/ollama/ollama` and run the following command:

> Make sure you have these tools in your local machine:
> cmake version 3.24 or higher
> go version 1.22 or higher
> gcc version 11.4.0 or higher

```bash
# At build time
export CGO_CFLAGS="-g"
# At runtime
export OLLAMA_DEBUG=1
go generate ./...
go build .

./ollama serve
./ollama run qwen:0.5b # mem usage: 827.8MB
# optional
# ./ollama run qwen:1.8b # mem usage: 1.56GB
```

> Ref: https://github.com/ollama/ollama/blob/main/docs/development.md

### start frontend

Prepare the frontend environment for gradio:

```bash
pip install -r requirements.txt
python app.py
```

## Thanks

- [trafilatura](https://github.com/adbar/trafilatura): Discover and Extract Text Data on the Web
- [langchain](https://github.com/langchain-ai/langchain): LangChain makes it easy to build applications using large language models and other sources of data.
- [ollama](https://github.com/jmorganca/ollama): Ollama is a fast, reliable, and open-source alternative to OpenAI's ChatGPT API.
- [gradio](https://github.com/gradio-app/gradio): Gradio is an open-source framework for building machine learning and data science apps.
- [qwen](https://github.com/QwenLM/Qwen-Chat): Qwen-Chat is a Chinese language model developed by Qwen Labs.
- [simple-one-api](https://github.com/fruitbars/simple-one-api): Various large models accessible through a standardized OpenAI API format, ready to use out of the box