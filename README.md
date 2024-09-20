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

throughput: 1.7mb/s
mem usage: 880.8MB
docker gpu mac支持
是否docker是否可以用加速
固定版本，同步的数据信息

> Ref: https://github.com/ollama/ollama/blob/main/docs/development.md

### start frontend

Prepare the frontend environment for gradio:

```bash
pip install -r requirements.txt
python app.py
```