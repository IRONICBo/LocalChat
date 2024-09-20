# LocalChat

一个用于与本地文档互动的基于本地LLM驱动的聊天应用。

[EN](README.md) | [中文](README_ZH.md)

## 先决条件

- Python 3.8 或更高版本
- Docker Desktop 4.9.1 (81317) 版本：20.10.16
- Go 1.22.5
- Conda 22.11.1

## 使用 Docker 运行

### 启动后端

```bash
docker run -d -v YOUR_LOCAL_OLLAMA_PATH:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 启动前端

```bash
pip install -r requirements.txt
python app.py
```

## 运行本地 LLM

### 启动后端

如果你有预训练模型文件，可以从 `https://ollama.com/library/qwen` 下载此文件，或者通过 ollama 命令下载。

对于 Mac:

从 `https://github.com/ollama/ollama` 克隆 ollama 并运行以下命令：

> 确保你的本地机器上有以下工具：
> cmake 版本 3.24 或更高
> go 版本 1.22 或更高
> gcc 版本 11.4.0 或更高

```bash
# 编译时
export CGO_CFLAGS="-g"
# 运行时
export OLLAMA_DEBUG=1
go generate ./...
go build .

./ollama serve
./ollama run qwen:0.5b # 内存使用: 827.8MB
# 可选
# ./ollama run qwen:1.8b # 内存使用: 1.56GB
```

> 参考: https://github.com/ollama/ollama/blob/main/docs/development.md

### 启动前端

为 gradio 准备前端环境：

```bash
pip install -r requirements.txt
python app.py
```