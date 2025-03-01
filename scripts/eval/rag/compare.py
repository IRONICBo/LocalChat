from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8787/v1",
)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)

def raw_chat(prompt, model="qwen2:0.5b"):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:11434/v1",
    )
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return completion.choices[0].message.content

def rag_chat(prompt, model="qwen2:0.5b"):
    def _get_retrieved_documents(question):
        results = retriever.get_relevant_documents(question)
        return results
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:11434/v1",
    )
    messages = [{"role": "user", "content": prompt}]
    knowledge_base = _get_retrieved_documents(prompt)
    kb_data = "Current data is, you need to refer to the following data: "
    for doc in knowledge_base:
        kb_data += doc.page_content + "\n"
    messages.append({"role": "user", "content": kb_data})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return completion.choices[0].message.content

def compare(func1, func2, a):
    b = func1(a)
    c = func2(a)

    messages = [
        {
            "role": "user",
            "content": f"Subject: 回答质量对比\nQuestion: 比较以下两个输出哪个更好？\nInput: {a}\nOutput 1: {b}\nOutput 2: {c}\nAnswer:"
        }
    ]
    print(f"Current comparasion: {messages}")

    completion = client.chat.completions.create(
        model="spark-lite",
        messages=messages,
        stream=False,
    )

    response = completion.choices[0].message.content

    if "Output 1" in response:
        return True
    elif "Output 2" in response:
        return False
    else:
        return None

def calculate_output2_score(prompts_file):
    total_prompts = 0
    output2_count = 0

    with open(prompts_file, "r", encoding="utf-8") as file:
        for line in file:
            prompt = line.strip()
            result = compare(raw_chat, rag_chat, prompt)
            if result is False:
                output2_count += 1
            total_prompts += 1

    if total_prompts > 0:
        output2_percentage = (output2_count / total_prompts) * 100
        print(f"Output 2 Score: {output2_percentage}%")
    else:
        print("No available prompts.")

if __name__ == "__main__":
    prompts_file = "prompts.txt"
    calculate_output2_score(prompts_file)
