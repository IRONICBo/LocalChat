from openai import OpenAI

prompt = "Subject: 数学\nQuestion: 1 + 1 = ?\nOptions:\nA: 1\nB: 2\nC: 3\nD: 4\nAnswer:"

messages  = [{"role": "user", "content": prompt}]

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:11434/v1",
)
completion = client.chat.completions.create(
    model="qwen:0.5b",
    messages=messages,
    temperature=0.1,
    max_tokens=1024,
    stream=False,
)

# Eval result
print(completion)
# ChatCompletion(id='chatcmpl-586', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The correct answer is D: 4. \n\nWhen you add two numbers, the result will be four. So, the correct answer is D: 4.', refusal=None, role='assistant', function_call=None, tool_calls=None))], created=1729823169, model='qwen:0.5b', object='chat.completion', service_tier=None, system_fingerprint='fp_ollama', usage=CompletionUsage(completion_tokens=34, prompt_tokens=51, total_tokens=85, completion_tokens_details=None))