import json


def generate_jsonl(filename, num_conversations=5):
    conversations = []

    for i in range(num_conversations):
        conversation = {
            "conversation_id": i,
            "messages": [
                {"role": "user", "content": f"你好，助手！这是我的第{i+1}次对话。"},
                {
                    "role": "assistant",
                    "content": f"你好！很高兴再次见到你。这是我们的第{i+1}次对话。",
                },
                {"role": "user", "content": f"我想了解一下关于PII处理的更多信息。"},
                {
                    "role": "assistant",
                    "content": f"当然可以！PII处理涉及识别和保护个人身份信息。",
                },
            ],
        }
        conversations.append(conversation)

    with open(filename, "w") as fout:
        for conversation in conversations:
            fout.write(json.dumps(conversation) + "\n")


generate_jsonl("input.jsonl")
