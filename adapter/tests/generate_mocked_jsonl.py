import json
from faker import Faker

from fake_pii_generator import FakePIIGenerator


def generate_mock_conversations(filename, num_conversations=5):
    fake_pii_generator = FakePIIGenerator(locale="en_US")
    conversations = []

    for i in range(num_conversations):
        conversation = {
            "conversation_id": i,
            "messages": [
                {
                    "role": "user",
                    "content": f"你好，我的名字是{fake_pii_generator.get_fake_value('person_name', '张三')}。",
                },
                {
                    "role": "assistant",
                    "content": f"你好，{fake_pii_generator.get_fake_value('person_name', '张三')}！很高兴见到你。",
                },
                {
                    "role": "user",
                    "content": f"我的电子邮件是{fake_pii_generator.get_fake_value('email_address', 'zhangsan@example.com')}。",
                },
                {"role": "assistant", "content": "好的，我会记住你的联系方式。"},
                {
                    "role": "user",
                    "content": f"我住在{fake_pii_generator.get_fake_value('street_address', '北京市朝阳区')}。",
                },
                {"role": "assistant", "content": "了解了，谢谢你的信息。"},
            ],
        }
        conversations.append(conversation)

    with open(filename, "w") as fout:
        for conversation in conversations:
            fout.write(json.dumps(conversation) + "\n")


generate_mock_conversations("input.jsonl")
