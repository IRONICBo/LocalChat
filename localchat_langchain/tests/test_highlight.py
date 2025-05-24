a = """
 <think>
嗯，用户问“你好！很高兴见到你，有什么我可以帮忙的吗？”。看起来他们是在打招呼，可能在聊天。接着，用户又说“你好！很高兴见到你，有什么我可以帮忙的吗？”这让我有点困惑，可能是重复了之前的问候。

然后，用户进一步询问：“你好！你是什么模型。” 这里明显提到了“模型”，但没有提供具体的信息。我需要理解用户的意图是什么。他们可能是在讨论AI技术，特别是生成模型，比如大语言模型（LLM）。也有可能是想了解关于生成模型的一些知识或应用。

考虑到用户提到“模型”这个词，我猜测他们可能对生成模型感兴趣，尤其是用于文本生成的应用。因此，我决定进一步询问他们的具体需求，以便更好地帮助他们。

所以，我的回复应该是欢迎他们提问，解释一下什么是模型，并说明为什么我会关注你，这样可以更有效地帮助他们。
</think>

你好！很高兴见到你，有什么我可以帮忙的吗？无论是学习、工作还是生活中的问题，都可以告诉我哦！
"""

import re


def convert_highlight_thinktext(text):
    """Convert <think> content to highlighted lines with > DeepThink:."""

    def format_think_content(content):
        lines = content.splitlines()
        # Line 1 with DeepThink label
        formatted_lines = [f"> DeepThink: {lines[0].strip()}"]
        # Line 2 and beyond
        for line in lines[1:]:
            formatted_lines.append(f"> {line.strip()}")
        return "\n".join(formatted_lines)

    new_text = re.sub(
        r"<think>(.*?)</think>",
        lambda match: f"{format_think_content(match.group(1))}",
        text,
        flags=re.DOTALL,
    )
    return new_text


def convert_highlight_thinktext1(text):
    """Convert text to HTML with highlighted text."""
    import re

    new_text = re.sub(
        r"<think>.*?</think>",
        lambda match: f"> DeepThink: {match.group(0)[7:-8].strip()}",
        text,
        flags=re.DOTALL,
    )
    return new_text


print(convert_highlight_thinktext(a))
