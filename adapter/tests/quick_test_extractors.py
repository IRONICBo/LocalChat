# -*- coding: utf-8 -*-
"""
快速测试提取器（不依赖外部库）
"""

import json
import re
import sys

# 测试样本
test_samples = [
    {
        "full_text": "Lost card 4532 1111 2222 3333! Block it and send replacement to 123 Main St, New York, NY 10001 for John Smith.",
        "spans": [
            {"entity_type": "CREDIT_CARD", "start": 10, "end": 29, "entity_value": "4532 1111 2222 3333"},
            {"entity_type": "ADDRESS", "start": 65, "end": 97, "entity_value": "123 Main St, New York, NY 10001"},
            {"entity_type": "PERSON", "start": 102, "end": 112, "entity_value": "John Smith"}
        ]
    },
    {
        "full_text": "Email john.doe@example.com for support or call +1-555-123-4567",
        "spans": [
            {"entity_type": "EMAIL", "start": 6, "end": 26, "entity_value": "john.doe@example.com"},
            {"entity_type": "PHONE", "start": 48, "end": 64, "entity_value": "+1-555-123-4567"}
        ]
    }
]


def test_regex_extractor():
    """测试正则表达式提取器"""
    print("\n" + "=" * 60)
    print("测试正则表达式提取器")
    print("=" * 60)

    patterns = {
        "CREDIT_CARD": r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{2,4}',
    }

    for i, sample in enumerate(test_samples):
        print(f"\n样本 {i+1}:")
        print(f"文本: {sample['full_text']}")
        print(f"真实实体: {len(sample['spans'])}")

        # 提取
        extracted = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, sample['full_text']):
                extracted.append({
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "entity_value": match.group()
                })

        print(f"提取实体: {len(extracted)}")
        for ent in extracted:
            print(f"  - {ent['entity_type']}: {ent['entity_value']} [{ent['start']}:{ent['end']}]")

        # 简单评估
        gt_positions = {(e["start"], e["end"]) for e in sample['spans']}
        pred_positions = {(e["start"], e["end"]) for e in extracted}

        tp = len(gt_positions & pred_positions)
        fp = len(pred_positions - gt_positions)
        fn = len(gt_positions - pred_positions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    print("\n✅ 正则表达式提取器测试完成")


def test_load_dataset():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("测试数据集加载")
    print("=" * 60)

    dataset_path = "../data/generated_size_10000_en.jsonl"

    try:
        count = 0
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只读取前5条
                    break

                data = json.loads(line)
                count += 1

                if i == 0:
                    print(f"\n样本示例:")
                    print(f"  文本: {data['full_text'][:100]}...")
                    print(f"  实体数: {len(data.get('spans', []))}")
                    if data.get('spans'):
                        print(f"  第一个实体: {data['spans'][0]}")

        print(f"\n✅ 成功加载数据集，测试读取了 {count} 条样本")

    except FileNotFoundError:
        print(f"\n❌ 数据集文件不存在: {dataset_path}")
    except Exception as e:
        print(f"\n❌ 加载数据集失败: {e}")


def test_evaluation_metrics():
    """测试评估指标计算"""
    print("\n" + "=" * 60)
    print("测试评估指标计算")
    print("=" * 60)

    # 模拟预测和真实标注
    predicted = [
        {"entity_type": "CREDIT_CARD", "start": 10, "end": 29},
        {"entity_type": "EMAIL", "start": 50, "end": 70},
        {"entity_type": "PHONE", "start": 100, "end": 115},  # 误报
    ]

    ground_truth = [
        {"entity_type": "CREDIT_CARD", "start": 10, "end": 29},
        {"entity_type": "EMAIL", "start": 50, "end": 70},
        {"entity_type": "PERSON", "start": 200, "end": 210},  # 漏报
    ]

    pred_set = {(e["entity_type"], e["start"], e["end"]) for e in predicted}
    gt_set = {(e["entity_type"], e["start"], e["end"]) for e in ground_truth}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n预测实体数: {len(predicted)}")
    print(f"真实实体数: {len(ground_truth)}")
    print(f"\nTrue Positive: {tp}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n✅ 评估指标计算测试完成")


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "PII 提取器快速测试".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    try:
        test_regex_extractor()
        test_load_dataset()
        test_evaluation_metrics()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 安装依赖: pip install pandas openpyxl requests")
        print("2. 运行完整测试: python3 pii_extraction_comparison.py")
        print()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
