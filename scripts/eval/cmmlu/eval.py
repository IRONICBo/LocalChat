import os
import argparse
import pandas as pd
import json
import time
from collections import defaultdict
import openai
from glob import glob
from tqdm import tqdm
from categories import name_en2zh, subcategories, categories
from evaluator import Evaluator

choices = ["A", "B", "C", "D"]

category2subject = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject[k].append(subject)
category2subject_list = defaultdict(list)
for key, value in category2subject.items():
    for val in value:
        category2subject_list[val] = [val, name_en2zh[val], key]
category2subject = category2subject_list

choices = ["A", "B", "C", "D"]


def openai_infer(prompt, temperature=0.2):
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:11434/v1",
    )
    response = client.chat.completions.create(
        model="qwen:0.5b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    answer = response.choices[0].message.content.strip()
    return answer


def eval_subject(
    subject_name,
    val_df,
    dev_df=None,
    few_shot=True,
    cot=False,
    checkpoint_file=None,
    completed_questions=None,
):
    correct = 0
    answers = []
    print(f"Total number of questions: {len(val_df)}")

    # Load checkpoint for current subject
    if checkpoint_file:
        _, sub_answers, sub_summary = load_checkpoint(checkpoint_file)
        answer = sub_answers.get(subject_name, [])
        correct = sub_summary.get("correct", 0)

    # start from the last processed question
    raw_val_df = val_df.copy()
    val_df = val_df.iloc[len(answers) :]
    print(
        f"Number of questions to be processed: {len(val_df)} starting from {len(answers)} with {correct} correct answers."
    )

    for idx, row in tqdm(val_df.iterrows()):
        question = row["Question"]
        options = "\n".join([f"{choice}: {row[choice]}" for choice in choices])
        answer = row["Answer"]
        print(
            f"\nCurrent idx: {idx} Question: {question}"
            + "\n"
            + options
            + "\n"
            + f"Answer: {answer}"
        )

        prompt = f"Subject: {subject_name}\nQuestion: {question}\nOptions:\n{options}\nAnswer:"
        if cot:
            prompt = "Let's think step-by-step to find the correct answer.\n" + prompt

        predicted_answer = openai_infer(prompt)
        print(f"Predicted Answer: {predicted_answer.upper()}")

        evaluator = Evaluator(choices=answer if answer is list else [answer])
        if evaluator.contains_valid_choice(predicted_answer):
            correct += 1
            print("Correct!")
        else:
            print("Incorrect!")

        answers.append(predicted_answer)

        # Load current progress
        if checkpoint_file and (idx + 1) % 10 == 0:
            print("Saving progress...")
            sub_answers = {}
            sub_answers[subject_name] = answers
            sub_summary = {}
            sub_summary[subject_name] = {
                "score": correct / (idx + 1) * 100,
                "num": idx + 1,
                "correct": correct,
            }
            save_checkpoint(
                checkpoint_file, completed_questions, sub_answers, sub_summary
            )
            print(f"Progress saved in {checkpoint_file} with {idx + 1} questions.")

    accuracy = correct / len(raw_val_df) * 100

    # clear checkpoint
    if checkpoint_file:
        os.remove(checkpoint_file)
    return accuracy, answers


def save_checkpoint(checkpoint_file, completed_subjects, all_answers, summary):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_subjects": list(completed_subjects),
                "all_answers": all_answers,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
        )


def load_checkpoint(checkpoint_file):
    # summary[subject_name] = {
    #     "score": correct_ratio,
    #     "num": len(val_df),
    #     "correct": correct_ratio * len(val_df) / 100,
    # }
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (
                set(data.get("completed_subjects", [])),
                data.get("all_answers", {}),
                data.get("summary", {}),
            )
    return set(), {}, {}


def main(args, take):
    # finished subjects and answers dump
    checkpoint_all_file = os.path.join(
        args.output_dir, f"checkpoint_all_take{take}.json"
    )
    # unfinished current subject and answers dump
    checkpoint_subject_file = os.path.join(
        args.output_dir, f"checkpoint_subject_take{take}.json"
    )

    subject_mapping = category2subject
    filenames = [s.split("/")[-1] for s in glob(args.input_dir + "/test/*csv")]
    print(filenames)
    subject_list = [val_file.replace(".csv", "") for val_file in filenames]
    accuracy, summary = {}, {}
    completed_subjects, all_answers, summary = load_checkpoint(checkpoint_all_file)

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    output_dir = args.output_dir
    save_result_dir = os.path.join(output_dir, f"take{take}")
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    try:
        for index, subject_name in enumerate(subject_list):
            if subject_name in completed_subjects:
                print(f"Skipping completed subject: {subject_name}")
                continue

            print(
                f"{index / len(subject_list)} Inference starts at {run_date} with subject of {subject_name}!"
            )
            val_file_path = os.path.join(
                args.input_dir + "/test", f"{subject_name}.csv"
            )
            dev_file_path = os.path.join(args.input_dir + "/dev", f"{subject_name}.csv")

            val_df = pd.read_csv(val_file_path)
            dev_df = pd.read_csv(dev_file_path) if args.few_shot else None

            correct_ratio, answers = eval_subject(
                subject_name,
                val_df,
                dev_df,
                few_shot=args.few_shot,
                cot=args.cot,
                checkpoint_file=checkpoint_subject_file,
                completed_questions=completed_subjects,
            )
            print(f"Subject: {subject_name}")
            print(f"Acc: {correct_ratio}")
            accuracy[subject_name] = correct_ratio
            summary[subject_name] = {
                "score": correct_ratio,
                "num": len(val_df),
                "correct": correct_ratio * len(val_df) / 100,
            }
            all_answers[subject_name] = answers

            # Update completed subjects
            completed_subjects.add(subject_name)
            save_checkpoint(
                checkpoint_all_file, completed_subjects, all_answers, summary
            )

    except KeyboardInterrupt:
        print("Interrupted! Saving progress...")
        save_checkpoint(checkpoint_all_file, completed_subjects, all_answers, summary)
        print("Progress saved.")

    # Save results
    json.dump(
        all_answers,
        open(save_result_dir + "/submission.json", "w"),
        ensure_ascii=False,
        indent=4,
    )
    print("\n\nAccuracy:")
    for k, v in accuracy.items():
        print(k, ": ", v)

    total_num = 0
    total_correct = 0
    summary["grouped"] = {
        "China specific": {"correct": 0.0, "num": 0},
        "STEM": {"correct": 0.0, "num": 0},
        "Social Science": {"correct": 0.0, "num": 0},
        "Humanities": {"correct": 0.0, "num": 0},
        "Other": {"correct": 0.0, "num": 0},
    }
    for subj, info in subject_mapping.items():
        print(summary)
        print(summary[subj]["num"])
        print(summary["grouped"][group]["num"])
        group = info[2]
        summary["grouped"][group]["num"] += summary[subj]["num"]
        summary["grouped"][group]["correct"] += summary[subj]["correct"]
    for group, info in summary["grouped"].items():
        info["score"] = info["correct"] / info["num"]
        total_num += info["num"]
        total_correct += info["correct"]
    summary["All"] = {
        "score": total_correct / total_num,
        "num": total_num,
        "correct": total_correct,
    }

    json.dump(
        summary,
        open(save_result_dir + "/summary.json", "w"),
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--cot", choices=["False", "True"], default="False")
    parser.add_argument("--few_shot", choices=["False", "True"], default="True")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n_times", default=1, type=int)
    parser.add_argument("--do_save_csv", choices=["False", "True"], default="False")
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--input_dir", type=str)

    args = parser.parse_args()

    args.cot = args.cot == "True"
    args.few_shot = args.few_shot == "True"
    args.do_save_csv = args.do_save_csv == "True"
    if args.cot is True:
        args.n_times = max(args.n_times, 1)
    print(args)

    for i in range(args.n_times):
        main(args, take=i)
