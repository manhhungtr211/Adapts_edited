import json
from tqdm import tqdm
import random


def flatten_table(table):
    return "\n".join(
        " | ".join(str(cell).strip() for cell in row)
        for row in table
    )

def build_dataset(samples):
    new_data = []
    id_sample = 0
    for sample in tqdm(samples):

        annotation = sample["annotation"]

        # turn hiện tại
        turn_ind = annotation["turn_ind"]

        questions = annotation["dialogue_break"]
        answers = annotation["exe_ans_list"]
        label = annotation["turn_program"][-1]

        pre_text = " ".join(sample["pre_text"])
        post_text = " ".join(sample["post_text"])
        table_text = flatten_table(sample["table"])

        context_parts = [
            pre_text,
            table_text,
            post_text,
        ]

        # lịch sử QA trước turn hiện tại
        for i in range(turn_ind):
            context_parts.append(
                f"{questions[i]} {answers[i]}"
            )

        # câu hỏi hiện tại
        context_parts.append(
            f"{questions[turn_ind]}"
        )
        new_data.append({
            "input": "\n\n".join(context_parts),
            "label": str(label)
        })

    return new_data


# Load data gốc
with open("train_turn.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Build dataset mới
new_data = build_dataset(data)
random.seed(42)      
random.shuffle(new_data)

for idx, item in enumerate(new_data):
    item["id"] = idx

# Save
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(new_data)} samples")