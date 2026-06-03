import pandas as pd
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import Dataset
import click
import yaml

def make_text(row):
        return f"""<|im_start|>system
    You are a helpful dialogue summarization assistant.
    Generate ONLY one concise summary sentence.
    <|im_end|>

    <|im_start|>user
    Summarize the following dialogue.

    {row["dialogue"]}
    <|im_end|>

    <|im_start|>assistant
    {row["summary"]}<|im_end|>"""

@click.command()
@click.option("--config", required=True, help="Path to YAML config")
def main(config):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    df_train = pd.read_csv('data/train.csv')

    df_valid = pd.read_csv('data/validation.csv')

    df_train = df_train.drop_duplicates(subset=['dialogue'])

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir="./qwen_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True,
    )

    train_df_small = df_train.iloc[:2000].copy()
    valid_df_small = df_valid.iloc[:300].copy()

    train_df_small["text"] = train_df_small.apply(make_text, axis=1)
    valid_df_small["text"] = valid_df_small.apply(make_text, axis=1)

    train_dataset = Dataset.from_pandas(train_df_small[["text"]])
    valid_dataset = Dataset.from_pandas(valid_df_small[["text"]])

    training_args = SFTConfig(
        output_dir="./qwen_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        fp16=False,
        packing=False,
        dataset_text_field="text",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./qwen_lora_final")
    tokenizer.save_pretrained("./qwen_lora_final")

if __name__ == "__main__":
    main()