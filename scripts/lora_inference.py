import pandas as pd
from unsloth import FastLanguageModel
from peft import PeftModel
import click
import yaml

def build_prompt(dialogue):
        return f"""<|im_start|>system
    You are a helpful dialogue summarization assistant.
    Generate ONLY one concise summary sentence.
    <|im_end|>

    <|im_start|>user
    Summarize the following dialogue.

    Dialogue:
    {dialogue}
    <|im_end|>

    <|im_start|>assistant
    """

@click.command()
@click.option("--config", required=True, help="Path to YAML config")
def main(config):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(base_model, "./qwen_lora_final")
    FastLanguageModel.for_inference(model)

    df_test = pd.read_csv('data/test.csv')
    df_test = df_test.drop_duplicates(subset=['dialogue'])

    df_zero = pd.read_csv('results/zero_shot_df.csv')

    sample_df = df_test.copy()

    prompts = [
        build_prompt(x)
        for x in sample_df["dialogue"].tolist()
    ]

    outputs = []

    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

        text = tokenizer.decode(
            out[0],
            skip_special_tokens=True
        )

        outputs.append(text)

    predictions = []

    for text in outputs:

        pred = text.split("assistant")[-1].strip()

        predictions.append(pred)

    df_zero['lora'] = predictions[:len(df_zero)]
    df_zero.to_csv('results/df_full.csv', index=False)

if __name__ == "__main__":
    main()