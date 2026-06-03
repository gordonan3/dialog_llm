import pandas as pd
from vllm import LLM, SamplingParams
import click
import yaml

def build_prompt(dialogue: str) -> str:
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

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    df = pd.read_csv('data/null_df.csv')

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=80,
        stop=["<|im_end|>", "\n\n"],
    )

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.65,
        max_model_len=2048,
        enforce_eager=True,
    )

    prompts = [build_prompt(x) for x in df["dialogue"].astype(str).tolist()]

    outputs = llm.generate(prompts, sampling_params)

    df["qwen_zero_shot"] = [
        out.outputs[0].text.strip()
        for out in outputs
    ]

    df.to_csv('results/zero_shot_df.csv', index=False)

if __name__ == "__main__":
    main()