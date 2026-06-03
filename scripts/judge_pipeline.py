from groq import Groq
import os
import pandas as pd
from dotenv import load_dotenv
import json
import time
from groq import APIConnectionError, RateLimitError, APIStatusError
from tqdm.auto import tqdm
import re
import click
import yaml


@click.command()
@click.option("--config", required=True, help="Path to YAML config")
def main(config):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    load_dotenv()

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    df_full = pd.read_csv('results/df_full.csv')

    def build_judge_prompt(dialogue, reference, summary_a, summary_b):
        return f"""
    You are an expert evaluator of dialogue summarization.

    Your task is to compare two candidate summaries for the same dialogue.

    Evaluate them using:
    1. factual correctness
    2. coverage of important information
    3. conciseness
    4. clarity and fluency

    Dialogue:
    {dialogue}

    Reference summary:
    {reference}

    Summary A:
    {summary_a}

    Summary B:
    {summary_b}

    Return ONLY valid JSON with this schema:
    {{
    "winner": "A" or "B" or "Tie",
    "score_a": integer from 1 to 5,
    "score_b": integer from 1 to 5,
    "reason": "brief explanation"
    }}
    """

    JUDGE_MODEL = "llama-3.1-8b-instant"

    def parse_judge_answer(text):
        text = text.strip()

        # убираем markdown fences
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        # вытаскиваем JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            text = match.group(0)

        return json.loads(text)

    def judge_one(row, max_retries=5):
        prompt = build_judge_prompt(
            dialogue=row["dialogue"],
            reference=row["summary"],
            summary_a=row["qwen_zero_shot"],
            summary_b=row["lora"],
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict and fair evaluator. Return only valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                    max_tokens=300,
                )

                text = response.choices[0].message.content.strip()

                try:
                    return parse_judge_answer(text)
                except json.JSONDecodeError:
                    return {
                        "winner": "parse_error",
                        "score_a": None,
                        "score_b": None,
                        "reason": text,
                    }

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                wait = 2 ** attempt
                print(f"Groq error: {type(e).__name__}. Retry in {wait}s...")
                time.sleep(wait)

        return {
            "winner": "api_error",
            "score_a": None,
            "score_b": None,
            "reason": "Groq API failed after retries",
        }

    judge_df = df_full.sample(100, random_state=42).copy()

    judge_results = []

    for _, row in tqdm(judge_df.iterrows(), total=len(judge_df)):
        result = judge_one(row)
        judge_results.append(result)
        time.sleep(1.5)

    judge_results_df = pd.DataFrame(judge_results)

    judge_eval_df = pd.concat(
        [
            judge_df.reset_index(drop=True),
            judge_results_df,
        ],
        axis=1,
    )

    judge_eval_df["winner"].value_counts()
    valid_judge = judge_eval_df[judge_eval_df["winner"].isin(["A", "B", "Tie"])]
    valid_judge["winner"].value_counts(normalize=True)
    valid_judge[["score_a", "score_b"]].mean()

    cols = ["dialogue", "summary", "qwen_zero_shot", "lora", "winner", "score_a", "score_b", "reason"]

    valid = valid_judge[valid_judge["winner"] == "A"][cols]

    valid.to_csv(f'{JUDGE_MODEL}.csv', index=False)

if __name__ == "__main__":
    main()