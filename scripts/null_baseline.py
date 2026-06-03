import re
import pandas as pd
import click
import yaml

def baseline(row):
        'преобразует диалог к форме Первая + Последняя строка'

        df_0 = row['dialogue'].splitlines() # строка

        L = len(df_0)
        first = df_0[0]
        last = df_0[L - 1]

        text_0 = first + last
        
        pattern = r"\#Person\d\#:"

        text = re.sub(pattern, '', text_0).strip()

        text = " ".join(text.split())

        return text

@click.command()
@click.option("--config", required=True, help="Path to YAML config")
def main(config):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv('data/test.csv')
    df = df.drop_duplicates(subset=['dialogue']) # удаляем дубликаты
    df['null_baseline'] = df.apply(baseline, axis=1) # первая + последняя строки

    df.to_csv('results/null_df.csv', index=False)

if __name__ == "__main__":
    main()