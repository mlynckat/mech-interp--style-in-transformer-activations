from backend.src.steering.classification.read_data import DataReader
from sklearn.model_selection import train_test_split
import os
import json
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_prompts_for_test(article: str):
    """Get prompts for test"""

    prompt_for_generation = f"Read the following article and generate a short prompt for a large language model to write an article on the same topic without passing the original article to the model or too many details.  Return only the prompt without any additional text. It should write an article on the same topic without any additional input: {article}."

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_for_generation},
                ],
            }
        ],
    )
    print(response.output[0].content[0].text)
    print("-"*20)
    return response.output[0].content[0].text

def main():

    base_dir = Path("data/steering/tests")
    base_dir.mkdir(parents=True, exist_ok=True)

    output_test_data = []

    X, y = DataReader.read_news_json_data()

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    y_test_list = y_test.tolist()
    X_test_list = X_test.tolist()
    for i, article in tqdm(enumerate(X_test_list)):
        prompt = get_prompts_for_test(article)
        output_test_data.append({
            "article": article,
            "prompt": prompt,
            "author": y_test_list[i],
        })
    
    with open(base_dir / "prompts_test_data.json", "w") as f:
        json.dump(output_test_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()


    



