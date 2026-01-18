from calendar import c
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

    prompt_for_generation = f"""Read the following article and generate a prompt for a large language model to write an article on the same topic.  Include short facts described in the article, people, dates, places mentioned. Only include short facts and nothing connected to the structure of the article or any stylistic details. Don't mention the author of the article and anything connected to them. Structure the prompt as follows:
First, prompt for action (same as in the original article): e.g., Write an article discussing .... <topic>.
Then include key facts for the article. \n Article: {article}"""

    response = client.responses.create(
        model= "gpt-4o-mini", #"gpt-5-mini",
        input=prompt_for_generation
    )

    print(response.output_text)
    print("-"*20)
    print(response)
    return response.output_text

def main(mode: str = "test", continue_generation: bool = False):

    base_dir = Path("data/steering/tests")
    base_dir.mkdir(parents=True, exist_ok=True)

    if continue_generation:
        with open(base_dir / f"prompts_{mode}_data__detailed.json", "r") as f:
            output_data = json.load(f)
        print([i["id"] for i in output_data])
    else:
        output_data = []

    X, y = DataReader.read_news_json_data()

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    if mode == "test":
        y_list = y_test.tolist()
        X_list = X_test.tolist()
    elif mode == "train":
        y_list = y_train.tolist()
        X_list = X_train.tolist()
    else:
        raise ValueError(f"Invalid mode: {mode}. Should be 'test' or 'train'.")

    new_output_data = []
    
    for i, article in tqdm(enumerate(X_list)):

        existing_indices = [i["id"] for i in output_data]
        
        if i in existing_indices and output_data[i]["article"][0:100] == article[0:100] and output_data[i]["prompt"]:
            new_output_data.append(output_data[i])
        else:
            print(f"Generating prompt for article {i}")
            if i in existing_indices:
                print(output_data[i]["article"])
                print("-"*20)
                print(article)
                print("*"*20)
                print(output_data[i]["prompt"])
                print("-"*20)
            
            prompt = get_prompts_for_test(article)
            new_output_data.append({
                "id": i,
                "article": article,
                "prompt": prompt,
                "author": y_list[i],
            })

        if i % 100 == 0:
            with open(base_dir / f"prompts_{mode}_data__detailed_{i}.json", "w") as f:
                json.dump(new_output_data, f, indent=2, ensure_ascii=False)

    
    with open(base_dir / f"prompts_{mode}_data__detailed.json", "w") as f:
        json.dump(new_output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(mode="train", continue_generation=True)


    



