import requests
import time
import random
from dotenv import dotenv_values
from pathlib import Path
from tqdm import tqdm

# Load variables directly from the .env file
CONFIG = dotenv_values(".env")  # Specify the path to .env file
# Fetch the API key from the .env configuration

api_key = CONFIG.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set GROQ_API_KEY in your .env file.")

URL = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

SAMPLES = 50
fails = 0
saveDir = Path("./Data/CSV/")
saveDir.mkdir(parents=True, exist_ok=True)

for sample in tqdm(range(SAMPLES), desc="Processing ", colour="#bd4ced"):
    NUM_COLUMNS = random.randint(2, 12)
    NUM_ROWS = random.randint(10, 30)
    BASE_PROMPT = """I want you to make up a CSV with some data for me. Create 5 columns and come up with headers for the CSV. For these headers you came up with generate suitable data to fill each row, usually business related data. Each row should be in a separate line, we need a valid CSV. You have the flexibility to think of different kinds of data this could be but it must be in the CSV format. Create 20 rows. 
Don't give any extra information. Respond purely with the CSV alone, then leave exactly one line and give a title for the CSV. 
                  """

    payload = {
        "messages": [{"role": "user", "content": BASE_PROMPT}],
        "model": "llama-3.1-70b-versatile",
        "temperature": 1,
        "max_tokens": 4096,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }

    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]
            content = content.split("\n")
            title = content[-1].strip()
            title = title.removesuffix("csv")
            content = content[:-3]

            saveFilename = f"{title}.csv"
            try:
                with open(saveDir / saveFilename, "w") as file:
                    for row in content:
                        file.write(row)
                        file.write("\n")
            except Exception as e:
                print(f"{sample}/{SAMPLES} : {e}\nSkipping.")
        except KeyError:
            fails += 1
    else:
        # Handle non-200 status codes
        print(f"API request failed with status code: {response.status_code}")
        fails += 1

    time.sleep(10)

print(f"Done. Failed to parse {fails} files.")
