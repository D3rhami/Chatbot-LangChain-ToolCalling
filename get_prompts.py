import json


def load_prompts():
    with open("prompts.json", "r") as file:
        return json.load(file)


prompts = load_prompts()