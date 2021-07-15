import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline

from pprint import pprint
import re


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)


def skip_special_tokens_and_prettify(text, tokenizer):
    recipe_maps = {"<sep>": "--", "<section>": "\n"}
    recipe_map_pattern = "|".join(map(re.escape, recipe_maps.keys()))

    text = re.sub(
        recipe_map_pattern, 
        lambda m: recipe_maps[m.group()], 
        re.sub("|".join(tokenizer.all_special_tokens), "", text)
    )

    data = {"title": "", "ingredients": [], "directions": []}
    for section in text.split("\n"):
        section = section.strip()
        section = section.strip()
        if section.startswith("title:"):
            data["title"] = section.replace("title:", "").strip()
        elif section.startswith("ingredients:"):
            data["ingredients"] = [s.strip() for s in section.replace("ingredients:", "").split('--')]
        elif section.startswith("directions:"):
            data["directions"] = [s.strip() for s in section.replace("directions:", "").split('--')]
        else:
            pass

    return data


def post_generator(output_tensors, tokenizer):
    output_tensors = [output_tensors[i]["generated_token_ids"] for i in range(len(output_tensors))]
    texts = tokenizer.batch_decode(output_tensors, skip_special_tokens=False)
    texts = [skip_special_tokens_and_prettify(text, tokenizer) for text in texts]
    return texts


# Example 
generate_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95,
    "num_return_sequences": 3
}
# items = "potato, cheese"
# generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
# generated = generator(items, return_tensors=True, return_text=False, **generate_kwargs)
# outputs = post_generator(generated, tokenizer)
# pprint(outputs)