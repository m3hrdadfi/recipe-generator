import json

import streamlit as st
from streamlit_tags import st_tags
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import beam_search
import top_sampling

with open("config.json") as f:
    cfg = json.loads(f.read())

st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name_or_path"])
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator, tokenizer


def sampling_changed(obj):
    print(obj)


predefined_ingredients = ["parmesan cheese", "fresh oregano", "basil", "whole wheat flour"]
custom_ingredients = ["salt"]

with st.spinner("Loading model..."):
    generator, tokenizer = load_model()

st.header("Chef Transformer 👩‍🍳 / 👨‍🍳")
st.markdown(
    "This demo uses [t5 trained on RecipeNLG](https://huggingface.co/flax-community/t5-recipe-generation) "
    "to generate recipe from a given set of ingredients"
)
img = st.sidebar.image("images/chef-transformer-transparent.png", width=310)
add_text_sidebar = st.sidebar.title("Popular Recipes")

preset_recipe1 = st.sidebar.button("Parmesan Herb Crackers")
if preset_recipe1:
    predefined_ingredients = ["parmesan cheese", "fresh oregano", "basil", "whole wheat flour"]
    custom_ingredients = ["salt"]

preset_recipe2 = st.sidebar.button("Lemon Herb Chicken")
if preset_recipe2:
    predefined_ingredients = ["fresh oregano", "basil", "chicken breasts"]
    custom_ingredients = ["salt", "lemon rind"]

add_text_sidebar = st.sidebar.title("Mode")
sampling_mode = st.sidebar.selectbox("Select a Mode", index=0, options=["Top Sampling", "Beam Search"])

original_keywords = st.multiselect(
    "Choose ingredients",
    cfg["first_100"],
    predefined_ingredients,
)

st.write("Add custom ingredients here:")
custom_keywords = st_tags(
    label="",
    text="Press enter to add more",
    value=custom_ingredients,
    suggestions=cfg["next_100"],
    maxtags=15,
    key="1",
)

all_ingredients = []
all_ingredients.extend(original_keywords)
all_ingredients.extend(custom_keywords)
all_ingredients = ", ".join(all_ingredients)
st.markdown("**Generate recipe for:** " + all_ingredients)


submit = st.button("Get Recipe!")
if submit:
    with st.spinner("Generating recipe..."):
        if sampling_mode == "Beam Search":
            generated = generator(
                all_ingredients, return_tensors=True, return_text=False, **beam_search.generate_kwargs
            )
            outputs = beam_search.post_generator(generated, tokenizer)
        elif sampling_mode == "Top Sampling":
            generated = generator(
                all_ingredients, return_tensors=True, return_text=False, **top_sampling.generate_kwargs
            )
            outputs = top_sampling.post_generator(generated, tokenizer)
    output = outputs[0]
    markdown_output = ""
    markdown_output += f"## {output['title'].title()}\n"
    markdown_output += f"#### Ingredients:\n"
    for o in output["ingredients"]:
        markdown_output += f"- {o}\n"
    markdown_output += f"#### Directions:\n"
    for o in output["directions"]:
        markdown_output += f"- {o}\n"
    st.markdown(markdown_output)
