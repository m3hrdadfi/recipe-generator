from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime as dt  
import streamlit as st
from streamlit_tags import st_tags
import beam_search
import top_sampling
from pprint import pprint
import json

with open("config.json") as f:
    cfg = json.loads(f.read())

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-recipe-generation")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return generator, tokenizer

def sampling_changed(obj):
    print(obj)
    

with st.spinner('Loading model...'):
    generator, tokenizer = load_model()
# st.image("images/chef-transformer.png", width=400)
st.header("Chef transformers (flax-community)")
st.markdown("This demo uses [t5 trained on recipe-nlg](https://huggingface.co/flax-community/t5-recipe-generation) to generate recipe from a given set of ingredients")
img = st.sidebar.image("images/chef-transformer.png", width=200)
add_text_sidebar = st.sidebar.title("Popular recipes:")
add_text_sidebar = st.sidebar.text("Recipe preset(example#1)")
add_text_sidebar = st.sidebar.text("Recipe preset(example#2)")

add_text_sidebar = st.sidebar.title("Mode:")
sampling_mode = st.sidebar.selectbox("select a Mode", index=0, options=["Beam Search", "Top-k Sampling"])


original_keywords = st.multiselect("Choose ingredients",
    cfg["first_100"],
    ["parmesan cheese", "fresh oregano", "basil", "whole wheat flour"]
)

st.write("Add custom ingredients here:")
custom_keywords = st_tags(
    label="",
    text='Press enter to add more',
    value=['salt'],
    suggestions=cfg["next_100"],
    maxtags = 15,
    key='1')
all_ingredients = []
all_ingredients.extend(original_keywords)
all_ingredients.extend(custom_keywords)
all_ingredients = ", ".join(all_ingredients)
st.markdown("**Generate recipe for:** "+all_ingredients)


submit = st.button('Get Recipe!')
if submit:
    with st.spinner('Generating recipe...'):
        if sampling_mode == "Beam Search":
            generated = generator(all_ingredients, return_tensors=True, return_text=False, **beam_search.generate_kwargs)
            outputs = beam_search.post_generator(generated, tokenizer)
        elif sampling_mode == "Top-k Sampling":
            generated = generator(all_ingredients, return_tensors=True, return_text=False, **top_sampling.generate_kwargs)
            outputs = top_sampling.post_generator(generated, tokenizer)
    output = outputs[0]
    markdown_output = ""
    markdown_output += f"## {output['title'].capitalize()}\n"
    markdown_output += f"#### Ingredients:\n"
    for o in output["ingredients"]:
        markdown_output += f"- {o}\n"
    markdown_output += f"#### Directions:\n"
    for o in output["directions"]:
        markdown_output += f"- {o}\n"
    st.markdown(markdown_output)
    st.balloons()

