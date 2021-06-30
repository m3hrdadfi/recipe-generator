# Recipe Generator


## Project Description
Given a list of ingredients, generate a recipe – similar to what the GPT3 API offers: OpenAI recipe generator. A model exists on the hub that does Recipe NLG, but it uses a GPT2 architecture. I’m curious if T5 or Bart will produce better results.

## Model:
T5, ByT5, Bart, GPT-2, GPT-3.

## Dataset:
Recipe NLG (2,231,142 recipe examples) - hugging face link - download site

## Data example:

```json
{
  "NER": [
    "oyster crackers",
    "salad dressing",
    "lemon pepper",
    "dill weed",
    "garlic powder",
    "salad oil"
  ],
  "directions": [
    "Combine salad dressing mix and oil.",
    "Add dill weed, garlic powder and lemon pepper.",
    "Pour over crackers; stir to coat.",
    "Place in warm oven.",
    "Use very low temperature for 15 to 20 minutes."
  ],
  "ingredients": [
    "12 to 16 oz. plain oyster crackers",
    "1 pkg. Hidden Valley Ranch salad dressing mix",
    "1/4 tsp. lemon pepper",
    "1/2 to 1 tsp. dill weed",
    "1/4 tsp. garlic powder",
    "3/4 to 1 c. salad oil"
  ],
  "link": "www.cookbooks.com/Recipe-Details.aspx?id=648947",
  "source": "Gathered",
  "title": "Hidden Valley Ranch Oyster Crackers"
}
```

## Expected result:
Give it a list of ingredients (e.g. sugar, flour, egg, peanut butter) and it spits out a recipe (peanut butter cookies). Useful for when you are trying to use up all those miscellaneous ingredients in your pantry and fridge!
