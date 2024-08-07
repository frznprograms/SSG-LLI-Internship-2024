# -*- coding: utf-8 -*-
"""
### Imports ###
"""

!pip install openai
import pandas as pd
import numpy as np
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt

"""### Prepare Data ###"""

thirty_thousand_data = pd.read_excel("30k_magic.xlsx")
print(thirty_thousand_data.head())

def clean_and_process_data(dataset):
  # Drop rows where text cannot be converted to a valid string
  def is_valid_string(value):
    try:
      str(value)
      return True
    except ValueError:
      return False

  dataset['Valid'] = dataset['Case Notes'].apply(is_valid_string)
  dataset = dataset[dataset['Valid'] == True]
  dataset = dataset.drop('Valid', axis = 1)
  return dataset

cleaned_data = clean_and_process_data(thirty_thousand_data)
print(cleaned_data.head(), len(cleaned_data))
split_data = np.array_split(cleaned_data, 30)

"""### Prompts ###"""

API_KEY = "<Insert API Key here>"

system_prompt = f"""
"""

classification_prompt = f""" """

"""### Classifier Class ###"""

class Classifier:
  def __init__(self, API_KEY, model_type, purpose = None):
    self.client = OpenAI(api_key = API_KEY)
    self.model = model_type
    self.purpose = purpose
    self.class_ = "Classifier"
    self.conversation_history = []

  def get_model_type(self):
    return self.model

  def get_purpose(self):
    return self.purpose

  def set_new_model(self, new_model):
    self.model = new_model

  def set_new_prupose(self, new_purpose):
    self.purpose = new_purpose

  def get_class_type(self):
    return self.class_

  def get_gpt_summary(self):
    info = {
        "model" : self.model,
        "class" : self.class_,
        "purpose" : self.purpose,
        "past_queries" : self.past_queries,
        "conversation_history" : self.conversation_history
    }
    return info

  def get_conversation_history(self):
    return self.conversation_history

  def clear_history(self):
    self.conversation_history = []

  def classify(self, prompt, case_note, temp = 0.2, top_p = 0.8, max_tokens = 1): # will classify
    response = self.client.chat.completions.create(
        model = self.model,
        messages = [
            {"role": "system", "content": self.purpose},
            {"role": "user", "content": prompt + str(case_note)}
          ],
        temperature = temp,
        top_p = top_p,
        max_tokens = max_tokens
    )
    answer = response.choices[0].message.content
    if answer is None or answer == '':
      return np.nan
    else:
      try:
        answer = int(answer)
      except:
        return np.nan

    self.conversation_history.append((prompt, answer))
    return answer

"""### Instantiate model and Classify ###"""

model = Classifier(API_KEY, "gpt-4o", purpose = system_prompt)

def process_data(split_data):
  for index, dataset in enumerate(split_data):
    print(f"Processing Dataset {index + 1} of 30.")
    results = []
    for index2, row in dataset.iterrows():
      result = model.classify(classification_prompt, row['Case Notes'])
      results.append(result)
    dataset['AI Classification'] = results
    dataset.to_excel(f"SubDataset_{index + 1}.xlsx", index = False)
    print(f"Dataset {index + 1} of 30 successfully processed.")
    model.clear_history()

  return "Processing complete!"

def merge_datasets():
  merged_data = pd.read_excel("SubDataset_1.xlsx")
  for index in range(2, 31):
    dataset = pd.read_excel(f"SubDataset_{index}.xlsx")
    merged_data = pd.concat([merged_data, dataset], axis = 'rows')

  return merged_data

"""### Test on just one dataset first ###"""

test = []
for index, row in split_data[0].head(500).iterrows():
  res = model.classify(classification_prompt, row['Case Notes'])
  test.append(res)

for index, number in enumerate(test):
  try:
    number = int(number)
  except:
    print(index, number)

print('processing done')
print(f"Length of output: {len(test)}")
print(f"Unique values: {set(test)}")
print(f"Count: {len(test)}")

"""### Now classify all datasets ###"""

process_data(split_data)

"""Now check the files -> should have 30 separate datasets, each with a new category"""

merged_data = merge_datasets()
merged_data.to_excel("<name>.xlsx", index = False)
