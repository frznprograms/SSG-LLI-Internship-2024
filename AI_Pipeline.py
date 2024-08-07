# -*- coding: utf-8 -*-
"""
### Import Packages ###
"""

!pip install openai

from openai import OpenAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""### Import Data ###"""

train_data = pd.read_excel("GPT_FineTuning2-1.xlsx")
#eval_data = pd.read_excel("20 case notes_for shane-1.xlsx")
#test_data = pd.read_excel("TestData.xlsx")

train_data = train_data[['Note', 'ForAI']].head(100)
train_data = train_data.dropna(subset = ['ForAI'])
#eval_data = eval_data[['Note', 'ForAI']].head(10)
#test_data = test_data[['Note', 'Final']].head(50)

#test_data['Final'] = test_data['Final'].apply(lambda x: int(x))

#train_data = pd.concat([train_data, test_data], axis = 'rows')

len(train_data)

train_data.ForAI.value_counts()

"""### Create Prompts ###"""

definitions = {
    "CPF" : "CPF refers to the Central Provident Fund, which is an organisation that manages the CPF funds of Singaporeans. Singaporeans accumulate their CPF funds by\
      working, where a percentage of their income is automatically tranferred to their CPF fund and is supplemented by their employers. The CPF fund cannot be cashed out until\
      age 55, and is used mainly to pay for public housing, public healthcare, and other public services.",
    "HR" : "Human Resources",
    "SFC" : "SFC refers to SkillsFuture Credit. All Singapore Citizens aged 25 and above receive SFC from the government which can be spent on eligible skills upgrading\
      courses, career advisory and other skills and career-related services.",
    "MySF" : "MySF refers to the MySkillsFuture Portal. All Singapore Citizens have an account on this website by default and can explore different courses and services available.",
    "SA" : "SA refers to Skills Ambassador, someone who conducts consultations with Clients to identify their goals and obstacles and helps guides them in their journey\
      to gaining more skills or advancing in their career.",
    "SAHM" : "SAHM refers to a Stay at Home Mum.",
    "MOE" : "MOE refers to the Ministry of Education, a part of the Singapore government.",
    "STA" : "STA refers to Skills and Training Advisory, which is they key service provided by my company.",
    "WSG" : "WSG refers to Workforce Singapore, another organisation focused on assisting Singaporeans with career development.",
    "RIASEC" : "RIASEC refers to Realistic, Investigative, Artistic, Social, Enterprising and Conventional. It is a tool that helps clients discover what careers and fields\
      of study are likely to satisfy them.",
    "SSG": "SkillsFuture Singapore, the primary administrator of SkillsFuture Credits and the leading government body in the field of skills and career upgrading.",
    "ACLP" : "Advanced Certificate in Learning and Performance",
    "SCTP" : "SkillsFuture Career Transition Programme",
    "SFLP" : "SkillsFuture Level-Up Programme",
    "Singpass" : "Secure network for Singaporeans to log into government websites and access personal data.",
    "BDVL" : "Bus Driver Vocational License",
    "PDVL" : "Private Driver Vocational License",
    "PMP" : "Project Management Professional",
    "WPLN" : "Workplace Literacy and Numeracy",
    "TDVL" : "Taxi Driver Vocational License",
}

# for the first model, which will create a summary of the case notes and prepare them for classification.

system_prompt_1 = """You are an intelligent summarization assistant. Your task is to extract the essence of the case note, focusing on the client's intention based on
the provided categories."""

summarization_prompt = f"""Summarize the following case note in no more than 100 words.
If you are unsure of the acronyms, refer to this dictionary: {definitions}. You may proceed to summarize."""

# for the second model, which will process the summarised case notes and classify them.

system_prompt_2 = f""" """


classification_prompt = f"""Classify the following case note into one of the four categories:

Example 1:
Case Note: {eval_data['Note'].iloc[0]}
Label: {eval_data['Final'].iloc[0]}

Example 2:
Case Note: {eval_data['Note'].iloc[1]}
Label: {eval_data['Final'].iloc[1]}

Example 3:
Case Note: {eval_data['Note'].iloc[2]}
Label: {eval_data['Final'].iloc[2]}

Example 4:
Case Note: {eval_data['Note'].iloc[3]}
Label: {eval_data['Final'].iloc[3]}

Example 5:
Case Note: {eval_data['Note'].iloc[4]}
Label: {eval_data['Final'].iloc[4]}

State only the number label, not the name and no other text.

Now, classify this case note: """

narrow_down_prompt = """ """

select_one_prompt = " "

"""### Modified Prompts with 3 categories ###"""

system_prompt_1 = """ """

summarization_prompt = f""" """

system_prompt_2 = f"""
"""

classification_prompt = f""" """

"""### Create GPT Base Models ###"""

API_KEY = "<Insert API Key here>"

class Summarizer:
  def __init__(self, API_KEY, model_type, purpose = None):
    self.client = OpenAI(api_key = API_KEY)
    self.model = model_type
    self.purpose = purpose
    self.class_ = "Summarizer"
    self.conversation_history = []

  def get_model_type(self):
    return self.model

  def get_purpose(self):
    return self.purpose

  def set_new_model(self, new_model):
    self.model = new_model

  def set_new_purpose(self, new_purpose):
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

  def summarize(self, prompt, case_note, temp = 0.4, max_tokens = 175): # will summarise
    response = self.client.chat.completions.create(
        model = self.model,
        messages = [
            {"role": "system", "content": self.purpose},
            {"role": "user", "content": prompt + case_note}
          ],
        temperature = temp,
        max_tokens = max_tokens # roughly 100 words
    )
    answer = response.choices[0].message.content
    if answer is None or answer == '':
      return "ChatGPT did not return any response for this query."

    self.conversation_history.append((prompt, answer))
    return answer

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

  def narrow_down(self, narrow_down_prompt, case_note, temp = 0.2, max_tokens = 7):
    response = self.client.chat.completions.create(
        model = self.model,
        messages = [
            {"role": "system", "content": self.purpose},
            {"role": "user", "content": narrow_down_prompt + case_note}
          ],
        temperature = temp,
        max_tokens = max_tokens
    )
    answer = response.choices[0].message.content
    if answer is None or answer == '':
      return "ChatGPT did not return any response for this query."
    self.conversation_history.append((narrow_down_prompt, answer))
    return answer

  def classify(self, prompt, case_note, temp = 0.2, top_p = 0.8, max_tokens = 1): # will classify
    response = self.client.chat.completions.create(
        model = self.model,
        messages = [
            {"role": "system", "content": self.purpose},
            {"role": "user", "content": prompt + case_note}
          ],
        temperature = temp,
        top_p = top_p,
        max_tokens = max_tokens
    )
    answer = response.choices[0].message.content
    if answer is None or answer == '':
      return "ChatGPT did not return any response for this query."
    else:
      try:
        answer = int(answer)
      except:
        print(answer) # see where the issue is
        raise TypeError("ChatGPT did not return a numeric answer for this query.")

    self.conversation_history.append((prompt, answer))
    return answer

"""### Create Pipeline ###"""

class ModelPipeline:
  def __init__(self, *models):
    self.models = models
    self.model_names = [model.get_class_type() for model in self.models]
    self.class_ = "Pipeline"
    # ensures seq is followed in push_data function later on
    for model in self.models:
      if model.get_class_type() == "Summarizer":
        self.summarizer = model
      elif model.get_class_type() == "Classifier":
        self.classifier = model
    self.num_components = len(self.models)
    self.results = []

  def get_model_names(self):
    return self.model_names

  def get_class_type(self):
    return self.class_

  def get_models(self):
    return self.models

  def get_component_summary(self):
    info = {
        "model_names" : self.model_names,
        "component_summaries" : [model.get_gpt_summary() for model in self.models]
    }
    return info

  def get_num_components(self):
    return self.num_components

  def clear_history(self):
    self.results = []

  def push_data(self, summarization_prompt, classification_prompt, datapoint, select_one_prompt = None, narrow_down_prompt = None):
    summarized_data = self.summarizer.summarize(summarization_prompt, datapoint)
    if narrow_down_prompt is not None:
      narrowed_data = self.classifier.narrow_down(narrow_down_prompt, summarized_data)
      classified_data = self.classifier.classify(select_one_prompt, summarized_data)
    else:
      classified_data = self.classifier.classify(classification_prompt, summarized_data)
    self.results.append(classified_data)
    return classified_data

"""### Test the Pipeline ###"""

import random

def make_predictions(pipeline, data, select_one_prompt = None, narrow_down_prompt = None):
  pipeline.clear_history()
  for model in pipeline.get_models():
    model.clear_history()
  for i in range(len(data)):
    datapoint = data['Note'].iloc[i]
    pipeline.push_data(summarization_prompt, classification_prompt, datapoint, select_one_prompt=select_one_prompt, narrow_down_prompt=narrow_down_prompt)

  return pipeline.results

def accuracy(pipeline, data):
  counter = 0
  for i in range(len(pipeline.results)):
    if pipeline.results[i] == data['ForAI'].iloc[i]:
      counter += 1
  return counter * 100 /len(pipeline.results)

"""### Try out GPT-3.5 ###"""

summarizer = Summarizer(API_KEY, "gpt-3.5-turbo", purpose = system_prompt_1)
classifier = Classifier(API_KEY, "gpt-3.5-turbo", purpose = system_prompt_2)
pipeline = ModelPipeline(summarizer, classifier)

make_predictions(pipeline, train_data)
acc = accuracy(pipeline, train_data)
print(acc)

# Compute confusion matrix
conf_matrix = confusion_matrix(list(train_data['ForAI'].head(len(pipeline.results))), pipeline.results)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""### Try out GPT-4 ###"""

summarizer = Summarizer(API_KEY, "gpt-3.5-turbo", system_prompt_1)
classifier = Classifier(API_KEY, "gpt-4o", system_prompt_2)
pipeline = ModelPipeline(summarizer, classifier)

make_predictions(pipeline, train_data)
acc = accuracy(pipeline, train_data)
print(acc)

# Compute confusion matrix
conf_matrix = confusion_matrix(list(train_data['ForAI'].head(len(pipeline.results))), pipeline.results)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

len(pipeline.results)

for i in range(len(pipeline.results)):
  if pipeline.results[i] != train_data['ForAI'].iloc[i]:
    print(f"Index: {i}", f"Actual: {train_data['ForAI'].iloc[i]}", f"Predicted: {pipeline.results[i]}")

"""### What if we didn't summarise? ###"""

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(1)

model = Classifier(API_KEY, "gpt-4o", system_prompt_2)
results = []
count = 0
for index, row in train_data.iterrows():
  result = model.classify(classification_prompt, row['Note'])
  results.append(result)
  if result == int(row['ForAI']):
    count += 1

print(count * 100 / 99)

#print(model.get_conversation_history()[:5])

results = []
for prompt, answer in model.get_conversation_history():
  results.append(answer)

# Compute confusion matrix
conf_matrix = confusion_matrix(list(train_data['ForAI'].head(len(model.get_conversation_history()))), results)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

for i in range(len(results)):
  if results[i] != train_data['ForAI'].iloc[i]:
    print(f"Index: {i}, Note: {train_data['Note'].iloc[i][:10]}", f"Actual: {train_data['ForAI'].iloc[i]}", f"Predicted: {results[i]}")

