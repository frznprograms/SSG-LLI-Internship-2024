#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd 
import numpy as np
import string
import os
import json


# In[18]:


# --- Data Preparation ---


# In[19]:


df = pd.read_excel("GPT_FineTuning.xlsx")
df = df.drop(df.index[3])
df['Note'] = df['Note'].astype(str)

def cleanstring(string:str): 
    if string[:4] == "Case": 
        string = string[9:].strip()
        return string
    else: 
        return string
    
df['CaseNote'] = df['Note'].apply(cleanstring)
df.drop(['Note'], axis = 1, inplace = True)

for index, row in df.iterrows():
    if row["CaseNote"] == 'nan':
        row["CaseNote"] = np.nan
        
df.dropna(subset = ["Category", "CaseNote"], inplace = True)

print(df["Category"].value_counts())


# In[20]:


print(os.getcwd())


# In[21]:


# --- Convert our Data to JSONL format for Fine-Tuning ---


# In[22]:


# Function to convert a dictionary to the required JSON format
# Converts a dictionary to the required JSON format. Takes in dictionary and converts it to JSON format.
# helper function for convert_to_json()

def dict_to_json(data_dict):
    return json.dumps(data_dict)

# Function to convert the dataset to the required JSONL format
# Converts a dataset to the required JSONL format and saves it to a file. Takes in the dataset (pandas df)
# and filename, the name you would like to give to the JSONL file. 

def convert_to_jsonl(dataset, filename):
    with open(filename, 'w') as f:
        for index, row in dataset.iterrows():
            json_line = {"messages": [
                {"role": "system", "content": "<>"},
                {"role": "user", "content": "State the type of client this is: " + row["CaseNote"]},
                {"role": "assistant", "content": str(row["Category"])}
            ]}
            jsonl_line = dict_to_json(json_line)
            f.write(jsonl_line + '\n')

# Convert the dataset to JSONL format
#convert_to_jsonl(df.head(40), 'converted_data_train.jsonl')
#convert_to_jsonl(df.tail(20), 'converted_data_validation.jsonl')


# In[23]:


# Function to read and display the contents of a JSONL file
def display_jsonl(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            print(json_data)

#display_jsonl("/Users/shaneryan_1/Downloads/converted_data_train.jsonl")


# In[24]:


def read_jsonl_as_list(file_path): 
    jsonl_data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            jsonl_data.append(json_data)
    return jsonl_data

json_list = read_jsonl_as_list("/Users/shaneryan_1/Downloads/converted_data_train.jsonl")
#print(type(json_list))
#print(json_list)


# In[25]:


# --- Fine-Tune the Model ---


# In[26]:


'''
# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Fine-tuning with JSONL data
# Set the fine-tuning parameters
fine_tuning_parameters = {
    'engine': 'gpt-3.5-turbo-0125',
    'data': {
        'jsonl': 'YOUR_TRAINING_DATA.jsonl',
        'valid_jsonl': 'YOUR_VALIDATION_DATA.jsonl'
    },
    'temperature': 0.5,
    'max_tokens': 100,
    'epochs': 5,
    'validation_every_n': 20,
    'print_every_n': 20
}

# Perform fine-tuning
fine_tuned_model = openai.FineTune.create(**fine_tuning_parameters)
'''


# In[37]:


import requests
import json

# Define your OpenAI API key
api_key = '<Insert API Key here>'

# Base URL for the OpenAI API
base_url = 'https://api.openai.com/v1'

# Function to list fine-tuning jobs
def list_fine_tuning_jobs():
    url = f"{base_url}/fine_tuning/jobs"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON data from the response
        return response.json()
    else:
        # Print the error status code and return None
        print(f"Error: {response.status_code}")
        return None

# Function to get checkpoints for a specific fine-tuning job
def get_checkpoints(fine_tuning_job_id, after=None, limit=20):
    url = f"{base_url}/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    params = {
        'after': after,
        'limit': limit
    }
    response = requests.get(url, headers=headers, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON data from the response
        return response.json()
    else:
        # Print the error status code and return None
        print(f"Error: {response.status_code}")
        return None

# List your fine-tuning jobs
fine_tuning_jobs = list_fine_tuning_jobs()
#print(json.dumps(fine_tuning_jobs, indent=2))

# get checkpoints for a specific job
fine_tuning_job_id = 'ftjob-tMIbHSH7VWn1dCHwlbZEkJhb'  # Replace with your actual fine-tuning job ID
checkpoints = get_checkpoints(fine_tuning_job_id)
print(json.dumps(checkpoints, indent=2))






