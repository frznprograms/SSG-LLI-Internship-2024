# -*- coding: utf-8 -*-
"""
##Imports##
"""

!python -m pip install setfit
!pip install transformers==4.39.0
!python -m pip install optuna

import pandas as pd
import numpy as np
import json

"""## Data Preparation ##"""

import re

train_data = pd.read_excel("GPT_FineTuning2.xlsx")
train_data = train_data[["Note", "MultiCategory"]]

# prepare validation/evaluation data
edata = pd.read_excel("20 case notes_for shane.xlsx")
edata = edata[["Note", "MultiCategory"]]

# prepare test data
test_data = pd.read_excel("TestData.xlsx")
test_data = test_data.head(50)
test_data = test_data[["Note", "MultiCategory"]]

def convert_to_list(entry):
    # Use regex to extract the numbers between square brackets
    numbers = re.findall(r'\d+', entry)
    # Convert the numbers to integers and return as a list
    return [int(num) for num in numbers]

# Apply the function to the 'MultiCategory' column
train_data['MultiCategory'] = train_data['MultiCategory'].apply(convert_to_list)
edata['MultiCategory'] = edata['MultiCategory'].apply(convert_to_list)
test_data['MultiCategory'] = test_data['MultiCategory'].apply(convert_to_list)

"""Convert the multi labels into binary labels:
For example: 2 becomes [0, 1, 0, 0] and [1, 3] becomes [1, 0, 1, 0]
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# convert the labels to binary arrays in the same form as one-hot encoding
datasets = [train_data, edata, test_data]
for dataset in datasets:
  mlb = MultiLabelBinarizer()
  binary_labels = mlb.fit_transform(dataset['MultiCategory'])
  binary_labels = binary_labels.tolist()
  dataset['binary_labels'] = binary_labels

test_data = test_data.drop(columns=['MultiCategory'])
edata = edata.drop(columns=['MultiCategory'])
train_data = train_data.drop(columns=['MultiCategory'])

"""Preprocess the Text

## Hyperparameter Tuning ##

##Test 1##
"""

import optuna
from optuna.pruners import MedianPruner
import gc
import torch
from setfit import Trainer, SetFitModel, sample_dataset, TrainingArguments

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [3, 4, 5, 6])
    num_epochs = trial.suggest_int("num_epochs", 3, 6)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    max_steps = trial.suggest_int("max_steps", 150, 500)

    # Define the model initialization function
    def model_init(params=None):
        params = {
            "head_params": {
                "max_iter": 100,
                "solver": "liblinear",
            }
        }
        gc.collect()
        torch.cuda.empty_cache()
        return SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", multi_target_strategy="multi-output", **params)

    args = TrainingArguments(
        batch_size = batch_size,
        num_epochs = num_epochs,
        body_learning_rate = learning_rate,
        max_steps = max_steps
    )

    # Initialize the model and trainer
    model = model_init()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args = args,
        column_mapping = {
            "Note" : "text",
            "binary_labels": "label"
        }
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    accuracy = metrics['accuracy']

    # Clean up to manage memory
    del trainer.model
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy

# Create a study object with MedianPruner and a random state
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=32), # Set a random state for reproducibility
    study_name = "setfit-multi-cat-test-1"
)

# Run the optimization
study.optimize(objective, n_trials=15, timeout=None, gc_after_trial=True)

# Get the best trial
best_trial = study.best_trial

print(f"Best trial accuracy: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")

# Example usage of the best hyperparameters
best_params = best_trial.params
batch_size = best_params['batch_size']
num_epochs = best_params['num_epochs']
learning_rate = best_params['learning_rate']
max_steps = best_params['max_steps']

args = TrainingArguments(
    batch_size = batch_size,
    num_epochs = num_epochs,
    body_learning_rate = learning_rate,
    max_steps = max_steps
)

# Initialize the model and trainer with best hyperparameters
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", multi_target_strategy="multi-output")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = args,
    column_mapping = {
        "Note" : "text",
        "binary_labels": "label"
    }
)

# Train the model with best hyperparameters
trainer.train()
# Evaluate the model with best hyperparameters
metrics = trainer.evaluate()
print(f"Final accuracy with best hyperparameters: {metrics['accuracy']}")

"""##Test 2##"""

import pandas as pd
import re
import spacy
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

def preprocess(text: str) -> str:
    stemmer = PorterStemmer()
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    final_list = []

    for token in doc:
        stemmed_token = stemmer.stem(token.text)
        if token.text.lower() in ['from', 'to']:
            continue
        elif token.text.lower() not in STOP_WORDS and not token.is_punct:
            final_list.append(stemmed_token)

    return ' '.join(final_list)

# Loading and preprocessing data
train_data = pd.read_excel("GPT_FineTuning2.xlsx")
train_data = train_data[["Note", "MultiCategory"]]
train_data['Preprocessed'] = train_data['Note'].apply(preprocess)

# prepare validation/evaluation data
edata = pd.read_excel("20 case notes_for shane.xlsx")
edata = edata[["Note", "MultiCategory"]]
edata['Preprocessed'] = edata['Note'].apply(preprocess)

# prepare test data
test_data = pd.read_excel("TestData.xlsx")
test_data = test_data.head(50)
test_data = test_data[["Note", "MultiCategory"]]
test_data['Preprocessed'] = test_data['Note'].apply(preprocess)

def convert_to_list(entry):
    numbers = re.findall(r'\d+', entry)
    return [int(num) for num in numbers]

# Apply the function to the 'MultiCategory' column
train_data['MultiCategory'] = train_data['MultiCategory'].apply(convert_to_list)
edata['MultiCategory'] = edata['MultiCategory'].apply(convert_to_list)
test_data['MultiCategory'] = test_data['MultiCategory'].apply(convert_to_list)

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# convert the labels to binary arrays in the same form as one-hot encoding
datasets = [train_data, edata, test_data]
for dataset in datasets:
  mlb = MultiLabelBinarizer()
  binary_labels = mlb.fit_transform(dataset['MultiCategory'])
  binary_labels = binary_labels.tolist()
  dataset['binary_labels'] = binary_labels

test_data = test_data.drop(columns=['MultiCategory'])
edata = edata.drop(columns=['MultiCategory'])
train_data = train_data.drop(columns=['MultiCategory'])

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(edata)
test_dataset = Dataset.from_pandas(test_data)

import optuna
from optuna.pruners import MedianPruner
import gc
import torch
from setfit import Trainer, SetFitModel, sample_dataset, TrainingArguments

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [3, 4, 5, 6])
    num_epochs = trial.suggest_int("num_epochs", 3, 6)
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    max_steps = trial.suggest_int("max_steps", 150, 500)

    # Define the model initialization function
    def model_init(params=None):
        params = {
            "head_params": {
                "max_iter": 100,
                "solver": "liblinear",
            }
        }
        gc.collect()
        torch.cuda.empty_cache()
        return SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", multi_target_strategy="multi-output", **params)

    args = TrainingArguments(
        batch_size = batch_size,
        num_epochs = num_epochs,
        body_learning_rate = learning_rate,
        max_steps = max_steps
    )

    # Initialize the model and trainer
    model = model_init()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args = args,
        column_mapping = {
            "Preprocessed" : "text",
            "binary_labels": "label"
        }
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    accuracy = metrics['accuracy']

    # Clean up to manage memory
    del trainer.model
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy

# Create a study object with MedianPruner and a random state
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=32), # Set a random state for reproducibility
    study_name = "setfit-multi-cat-test-1"
)

# Run the optimization
study.optimize(objective, n_trials=15, timeout=None, gc_after_trial=True)

# Get the best trial
best_trial = study.best_trial

print(f"Best trial accuracy: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")

fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Example usage of the best hyperparameters
best_params = best_trial.params
batch_size = best_params['batch_size']
num_epochs = best_params['num_epochs']
learning_rate = best_params['learning_rate']
max_steps = best_params['max_steps']

args = TrainingArguments(
    batch_size = batch_size,
    num_epochs = num_epochs,
    body_learning_rate = learning_rate,
    max_steps = max_steps
)

# Initialize the model and trainer with best hyperparameters
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", multi_target_strategy="multi-output")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = args,
    column_mapping = {
        "Preprocessed" : "text",
        "binary_labels": "label"
    }
)

# Train the model with best hyperparameters
trainer.train()
# Evaluate the model with best hyperparameters
metrics = trainer.evaluate()
print(f"Final accuracy with best hyperparameters: {metrics['accuracy']}")
