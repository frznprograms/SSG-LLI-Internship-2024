# -*- coding: utf-8 -*-
"""
Imports

"""

!python -m pip install setfit
!pip install transformers==4.39.0

import pandas as pd
import numpy as np
import json

"""##Data Preparation##"""

# /Users/shaneryan_1/Downloads/

train_data = pd.read_excel("GPT_FineTuning2.xlsx")
train_data = train_data.head(100)
train_data = train_data[["Note", "Final"]]

# prepare validation/evaluation data
edata = pd.read_excel("20 case notes_for shane.xlsx")
edata = edata[["Note", "Final"]]
edata = edata.head(20)

# prepare test data
test_data = pd.read_excel("TestData.xlsx")
test_data = test_data.head(50)
test_data = test_data[["Note", "Final"]]

# Convert Labels to Type Integer
train_data['Final'] = train_data['Final'].astype(int)
edata['Final'] = edata['Final'].astype(int)
test_data['Final'] = test_data['Final'].astype(int)

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(edata)
test_dataset = Dataset.from_pandas(test_data)

"""Hyperparameter Tuning with Optuna"""

!python -m pip install optuna

"""##Test 1##"""

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
        return SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", **params)

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
        column_mapping = {
            "Note" : "text",
            "Final": "label"
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
    sampler=optuna.samplers.TPESampler(seed=42), 
    study_name = "setfit-test-1"
)

# Run the optimization
study.optimize(objective, n_trials=7, timeout=None, gc_after_trial=True)

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

"""Best hyperparameters: {'batch_size': 4, 'num_epochs': 4, 'learning_rate': 1.203017887115466e-05, 'max_steps': 341}

Train a Model with Optimised Hyperparameters
"""

batch_size = 4
num_epochs = 4
learning_rate = 1.203017887115466e-05
max_steps = 341

args = TrainingArguments(
    batch_size = batch_size,
    num_epochs = num_epochs,
    body_learning_rate = learning_rate,
    max_steps = max_steps
)

# Initialize the model and trainer with best hyperparameters
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = args,
    column_mapping = {
        "Note" : "text",
        "Final": "label"
    }
)

# Train the model with best hyperparameters
trainer.train()
# Evaluate the model with best hyperparameters
metrics = trainer.evaluate()
print(f"Final accuracy with best hyperparameters: {metrics['accuracy']}")

"""Best hyperparameters: {'batch_size': 4, 'num_epochs': 4, 'learning_rate': 1.203017887115466e-05, 'max_steps': 341}

Final accuracy with best hyperparameters: 0.75
"""

batch_size = 4
num_epochs = 4
learning_rate = 1.203017887115466e-05
max_steps = 341

args = TrainingArguments(
    batch_size = batch_size,
    num_epochs = num_epochs,
    body_learning_rate = learning_rate,
    max_steps = max_steps
)

# Initialize the model and trainer with best hyperparameters
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = args,
    column_mapping = {
        "Note" : "text",
        "Final": "label"
    }
)

# Train the model with best hyperparameters
trainer.train()
# Evaluate the model with best hyperparameters
metrics = trainer.evaluate(test_dataset)
print(f"Final accuracy with best hyperparameters: {metrics['accuracy']}")

model_save_path = "/Users/shaneryan_1/Downloads/SetFitModel"
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

"""## Reproduce Accuracy ##"""

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
        return SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", **params)

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
            "Final": "label"
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
  
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=44), 
    study_name = "setfit-test-1"
)

# Run the optimization
study.optimize(objective, n_trials=7, timeout=None, gc_after_trial=True)

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

batch_size = 6
num_epochs = 4
learning_rate = 5.135776127610297e-06
max_steps = 189

args = TrainingArguments(
    batch_size = batch_size,
    num_epochs = num_epochs,
    body_learning_rate = learning_rate,
    max_steps = max_steps
)

# Initialize the model and trainer with best hyperparameters
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = args,
    column_mapping = {
        "Note" : "text",
        "Final": "label"
    }
)

# Train the model with best hyperparameters
trainer.train()
# Evaluate the model with best hyperparameters
metrics = trainer.evaluate(test_dataset)
print(f"Final accuracy with best hyperparameters: {metrics['accuracy']}")
