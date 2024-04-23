#importing necessary libraries
import datasets
import pandas as pd
from datasets import load_dataset
import csv
import torch

#seeting up Cuda or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Downloading the coqa dataset
data_train = load_dataset("stanfordnlp/coqa", split="train")
data_val = load_dataset("stanfordnlp/coqa", split="validation")

# Convert the dataset to a dictionary
data_dict = data_train.to_dict()
# Create a DataFrame from the dictionary
df_train = pd.DataFrame.from_dict(data_dict)
# Convert the dataset to a dictionary
data_dict_val = data_val.to_dict()
# Create a DataFrame from the dictionary
df_val = pd.DataFrame.from_dict(data_dict_val)
# Preprocessing data
df_train = df_train[['story', 'questions', 'answers']]
df_val = df_val[['story', 'questions', 'answers']]
# Adjusting column header to match with squad dataset
df_train.columns = ['context', 'questions', 'answers']
df_val.columns = ['context', 'questions', 'answers']

val_dataset2 = datasets.Dataset.from_pandas(df_val)
train_dataset2 = datasets.Dataset.from_pandas(df_train)


# Method for preparing the dataset similar to squad
def preprocess_dataframe(df):
    preprocessed_data = []

    for index, row in df.iterrows():
        context = row['context']
        questions = row['questions']
        answer_texts = row['answers']['input_text']
        answer_starts = row['answers']['answer_start']
        answer_ends = row['answers']['answer_end']

        for i, question in enumerate(questions):
            new_entry = {
                'context': context,
                'question': question,
                'answer': {
                    'text': answer_texts[i],
                    'answer_start': answer_starts[i],
                    'answer_end': answer_ends[i]
                }
            }
            preprocessed_data.append(new_entry)

    # Convert the list of dictionaries back into a DataFrame
    return pd.DataFrame(preprocessed_data)


# Prepare training data
with open('train.csv', 'w', newline='') as file:
   writer = csv.writer(file)
   writer.writerow(['context', 'questions', 'answers'])

preprocessed_df_train = preprocess_dataframe(df_train)
print(preprocessed_df_train)
preprocessed_df_train.to_csv('train.csv', index=False)

# Prepare validation data
with open('val.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['context', 'questions', 'answers'])

preprocessed_df_val = preprocess_dataframe(df_val)
print(preprocessed_df_val)
preprocessed_df_val.to_csv('val.csv', index=False)