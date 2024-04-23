# Import necessary libraries
from datasets import load_dataset
import datasets
import torch
from transformers import AutoTokenizer
import pandas as pd
import ast
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DefaultDataCollator
from transformers import TrainingArguments
from transformers import Trainer

# Huggingface login
from huggingface_hub import notebook_login
notebook_login()


# Setting up Cuda or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Import tokenizer
model_checkpoint = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.is_fast


# Fetching data
df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')


# Tokenize training and validation
def tokenize_and_align_labels(tokenizer, df):
    tokenized_inputs = tokenizer(
        df['question'].tolist(),
        df['context'].tolist(),
        truncation=True,
        max_length=384,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True
    )

    # You may need to adjust the answer positions due to tokenization offsets
    start_positions = []
    end_positions = []

    for i in range(len(df)):
        # Get the character start position and end position
        start_char = df.iloc[i]['answer']['answer_start']
        end_char = df.iloc[i]['answer']['answer_end']

        # Find the token index corresponding to the start character
        sequence_ids = tokenized_inputs.sequence_ids(i)
        offset_mapping = tokenized_inputs['offset_mapping'][i]

        # Loop through each offset and find the token index that matches the start and end character
        input_ids = tokenized_inputs['input_ids'][i]

        # Initialize token positions to None
        start_token = None
        end_token = None

        for idx, (offset_start, offset_end) in enumerate(offset_mapping):
            if sequence_ids[idx] == 1:  # Only consider the context part for the answer
                if (start_char >= offset_start) and (start_char < offset_end):
                    start_token = idx
                if (end_char > offset_start) and (end_char <= offset_end):
                    end_token = idx

        # Handle cases where start or end token indices are None due to truncation or no valid answer
        if start_token is None or end_token is None:
            start_token = tokenizer.model_max_length  # Use max_length as a default index when answer is not found
            end_token = tokenizer.model_max_length

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized_inputs.update({'start_positions': torch.tensor(start_positions), 'end_positions': torch.tensor(end_positions)})
    return tokenized_inputs


# Apply the tokenization to the DataFrame
tokenized_train = tokenize_and_align_labels(tokenizer, df_train)
tokenized_val = tokenize_and_align_labels(tokenizer, df_val)

# Removing unnecessary columns
del tokenized_train["offset_mapping"]


# Preparing dataset for finetuning
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Ensure start and end positions are always tensors with a batch dimension
        item['start_positions'] = torch.tensor([item['start_positions']])
        item['end_positions'] = torch.tensor([item['end_positions']])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


# Build datasets for both our training and validation sets
train_dataset = SquadDataset(tokenized_train)
val_dataset = SquadDataset(tokenized_val)


# Fetching model
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# Initialize data loader for training data, validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Training
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="finetuned_roberta_qa_coqa",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
