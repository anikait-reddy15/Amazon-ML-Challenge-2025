# --- Import Libraries ---
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# --- Define Core Paths ---
# The dataset folder is assumed to be in the same directory as this script
DATASET_FOLDER = 'dataset/'
TRAIN_CSV = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_CSV = os.path.join(DATASET_FOLDER, 'test.csv') # Final test set

# The model will be saved and loaded from a local directory
MODEL_SAVE_PATH = 'results_smape_loss'
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# --- Data Loading and Preprocessing ---
print("--- Data Loading and Preprocessing ---")
df_train_full = pd.read_csv(TRAIN_CSV)
df_train_full['catalog_content'] = df_train_full['catalog_content'].astype(str).str.lower()
df_train_full['log_price'] = np.log1p(df_train_full['price'])

# --- Split training data for validation ---
df_train, df_val = train_test_split(df_train_full, test_size=0.1, random_state=42)

print(f"Loaded {len(df_train_full)} training records.")
print(f"Training set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")

# --- Tokenizer and Dataset Class ---
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PriceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --- Create dataset objects ---
train_dataset = PriceDataset(
    texts=df_train.catalog_content.tolist(),
    labels=df_train.log_price.tolist(),
    tokenizer=tokenizer
)

val_dataset = PriceDataset(
    texts=df_val.catalog_content.tolist(),
    labels=df_val.log_price.tolist(),
    tokenizer=tokenizer
)
print("Dataset objects created successfully.")

# --- Custom Loss and Trainer ---
def smape_loss(y_pred, y_true, epsilon=1e-8):
    original_pred = torch.expm1(y_pred)
    original_true = torch.expm1(y_true)
    numerator = torch.abs(original_pred - original_true)
    denominator = (torch.abs(original_true) + torch.abs(original_pred)) / 2 + epsilon
    return torch.mean(numerator / denominator) * 100

class SmapeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = smape_loss(logits.squeeze(-1), labels)
        return (loss, outputs) if return_outputs else loss

def smape_metric(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1e-8
    return np.mean(numerator / denominator) * 100

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    original_preds = np.expm1(logits.flatten())
    original_labels = np.expm1(labels.flatten())
    return {"smape": smape_metric(original_labels, original_preds)}

print("Custom SMAPE Loss and SmapeTrainer defined.")

# --- Model Training ---
print("--- Model Training ---")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,
    weight_decay=0.0001,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="smape",
    greater_is_better=False,
    save_total_limit=2,
    logging_dir=os.path.join(MODEL_SAVE_PATH, 'logs'),
    logging_steps=100,
    report_to="none"
)

trainer = SmapeTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Training Starts...")
trainer.train()
print("Training complete. Best model saved to:", trainer.state.best_model_checkpoint)

# -----------------------------------------------------------------------------
# --- Prediction Block ---

# --- Load the final trained model and tokenizer from the saved checkpoint ---
print("\n--- Loading best model for prediction ---")
try:
    final_model_path = trainer.state.best_model_checkpoint
    tokenizer_for_pred = AutoTokenizer.from_pretrained(final_model_path)
    model_for_pred = AutoModelForSequenceClassification.from_pretrained(final_model_path, num_labels=1)
    model_for_pred.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_for_pred.to(device)
    print(f"Best model loaded from {final_model_path} and moved to {device}.")
except Exception as e:
    print(f"Error loading final model for prediction: {e}")
    tokenizer_for_pred = None
    model_for_pred = None
    device = 'cpu'

# --- Define the core predictor function ---
def predictor(sample_id, catalog_content, image_link):
    '''
    Predicts the price for a single product sample using the trained model.
    
    Parameters:
    - sample_id: Unique identifier for the sample (not used)
    - catalog_content: Text containing product title and description
    - image_link: URL to product image (not used)
    
    Returns:
    - price: Predicted price as a float
    '''
    if model_for_pred is None or tokenizer_for_pred is None:
        return 0.0

    text_input = str(catalog_content).lower()
    encoding = tokenizer_for_pred.encode_plus(
        text_input,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model_for_pred(input_ids=input_ids, attention_mask=attention_mask)
        log_price = outputs.logits.squeeze().item()
    
    predicted_price = np.expm1(log_price)
    
    return float(max(0.0, predicted_price))

# --- Main execution block for prediction ---
if __name__ == "__main__":
    df_final_test = pd.read_csv(TEST_CSV)
    df_final_test['catalog_content'] = df_final_test['catalog_content'].astype(str).str.lower()
    
    print(f"\nLoaded {len(df_final_test)} records from the final test set.")
    print("Making predictions...")
    
    # Apply the predictor function to each row
    df_final_test['price'] = df_final_test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = df_final_test[['sample_id', 'price']]
    
    # Save the predictions to the final output file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"\nPredictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print("\nSample predictions:")
    print(output_df.head())