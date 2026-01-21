import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score

# =========================
# 1. Load & Prepare Dataset
# =========================

df = pd.read_csv(r"C:\Users\DELL\fake_news\FakeNewsNet.csv")
df = df[['title', 'real']]
df.rename(columns={'real': 'label'}, inplace=True)

df.dropna(subset=['title'], inplace=True)
df['title'] = df['title'].astype(str)

print(df['label'].value_counts())

# =================
# 2. Text Cleaning
# =================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['title'] = df['title'].apply(clean_text)

# ======================
# 3. Train–Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    df['title'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# ======================
# 4. Tokenizer & Model
# ======================

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =================
# 5. Tokenization
# =================

train_enc = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

test_enc = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

print("Setup completed successfully 🚀")

# ==========================
# 6. PyTorch Dataset Class
# ==========================

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_ds = NewsDataset(train_enc, y_train)
test_ds = NewsDataset(test_enc, y_test)

# =================
# 7. Training
# =================

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()

# =================
# 8. Evaluation
# =================

pred = trainer.predict(test_ds)
y_pred = pred.predictions.argmax(axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))


model.save_pretrained("fake_news_model")
tokenizer.save_pretrained("fake_news_model")
