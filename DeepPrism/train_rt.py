import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from utils import get_default_device

MAX_LEN = 128
NUM_CLASSES = 2
BATCH_SIZE = 100
EPOCHS = 10
LR = 1e-3


class RTDataset(Dataset):
    def __init__(self, split, tokenizer):
        data = load_dataset("rotten_tomatoes", split=split)
        self.texts = data["text"]
        self.labels = data["label"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class RTClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.fc1 = nn.Linear(100 * MAX_LEN, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (batch, seq, embed)
        x = x.view(x.size(0), -1)      # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def train_model(epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds = RTDataset("train", tokenizer)
    test_ds = RTDataset("test", tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = RTClassifier(tokenizer.vocab_size).to(get_default_device())
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        model.train()
        total_loss = 0
        total_samples = 0
        for input_ids, attn_mask, labels in tqdm(train_loader):
            input_ids = input_ids.to(get_default_device())
            labels = labels.to(get_default_device())

            preds = model(input_ids)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        print(f"\tAvg. training loss: {total_loss / total_samples:.4f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            correct = 0
            val_loss = 0
            total = 0
            with torch.no_grad():
                for input_ids, attn_mask, labels in tqdm(test_loader):
                    input_ids = input_ids.to(get_default_device())
                    labels = labels.to(get_default_device())
                    preds = model(input_ids)
                    val_loss += F.cross_entropy(preds, labels, reduction='sum').item()
                    correct += (preds.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            print(f"----validation loss: {val_loss / total:.4f}")
            print(f"----accuracy: {correct / total * 100:.2f}%")

        torch.save(model.state_dict(), "saved/rt.pt")

if __name__ == "__main__":
    train_model()