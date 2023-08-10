import json
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# 1. Loading Labeled Data
with open("/home/kbh/paper_tree/references_dataset.json", 'r', encoding='utf-8') as file:
    dataset = json.load(file)
    texts = [entry["references"] for entry in dataset]

# 2. Data Preprocessing
MAX_NUM_WORDS = 20000  # Maximum number of words in the vocabulary
tokenizer = get_tokenizer("basic_english")
counter = Counter()
for text in texts:
    counter.update(tokenizer(text))
vocab = Vocab(counter)
data_sequences = [torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long) for text in texts]
data_padded = torch.nn.utils.rnn.pad_sequence(data_sequences, batch_first=True, padding_value=0)
labels_padded = torch.nn.utils.rnn.pad_sequence([torch.ones(len(seq)) for seq in data_sequences], batch_first=True, padding_value=0).long()

# 3. Model Architecture
class TransformerForSequenceLabelingWithReLU(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_classes=1):
        super(TransformerForSequenceLabelingWithReLU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        output = self.fc(x)
        return output

D_MODEL = 128  # Dimension of model
NHEAD = 8  # Number of self-attention heads in Transformer
NUM_ENCODER_LAYERS = 3  # Number of transformer encoder layers
model_with_relu = TransformerForSequenceLabelingWithReLU(len(vocab), D_MODEL, NHEAD, NUM_ENCODER_LAYERS)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_with_relu.parameters())

train_data, val_data, train_labels, val_labels = train_test_split(data_padded, labels_padded, test_size=0.2)
BATCH_SIZE = 2
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataset = TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets.view(-1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), targets.view(-1).float())
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
    torch.save(model.state_dict(), "refer.pth")
    return train_losses, val_losses

train_losses, val_losses = train_model(model_with_relu, criterion, optimizer, train_loader, val_loader, num_epochs=10)

