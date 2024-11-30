import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CyclicLR
import torch.nn as nn
import torch.optim as optim


class BasketballDataset(Dataset):
    # 1. Data Preparation
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Filter rows with valid InputVec
        self.data = self.data[self.data['InputVec'] != '0']
        self.features = np.array(self.data['InputVec'].apply(
            eval).tolist())  # Convert InputVec to numpy array
        self.labels = self.data['H'].values.astype(
            np.float32)  # Home team win (1 or 0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# Initialize dataset
dataset = BasketballDataset(r"D:\_FEL\QQH2024\testing\kuba\all_games_data.csv")

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"Train size: {train_size}, Validation size: {val_size}")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.4):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU with LeakyReLU
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout)
        )
        # Shortcut for residual connection
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x) + self.shortcut(x)

# Adding an Attention Mechanism


# class Attention(nn.Module):
#     def __init__(self, input_size, attention_size):
#         super(Attention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(input_size, attention_size),
#             nn.Tanh(),
#             nn.Linear(attention_size, 1)
#         )

#     def forward(self, x):
#         weights = torch.softmax(self.attention(x), dim=1)
#         return torch.sum(weights * x, dim=1)


# class BasketballANNWithAttention(nn.Module):
#     def __init__(self, input_size):
#         super(BasketballANNWithAttention, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size, 128, batch_first=True, bidirectional=True)
#         self.attention = Attention(128 * 2, 128)
#         self.block1 = ResidualBlock(128, 256, dropout=0.5)
#         self.block2 = ResidualBlock(256, 128, dropout=0.4)
#         self.fc = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.Mish(),  # Mish activation function
#             nn.BatchNorm1d(64),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x, _ = self.lstm(x.unsqueeze(1))  # Adding LSTM layer
#         x = x.squeeze(1)
#         x = self.attention(x)  # Adding Attention layer
#         x = self.block1(x)
#         x = self.block2(x)
#         return self.fc(x)

# class BasketballANN(nn.Module):
#     def __init__(self, input_size):
#         super(BasketballANN, self).__init__()
#         self.lstm = nn.LSTM(input_size, 128, batch_first=True)
#         self.block1 = ResidualBlock(128, 256, dropout=0.5)
#         self.block2 = ResidualBlock(256, 128, dropout=0.4)
#         self.block3 = ResidualBlock(128, 64, dropout=0.3)
#         self.fc = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.Mish(),  # Mish activation function
#             nn.BatchNorm1d(32),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x, _ = self.lstm(x.unsqueeze(1))  # Adding LSTM layer
#         x = x.squeeze(1)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return self.fc(x)


class BasketballANN(nn.Module):
    def __init__(self, input_size):
        super(BasketballANN, self).__init__()
        self.block1 = ResidualBlock(input_size, 512, dropout=0.4)
        self.block2 = ResidualBlock(512, 256, dropout=0.4)
        self.block3 = ResidualBlock(256, 128, dropout=0.3)
        self.block4 = ResidualBlock(128, 64, dropout=0.3)  # New block
        # self.fc = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU with LeakyReLU
        #     nn.BatchNorm1d(64),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),  # Swish activation
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.fc(x)


# Initialize model
input_size = dataset[0][0].shape[0]  # Number of features in InputVec
assert input_size == 393  # Ensure input size is correct

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = BasketballANN(input_size).to(device)
# model = BasketballANNWithAttention(input_size).to(device)
EPOCHS = 100
# Initialize criterion and optimizer
# Compute class imbalance ratio
num_positive = sum(dataset.data['H'] == 1)
num_negative = sum(dataset.data['H'] == 0)
pos_weight = torch.tensor([num_negative / num_positive],
                          dtype=torch.float32).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

criterion = nn.BCELoss()  # Loss function (automatically works with GPU tensors)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=0.0005, steps_per_epoch=len(train_loader), epochs=EPOCHS)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,     # The lower bound of the learning rate range
    max_lr=1e-3,      # The upper bound of the learning rate range
    step_size_up=1000,  # Number of iterations to increase the learning rate
    mode='triangular',  # Options: 'triangular', 'triangular2', 'exp_range'
    cycle_momentum=False  # Set False if using AdamW, as it doesn't use momentum
)

# Training Loop with GPU Support
best_val_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(EPOCHS):  # Max epochs
    # Training
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(
            device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(
                device)  # Move data to GPU
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics and Logging
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = correct / total
    # scheduler.step()
    # scheduler.step(val_loss)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    predictions = (all_outputs >= 0.5).astype(int)

    roc_auc = roc_auc_score(all_labels, all_outputs)
    f1 = f1_score(all_labels, predictions)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {
          val_loss:.4f}, Val Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "best_model_epoch.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
