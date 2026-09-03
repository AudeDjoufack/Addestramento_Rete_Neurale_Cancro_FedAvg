import copy
import os

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split


class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_client(model, dataloader, epochs, lr, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def federated_averaging(global_model, client_weights, client_sizes):
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    for client_w, size in zip(client_weights, client_sizes):
        weight_factor = size / total_samples
        for key in global_dict.keys():
            global_dict[key] += client_w[key] * weight_factor

    global_model.load_state_dict(global_dict)
    return global_model

# Funzione di valutazione aggiunta
def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total

def run_federated_learning(global_model, client_loaders, client_sizes, test_loader, rounds, local_epochs, lr, device):
    for round_idx in range(rounds):
        print(f"--- Round Globale {round_idx + 1}/{rounds} ---")
        client_weights = []
        global_weights = global_model.state_dict()

        for client_idx, dataloader in enumerate(client_loaders):
            client_model = copy.deepcopy(global_model)
            client_model.load_state_dict(global_weights)
            client_model.to(device)

            weights = train_client(client_model, dataloader, local_epochs, lr, device)
            client_weights.append(weights)

        # Aggregazione FedAvg
        global_model = federated_averaging(global_model, client_weights, client_sizes)

        # FASE DI TEST: Valutiamo il modello globale aggregato sul test set centrale
        test_loss, test_acc = evaluate_model(global_model, test_loader, device)
        print(f"Risultati Test Globale -> Loss: {test_loss:.4f} | Accuratezza: {test_acc * 100:.2f}%\n")

    return global_model



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = NetCNN().to(device)

    csv_file = "data/labels.csv"
    img_dir = "data"
    classes = ('benign', 'malignant', 'normal')
    df = pd.read_csv(csv_file)
    print(df.head())
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]


    class BreastDataset(Dataset):
        def __init__(self, dataframe, img_dir, transform=None):
            self.data = dataframe
            self.img_dir = img_dir
            self.transform = transform
            self.classes = sorted(self.data['case category'].unique())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = self.data.iloc[idx]['filepath']
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            label_str = self.data.iloc[idx]['case category']
            label = self.class_to_idx[label_str]
            if self.transform:
                image = self.transform(image)
            return image, label

    batch_size = 16
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = BreastDataset(train_df, img_dir="data", transform=transform)

    test_dataset = BreastDataset(test_df, img_dir="data", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dim=int(len(train_dataset)/3)
    set1, set2, set3 = random_split(train_dataset, [dim, dim, dim])
    # Dati dei 3 client
    loader1 = DataLoader(set1, batch_size=16, shuffle=True)
    loader2 = DataLoader(set2, batch_size=16, shuffle=True)
    loader3 = DataLoader(set3, batch_size=16, shuffle=True)

    client_loaders = [loader1, loader2, loader3]
    client_sizes = [dim, dim, dim]

    # Avvio del Federated Learning con test integrato
    trained_global_model = run_federated_learning(
        global_model=global_model,
        client_loaders=client_loaders,
        client_sizes=client_sizes,
        test_loader=test_loader,
        rounds=20,
        local_epochs=1,
        lr=0.001,
        device=device
    )