#Import Libraries 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms 
from PIL import Image 
import sys, pathlib
import torch.nn as nn
from Data_Loader import UnitCellDS
from Data_Loader import load_bandgap_data
import torch.optim as optim

#load Dataset 
dataset = load_bandgap_data('bandgap_data.mat', resize=64)

#Perform Snaity Check
# print(f"Total samples: {len(dataset)}")
# x_sample, y_sample = dataset[78]
# print(f"Image Shape: {x_sample.shape}")
# print(f"Lable shape: {y_sample.shape}")
# print(f"Label values: {y_sample}")

class FullCNN(nn.Module):
    def __init__(self):
        super(FullCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 256 * 4 * 4) #Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Trian Test Val Split 
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

#Set up training loop:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

model = FullCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#Track Metrics
train_losses = []
val_losses = []
train_accs = []
val_accs = []

all_labels = []
for i in range(len(dataset)):
    _, label = dataset[i]
    all_labels.append(label.numpy())

all_labels = np.array(all_labels)  # Shape: [N, 3]

print("\nLabel Distribution:")
for i in range(3):
    freq_labels = all_labels[:, i]
    n_gaps = freq_labels.sum()
    n_no_gaps = len(freq_labels) - n_gaps
    print(f"Freq Range {i}: No Gap={n_no_gaps} ({100*n_no_gaps/len(freq_labels):.1f}%), Has Gap={n_gaps} ({100*n_gaps/len(freq_labels):.1f}%)")

num_epochs = 20
print("Starting Training ...\n")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        #move to device
        images = images.to(device)
        labels = labels.to(device).float()

        #perform forward pass
        optimizer.zero_grad() #clear old gradients from prev step
        outputs = model(images) # runs images through the network
        loss = criterion(outputs, labels) #calculates how wrong prediction is woth BCElogitsloss


        #backward pass 
        loss.backward() #calcs gradient in backwards step
        optimizer.step() #updates weights

        #Update Metrics
        train_loss += loss.item()
        #Multi-label accuracy calculation
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.numel()  # Total number of predictions (batch_size × 3)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)  

    #validate model:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # no gradient computation during validaiton 
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.numel()

    #Calc avg val metrics
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print()

print("Training Complete")
torch.save(model.state_dict(), 'bandgap_cnn_simple.pth')
print("model saved as ")







