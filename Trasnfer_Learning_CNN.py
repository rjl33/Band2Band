import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from Data_Loader import load_bandgap_data
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, classification_report
import numpy as np


# Load data
dataset = load_bandgap_data('bandgap_data.mat', resize=64)
print(f"Total samples: {len(dataset)}")

# Split data
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=torch.manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create transfer learning model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

with torch.no_grad():
    w = model.conv1.weight            # [64,3,7,7], pretrained
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.copy_(w.mean(dim=1, keepdim=True))  # [64,1,7,7]
model.fc = nn.Linear(model.fc.in_features, 5)
#keep more spatial detail early
model.maxpool = nn.Identity()

# warm-up: freeze backbone (train conv1 + fc first)
for name, p in model.named_parameters():
    if not (name.startswith("fc.") or name.startswith("conv1.")):
        p.requires_grad = False
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

all_y = []
for _, y in dataset:
    all_y.append(y.numpy())            # y should be shape [3] float
prev = np.clip(np.mean(np.vstack(all_y), axis=0), 1e-6, 1-1e-6)  # per-class prevalence
pos_weight = torch.tensor((1 - prev) / prev, device=device, dtype=torch.float32)

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # or drop pos_weight if you prefer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# (nice-to-have) LR scheduler on val loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

# Training
num_epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\nStarting Training...\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    if epoch == 5: 
        print("Unfreezing backbone and fine-tuning...")
        for p in model.parameters():
             p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        # (optional) reset scheduler to match new optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.numel()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.numel()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print()

print("Training complete!")

# Save model
torch.save(model.state_dict(), 'transfer_learning_model.pth')

# =========================
# Test / Evaluation Section
# =========================
model = models.resnet18(weights=None)  # Don't load pretrained weights

# Apply the same modifications
with torch.no_grad():
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Don't need to copy weights here since we're loading saved ones
    
model.fc = nn.Linear(model.fc.in_features, 5)
model.maxpool = nn.Identity()

# Load the saved weights
model = model.to(device)
model.load_state_dict(torch.load('transfer_learning_model.pth'))
model.eval()

test_loss = 0.0
elem_correct = 0
elem_total = 0

# For exact-match and per-class metrics
all_tp = torch.zeros(5, dtype=torch.long)
all_fp = torch.zeros(5, dtype=torch.long)
all_fn = torch.zeros(5, dtype=torch.long)
exact_match_count = 0
num_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()   # shape [B, 3], values {0,1}

        logits = model(images)               # [B, 3]
        loss = criterion(logits, labels)     # BCEWithLogitsLoss
        test_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()        # threshold

        # Element-wise accuracy
        elem_correct += (preds == labels).sum().item()
        elem_total   += labels.numel()

        # Exact-match accuracy (all labels correct for a sample)
        exact_match_count += (preds.eq(labels).all(dim=1)).sum().item()
        num_samples += labels.size(0)

        # Per-class TP/FP/FN
        # Convert to ints for counting
        p = preds.long()
        y = labels.long()
        tp = (p.eq(1) & y.eq(1)).sum(dim=0)
        fp = (p.eq(1) & y.eq(0)).sum(dim=0)
        fn = (p.eq(0) & y.eq(1)).sum(dim=0)

        all_tp += tp.cpu()
        all_fp += fp.cpu()
        all_fn += fn.cpu()

avg_test_loss = test_loss / len(test_loader)
elem_acc = 100.0 * elem_correct / elem_total
exact_match_acc = 100.0 * exact_match_count / num_samples

# Per-class precision/recall/F1 (safe division)
eps = 1e-8
precisions = (all_tp.float()) / (all_tp + all_fp + eps).float()
recalls    = (all_tp.float()) / (all_tp + all_fn + eps).float()
f1s        = 2 * precisions * recalls / (precisions + recalls + eps)

print("\n=== Test Results ===")
print(f"Test Loss:           {avg_test_loss:.4f}")
print(f"Element-wise Acc:    {elem_acc:.2f}%")
print(f"Exact-match Acc:     {exact_match_acc:.2f}%")

for i in range(5):
    print(f"Class {i}: "
          f"Precision={precisions[i].item():.3f}, "
          f"Recall={recalls[i].item():.3f}, "
          f"F1={f1s[i].item():.3f}, "
          f"TP={all_tp[i].item()}, FP={all_fp[i].item()}, FN={all_fn[i].item()}")


# Loss vs epoch
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# Accuracy vs epoch
plt.figure()
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.tight_layout()
plt.show()


