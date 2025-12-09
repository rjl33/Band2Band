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
import json


# Load data
dataset = load_bandgap_data('bandgap_data.mat', resize=128)
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
model.fc = nn.Sequential(
    nn.Dropout(0.35),  
    nn.Linear(512, 5)   
)
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
num_epochs = 30
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

    scheduler.step(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print()

print("Training complete!")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'resolution': 128
}, 'Final_Model.pth')

print("✓ Saved: Final_Model.pth")

#=========================
#Optimize the per class thresholds
#==========================
def find_optimal_thresholds(model, val_loader, device):
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    #Grid search for each class
    best_thresholds = []

    for class_idx in range(5):
        best_f1 = 0
        best_thresh = 0.5
        best_metrics = {}

        #threshold ranges: 0.2 to .8
        for thresh in np.arange(0.2, 0.85, 0.05):
            preds = (all_probs[:, class_idx] > thresh).float()

            tp = ((preds == 1) & (all_labels[:, class_idx] == 1)).sum().item()
            fp = ((preds == 1) & (all_labels[:, class_idx] == 0)).sum().item()
            fn = ((preds == 0) & (all_labels[:, class_idx] == 1)).sum().item()
        
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
        
        best_thresholds.append(best_thresh)
        print(f"Class {class_idx}: threshold={best_thresh:.2f}, "
              f"Precision={best_metrics['precision']:.3f}, "
              f"Recall={best_metrics['recall']:.3f}, "
              f"F1={best_metrics['f1']:.3f}")
        
    return torch.tensor(best_thresholds)

optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
print(f"\nOptimal thresholds: {optimal_thresholds.numpy()}")
print("="*70 + "\n")

# =========================
# Test / Evaluation Section - With both Optimal and Standard (0.5) Thresholds
# =========================
model = models.resnet18(weights=None)  # Don't load pretrained weights

# Apply the same modifications
with torch.no_grad():
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Don't need to copy weights here since we're loading saved ones
    
model.fc = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(512, 5)
)
model.maxpool = nn.Identity()

# Load the saved weights
model = model.to(device)
checkpoint = torch.load('Final_Model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def evaluate_with_thresholds(model, test_loader, criterion, device, thresholds):

    test_loss = 0.0
    elem_correct = 0.0
    elem_total = 0

    all_tp = torch.zeros(5, dtype=torch.long)
    all_fp = torch.zeros(5, dtype=torch.long)
    all_fn = torch.zeros(5, dtype=torch.long)
    exact_match_count = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images)
            loss = criterion(logits, labels)
            test_loss += loss.item()

            probs = torch.sigmoid(logits)

            #Apply per class thresholds
            preds = torch.zeros_like(probs)
            for i in range(5):
                preds[:, i] = (probs[:, i] > thresholds[i]).float()

            #element wise accruacy 
            elem_correct += (preds == labels).sum().item()
            elem_total += labels.numel()

            #Exact Match accuracy 
            exact_match_count += (preds.eq(labels).all(dim=1)).sum().item()
            num_samples += labels.size(0)

            #Per-class TP/FP/FN
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

    # Per-class metrics
    eps = 1e-8
    precisions = (all_tp.float()) / (all_tp + all_fp + eps).float()
    recalls = (all_tp.float()) / (all_tp + all_fn + eps).float()
    f1s = 2 * precisions * recalls / (precisions + recalls + eps)
    
    return {
        'test_loss': avg_test_loss,
        'elem_acc': elem_acc,
        'exact_match_acc': exact_match_acc,
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn
    }

#Test with STANDARD Thresholds:
print("="*70)
print("TEST RESULTS WITH STANDARD THRESHOLDS (0.5)")
print("="*70)
standard_thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], device=device)
results_standard = evaluate_with_thresholds(model, test_loader, criterion, device, standard_thresholds)

print(f"\nTest Loss:           {results_standard['test_loss']:.4f}")
print(f"Element-wise Acc:    {results_standard['elem_acc']:.2f}%")
print(f"Exact-match Acc:     {results_standard['exact_match_acc']:.2f}%")

for i in range(5):
    print(f"Class {i}: "
          f"Precision={results_standard['precisions'][i].item():.3f}, "
          f"Recall={results_standard['recalls'][i].item():.3f}, "
          f"F1={results_standard['f1s'][i].item():.3f}, "
          f"TP={results_standard['tp'][i].item()}, "
          f"FP={results_standard['fp'][i].item()}, "
          f"FN={results_standard['fn'][i].item()}")


#Test with Optimzed Thresholds:
print("\n" + "="*70)
print("TEST RESULTS WITH OPTIMIZED THRESHOLDS")
print("="*70)
print(f"Thresholds: {optimal_thresholds.numpy()}")
print()

results_optimized = evaluate_with_thresholds(model, test_loader, criterion, device, 
                                             optimal_thresholds.to(device))

print(f"Test Loss:           {results_optimized['test_loss']:.4f}")
print(f"Element-wise Acc:    {results_optimized['elem_acc']:.2f}%")
print(f"Exact-match Acc:     {results_optimized['exact_match_acc']:.2f}%")

for i in range(5):
    print(f"Class {i}: "
          f"Precision={results_optimized['precisions'][i].item():.3f}, "
          f"Recall={results_optimized['recalls'][i].item():.3f}, "
          f"F1={results_optimized['f1s'][i].item():.3f}, "
          f"TP={results_optimized['tp'][i].item()}, "
          f"FP={results_optimized['fp'][i].item()}, "
          f"FN={results_optimized['fn'][i].item()}")

print("\n" + "="*70)
print("IMPROVEMENT FROM THRESHOLD OPTIMIZATION")
print("="*70)
improvement = results_optimized['elem_acc'] - results_standard['elem_acc']
print(f"Element-wise Acc: {results_standard['elem_acc']:.2f}% → {results_optimized['elem_acc']:.2f}% "
      f"(+{improvement:.2f}%)")

improvement_exact = results_optimized['exact_match_acc'] - results_standard['exact_match_acc']
print(f"Exact-match Acc:  {results_standard['exact_match_acc']:.2f}% → {results_optimized['exact_match_acc']:.2f}% "
      f"(+{improvement_exact:.2f}%)")

print("\nPer-class F1 improvements:")
for i in range(5):
    f1_std = results_standard['f1s'][i].item()
    f1_opt = results_optimized['f1s'][i].item()
    delta = f1_opt - f1_std
    arrow = "✓" if delta > 0.01 else "~"
    print(f"  Class {i}: {f1_std:.3f} → {f1_opt:.3f} ({delta:+.3f}) {arrow}")

print("="*70)


# Save results
results_dict = {
    'experiment': 'Step 5: Final Optimization',
    'architecture': 'ResNet18',
    'resolution': '128x128',
    'dropout': 0.35,
    'epochs': 30,
    'standard_thresholds': {
        'test_loss': float(results_standard['test_loss']),
        'element_wise_acc': float(results_standard['elem_acc']),
        'exact_match_acc': float(results_standard['exact_match_acc']),
        'class_metrics': {
            f'class_{i}': {
                'precision': float(results_standard['precisions'][i]),
                'recall': float(results_standard['recalls'][i]),
                'f1': float(results_standard['f1s'][i]),
                'tp': int(results_standard['tp'][i]),
                'fp': int(results_standard['fp'][i]),
                'fn': int(results_standard['fn'][i])
            } for i in range(5)
        }
    },
    'optimized_thresholds': {
        'thresholds': optimal_thresholds.tolist(),
        'test_loss': float(results_optimized['test_loss']),
        'element_wise_acc': float(results_optimized['elem_acc']),
        'exact_match_acc': float(results_optimized['exact_match_acc']),
        'class_metrics': {
            f'class_{i}': {
                'precision': float(results_optimized['precisions'][i]),
                'recall': float(results_optimized['recalls'][i]),
                'f1': float(results_optimized['f1s'][i]),
                'tp': int(results_optimized['tp'][i]),
                'fp': int(results_optimized['fp'][i]),
                'fn': int(results_optimized['fn'][i])
            } for i in range(5)
        }
    },
    'improvement': {
        'element_wise_acc': float(improvement),
        'exact_match_acc': float(improvement_exact)
    }
}

with open('Final_Model.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n✓ Saved: Final_Model.json")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss
ax1.plot(train_losses, label="Train", linewidth=2)
ax1.plot(val_losses, label="Val", linewidth=2)
ax1.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Loss vs. Epoch (30 epochs, dropout 0.35)", fontsize=14, weight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
ax2.plot(train_accs, label="Train", linewidth=2)
ax2.plot(val_accs, label="Val", linewidth=2)
ax2.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Accuracy vs. Epoch (30 epochs, dropout 0.35)", fontsize=14, weight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('step5_training_curves.png', dpi=150, bbox_inches='tight')
print("✓ Saved: step5_training_curves.png")
plt.show()

# test_loss = 0.0
# elem_correct = 0
# elem_total = 0

# # For exact-match and per-class metrics
# all_tp = torch.zeros(5, dtype=torch.long)
# all_fp = torch.zeros(5, dtype=torch.long)
# all_fn = torch.zeros(5, dtype=torch.long)
# exact_match_count = 0
# num_samples = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device).float()   # shape [B, 3], values {0,1}

#         logits = model(images)               # [B, 3]
#         loss = criterion(logits, labels)     # BCEWithLogitsLoss
#         test_loss += loss.item()

#         probs = torch.sigmoid(logits)
#         preds = (probs > 0.5).float()        # threshold

#         # Element-wise accuracy
#         elem_correct += (preds == labels).sum().item()
#         elem_total   += labels.numel()

#         # Exact-match accuracy (all labels correct for a sample)
#         exact_match_count += (preds.eq(labels).all(dim=1)).sum().item()
#         num_samples += labels.size(0)

#         # Per-class TP/FP/FN
#         # Convert to ints for counting
#         p = preds.long()
#         y = labels.long()
#         tp = (p.eq(1) & y.eq(1)).sum(dim=0)
#         fp = (p.eq(1) & y.eq(0)).sum(dim=0)
#         fn = (p.eq(0) & y.eq(1)).sum(dim=0)

#         all_tp += tp.cpu()
#         all_fp += fp.cpu()
#         all_fn += fn.cpu()

# avg_test_loss = test_loss / len(test_loader)
# elem_acc = 100.0 * elem_correct / elem_total
# exact_match_acc = 100.0 * exact_match_count / num_samples

# # Per-class precision/recall/F1 (safe division)
# eps = 1e-8
# precisions = (all_tp.float()) / (all_tp + all_fp + eps).float()
# recalls    = (all_tp.float()) / (all_tp + all_fn + eps).float()
# f1s        = 2 * precisions * recalls / (precisions + recalls + eps)

# print("\n=== Test Results ===")
# print(f"Test Loss:           {avg_test_loss:.4f}")
# print(f"Element-wise Acc:    {elem_acc:.2f}%")
# print(f"Exact-match Acc:     {exact_match_acc:.2f}%")

# for i in range(5):
#     print(f"Class {i}: "
#           f"Precision={precisions[i].item():.3f}, "
#           f"Recall={recalls[i].item():.3f}, "
#           f"F1={f1s[i].item():.3f}, "
#           f"TP={all_tp[i].item()}, FP={all_fp[i].item()}, FN={all_fn[i].item()}")

# results = {
#     'experiment': 'Step 2: Higher Resolution (Isolated)',
#     'architecture': 'ResNet18',
#     'resolution': '128x128',
#     'modifications': 'Resolution: 128x128 (ONLY change - no class weights)',
#     'test_loss': float(avg_test_loss),
#     'element_wise_acc': float(elem_acc),
#     'exact_match_acc': float(exact_match_acc),
#     'class_metrics': {
#         f'class_{i}': {
#             'precision': float(precisions[i]),
#             'recall': float(recalls[i]),
#             'f1': float(f1s[i]),
#             'tp': int(all_tp[i]),
#             'fp': int(all_fp[i]),
#             'fn': int(all_fn[i])
#         } for i in range(5)
#     }
# }

# with open('step2_results.json', 'w') as f:
#     json.dump(results, f, indent=2)

# print("\n✓ Saved: step2_results.json")
# # Loss vs epoch
# # Plot training curves
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# # Loss
# ax1.plot(train_losses, label="Train", linewidth=2)
# ax1.plot(val_losses, label="Val", linewidth=2)
# ax1.set_xlabel("Epoch", fontsize=12)
# ax1.set_ylabel("Loss", fontsize=12)
# ax1.set_title("Step 2: Loss vs. Epoch (128x128)", fontsize=14, weight='bold')
# ax1.legend()
# ax1.grid(alpha=0.3)

# # Accuracy
# ax2.plot(train_accs, label="Train", linewidth=2)
# ax2.plot(val_accs, label="Val", linewidth=2)
# ax2.set_xlabel("Epoch", fontsize=12)
# ax2.set_ylabel("Accuracy (%)", fontsize=12)
# ax2.set_title("Step 2: Accuracy vs. Epoch (128x128)", fontsize=14, weight='bold')
# ax2.legend()
# ax2.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig('step2_training_curves.png', dpi=150, bbox_inches='tight')
# print("✓ Saved: step2_training_curves.png")
# plt.show()

