"""Cross Entropy loss helpers."""

from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch


class CrossEntropyMethod(nn.Module):
    """Compatibility wrapper for older method-based training code."""

    def __init__(self, config=None):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        optimizer = torch.optim.SGD(params=model.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         nesterov=True)

    def compute_loss(self, model, images, targets):
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return loss, logits

    def validation_loss(self, logits, targets):
        return self.loss_fn(logits, targets)




model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


# ----------------------  ----------------------

with open(txtfile, 'w') as f:
    f.write('epoch train_acc val_acc test_acc\n')

# ----------------------  ----------------------
def accuracy(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels, *_ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100. * correct / total

# ----------------------  ----------------------
best_val_acc = 0.0
test_at_best = 0.0
best_test_acc = 0.0
last5 = deque(maxlen=5)

for epoch in range(0, args.epochs):
    model.train()
    for imgs, labels, *_ in train_loader:
        imgs = imgs.to(device)
        labels = labels.long().to(device)
        # print("imgs device:", imgs.device)
        # print("labels type:", type(labels))
        # if isinstance(labels, torch.Tensor):
        #     print("labels device:", labels.device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    # 
    tr_acc = accuracy(train_loader)
    val_acc = accuracy(val_loader)
    te_acc = accuracy(test_loader)

    scheduler.step()
    # 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_at_best = te_acc
    best_test_acc = max(best_test_acc, te_acc)
    last5.append(te_acc)

    #  log
    with open(txtfile, 'a') as f:
        f.write(f'{epoch} {tr_acc:.4f} {val_acc:.4f} {te_acc:.4f}\n')
    print(f'Epoch {epoch:03d}/{args.epochs} | train {tr_acc:.2f}%  val {val_acc:.2f}%  test {te_acc:.2f}%')

    noise_rate_str = str(args.noise_rate).replace('.', '_')
    save_dir = os.path.join("/mnt/ssd1/user/ce", args.dataset, args.noise_type, f"nr{noise_rate_str}")
    os.makedirs(save_dir, exist_ok=True)

    # torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))


# ----------------------  ----------------------
avg_last5 = sum(last5) / len(last5)
print('\n======== Final Report ========')
print(f'1) Val-best test acc : {test_at_best:.2f}%')
print(f'2) Best   test acc   : {best_test_acc:.2f}%')
print(f'3) Last-5 test avg   : {avg_last5:.2f}%')

#  txt
with open(txtfile, 'a') as f:
    f.write('\n# Final Report\n')
    f.write(f'val_best_test_acc {test_at_best:.3f}\n')
    f.write(f'test_max_acc      {best_test_acc:.3f}\n')
    f.write(f'test_last5_avg    {avg_last5:.3f}\n')