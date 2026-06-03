"""Symmetric Cross Entropy loss helpers."""

import torch.nn as nn
import torch
from torch.nn.functional import F

from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.routes import ROOT

class SCELoss(nn.Module):
    def __init__(self, alpha, device, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class SCEMethod():
    def __init__(self, config, model, device):
        super().__init__()
        self.config = config
        self.num_class = config.num_class
        alpha = config.alpha
        beta = config.beta
        self.device = device
        self.epochs = config.epochs
        self.lr = config.lr
        backbone = (config.model).backbone
        self.checkpoint_path = ROOT / "outputs" / "checkpoint" / f"{backbone}_sce"


        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        self.model = model.to(device)

        self.criterion = SCELoss(alpha= alpha, beta= beta, device=device, num_classes=self.num_class)
        self.optimizer = torch.optim.SGD(params=model.parameters(),
                                         lr=self.lr,
                                         momentum=0.9,
                                         nesterov=True)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.1)
                                         
    def train(self, train_loader, val_loader):
        start_epoch = 0

        # checkpoint existente
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1

            print(f"Checkpoint encontrado. Continuando do epoch {start_epoch}")

        except FileNotFoundError as e:
            print("Treinando pela primeira vez")
            
        
        for epoch in range(start_epoch, self.epochs):
                self.model.train()

                train_loss_sum = 0.0
                train_correct = 0
                train_total = 0

                for imgs, labels in train_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.long().to(self.device)

                    self.optimizer.zero_grad()

                    pred = self.model(imgs)
                    loss = self.criterion(pred, labels)

                    loss.backward()
                    self.optimizer.step()

                    batch_size = imgs.size(0)
                    train_loss_sum += loss.item() * batch_size

                    predicted_classes = pred.argmax(dim=1)
                    train_correct += (predicted_classes == labels).sum().item()
                    train_total += batch_size

                train_loss = train_loss_sum / train_total
                train_acc = train_correct / train_total

                # validação
                self.model.eval()

                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device)
                        labels = labels.long().to(self.device)

                        pred = self.model(imgs)
                        loss = self.criterion(pred, labels)

                        batch_size = imgs.size(0)
                        val_loss_sum += loss.item() * batch_size

                        predicted_classes = pred.argmax(dim=1)
                        val_correct += (predicted_classes == labels).sum().item()
                        val_total += batch_size

                val_loss = val_loss_sum / val_total
                acc = val_correct / val_total

                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Train Acc: {train_acc:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"Val Acc: {acc:.4f}"
                )

                save_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    acc,
                    getattr(self, "configs", {}),
                )

                self.scheduler.step(val_loss)
        return self.model, (val_loss, acc)

def test(self, test_loader):
    try:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint carregado para teste: {self.checkpoint_path}")

    except FileNotFoundError:
        print("Nenhum checkpoint encontrado. Testando com o modelo atual.")

    self.model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(self.device)
            labels = labels.long().to(self.device)

            pred = self.model(imgs)
            loss = self.criterion(pred, labels)

            batch_size = imgs.size(0)
            test_loss_sum += loss.item() * batch_size

            predicted_classes = pred.argmax(dim=1)

            test_correct += (predicted_classes == labels).sum().item()
            test_total += batch_size

            all_preds.append(predicted_classes.cpu())
            all_labels.append(labels.cpu())

    test_loss = test_loss_sum / test_total
    test_acc = test_correct / test_total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

    return test_loss, test_acc, all_preds, all_labels