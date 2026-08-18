#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 — Common utilities: dataset, transforms, metrics, plotting.
Final Project Pembelajaran Mesin 2 — Farrel Ghozy Affifudin (452024611053)
"""
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- dataset
class ImageFolderWithPaths(ImageFolder):
    """ImageFolder yang juga mengembalikan path file (untuk error analysis)."""

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loaders(data_root, batch_size, num_workers=6, seed=42):
    train_ds = ImageFolderWithPaths(
        f'{data_root}/train', transform=get_transforms(augment=True))
    val_ds = ImageFolderWithPaths(
        f'{data_root}/val', transform=get_transforms(augment=False))
    test_ds = ImageFolderWithPaths(
        f'{data_root}/test', transform=get_transforms(augment=False))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g,
                              persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=True,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=True,
                             pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------------------------------------------- models
class CustomCNN(nn.Module):
    """Baseline: CNN sederhana dari nol (tanpa pretrained)."""

    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes, pretrained=True):
    import torchvision.models as tv
    if name == 'e1_custom_cnn':
        return CustomCNN(num_classes)
    if name == 'e2_mobilenetv3':
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    if name == 'e3_efficientnet':
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == 'e4_mobilenetv3_ft':
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    raise ValueError(name)


def freeze_backbone(model, freeze=True):
    """Freeze semua layer kecuali classifier/head (nett: SE block juga punya 'fc')."""
    if not freeze:
        return
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------- metrics
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_paths, all_probs = [], [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
            all_paths.extend(p)
            all_probs.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None)
    return {
        'accuracy': float(acc),
        'precision_macro': float(p),
        'recall_macro': float(r),
        'f1_macro': float(f1),
        'per_class_precision': per_class[0].tolist(),
        'per_class_recall': per_class[1].tolist(),
        'per_class_f1': per_class[2].tolist(),
        'y_true': y_true,
        'y_pred': y_pred,
        'probs': probs,
        'paths': all_paths,
    }


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- plots
def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0].plot(history['val_acc'], label='Val Acc', marker='o')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[1].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax, annot_kws={'size': 5})
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.xticks(rotation=90, fontsize=6); plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def save_classification_report(y_true, y_pred, class_names, out_path):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(out_path, 'w') as f:
        f.write(report)
    return report
