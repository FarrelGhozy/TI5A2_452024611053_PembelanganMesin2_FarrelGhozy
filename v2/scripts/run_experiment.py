#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 — Runner eksperimen Final Project Pembelajaran Mesin 2.
Farrel Ghozy Affifudin (452024611053) — TI5 A2 — Universitas Darussalam Gontor

Eksperimen:
  e1  Custom CNN (baseline, dari nol)               lr=1e-3, Adam
  e2  MobileNetV3-Small (transfer learning, frozen)  lr=1e-3, Adam
  e3  EfficientNet-B0   (transfer learning, frozen)  lr=1e-3, Adam
  e4  MobileNetV3-Small (transfer learning, fine-tune semua) lr=1e-4, Adam

Cara pakai:
  python run_experiment.py --exp e1 --epochs 12 --batch 32
Hasil → <out_root>/<exp>/  (metrics.json, history.json, plot, CM, report)
"""
import argparse
import copy
import json
import os
import sys
import time
import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, make_loaders, build_model, freeze_backbone,
                    count_parameters, evaluate_model, save_json,
                    plot_history, plot_confusion_matrix,
                    save_classification_report, CustomCNN)

EXPERIMENTS = {
    'e1': dict(name='e1_custom_cnn', pretrained=False, freeze=False, lr=1e-3, aug=True,
               desc='Custom CNN baseline (from scratch)'),
    'e2': dict(name='e2_mobilenetv3', pretrained=True, freeze=True, lr=1e-3, aug=True,
               desc='MobileNetV3-Small TL, backbone frozen'),
    'e3': dict(name='e3_efficientnet', pretrained=True, freeze=True, lr=1e-3, aug=True,
               desc='EfficientNet-B0 TL, backbone frozen'),
    'e4': dict(name='e4_mobilenetv3_ft', pretrained=True, freeze=False, lr=1e-4, aug=True,
               desc='MobileNetV3-Small TL, fine-tune penuh'),
}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total, time.time() - t0


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=list(EXPERIMENTS))
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--data-root', default=os.path.expanduser('~/pm2_v2/data/plantvillage_split'))
    ap.add_argument('--out-root', default=os.path.expanduser('~/pm2_v2/results'))
    ap.add_argument('--model-root', default=os.path.expanduser('~/pm2_v2/models'))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.exp]
    lr = args.lr or cfg['lr']
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | {torch.cuda.get_device_name(0) if device.type=="cuda" else ""}')
    torch.backends.cudnn.benchmark = True

    out_dir = os.path.join(args.out_root, args.exp)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)

    print(f'\n=== {args.exp}: {cfg["desc"]} ===')
    print(f'Config: epochs={args.epochs} batch={args.batch} lr={lr} workers={args.workers}')

    train_loader, val_loader, test_loader, class_names = make_loaders(
        args.data_root, args.batch, args.workers, args.seed)
    num_classes = len(class_names)
    print(f'Kelas: {num_classes} | train={len(train_loader.dataset)} '
          f'val={len(val_loader.dataset)} test={len(test_loader.dataset)}')

    model = build_model(cfg['name'], num_classes, pretrained=cfg['pretrained'])
    freeze_backbone(model, cfg['freeze'])
    model = model.to(device)
    total_p, train_p = count_parameters(model)
    print(f'Param total: {total_p:,} | trainable: {train_p:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    early_stop_counter = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, dt = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['lr'].append(cur_lr)
        print(f'Epoch {epoch:02d}/{args.epochs} | '
              f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | '
              f'val_loss={va_loss:.4f} val_acc={va_acc:.4f} | '
              f'lr={cur_lr:.2e} | {dt:.0f}s')

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f'Early stopping @ epoch {epoch}')
                break

    train_secs = time.time() - t_start
    model.load_state_dict(best_state)

    # ---- evaluasi test ----
    print('\n=== Evaluasi Test Set ===')
    res = evaluate_model(model, test_loader, device, num_classes)

    metrics = {
        'experiment': args.exp,
        'description': cfg['desc'],
        'model': cfg['name'],
        'config': {'epochs': args.epochs, 'batch': args.batch, 'lr': lr,
                   'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
                   'pretrained': cfg['pretrained'], 'freeze_backbone': cfg['freeze'],
                   'augmentation': cfg['aug'], 'early_stopping_patience': args.patience,
                   'seed': args.seed, 'epochs_ran': len(history['train_loss']),
                   'best_epoch': best_epoch},
        'train_seconds': train_secs,
        'best_val_loss': float(best_val_loss),
        'test': {k: v for k, v in res.items()
                 if k in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')},
        'per_class': {
            'classes': class_names,
            'precision': res['per_class_precision'],
            'recall': res['per_class_recall'],
            'f1': res['per_class_f1'],
        },
        'params': {'total': total_p, 'trainable': train_p},
        'class_names': class_names,
        'run_time': datetime.datetime.now().isoformat(),
    }
    save_json(metrics, os.path.join(out_dir, 'metrics.json'))
    save_json(history, os.path.join(out_dir, 'history.json'))
    save_json(metrics['test'], os.path.join(out_dir, 'test_metrics.json'))

    plot_history(history, os.path.join(out_dir, 'training_curves.png'))
    cm = plot_confusion_matrix(res['y_true'], res['y_pred'], class_names,
                               os.path.join(out_dir, 'confusion_matrix.png'))
    save_classification_report(res['y_true'], res['y_pred'], class_names,
                               os.path.join(out_dir, 'classification_report.txt'))

    # ---- error analysis: top-25 misclassification + per-class worst ----
    y_true, y_pred, probs = res['y_true'], res['y_pred'], res['probs']
    paths = res['paths']
    mis_idx = np.where(y_true != y_pred)[0]
    conf = probs.max(axis=1)
    order = mis_idx[np.argsort(conf[mis_idx])]  # paling yakin tapi salah
    errors = []
    for i in order[:25]:
        errors.append({'path': paths[i], 'true': class_names[y_true[i]],
                       'pred': class_names[y_pred[i]], 'confidence': float(conf[i])})
    save_json(errors, os.path.join(out_dir, 'top_errors.json'))

    # akurasi per kelas (terburuk & terbaik)
    per_class_acc = {}
    for c in range(num_classes):
        mask = y_true == c
        per_class_acc[class_names[c]] = float((y_pred[mask] == c).mean())
    worst = sorted(per_class_acc.items(), key=lambda kv: kv[1])[:10]
    best = sorted(per_class_acc.items(), key=lambda kv: -kv[1])[:10]
    per_class_info = {'worst': [{'class': k, 'acc': v} for k, v in worst],
                      'best': [{'class': k, 'acc': v} for k, v in best]}
    save_json(per_class_info, os.path.join(out_dir, 'per_class_analysis.json'))

    torch.save(best_state, os.path.join(args.model_root, f'{args.exp}_best.pt'))

    print('\n=== RINGKASAN ===')
    t = metrics['test']
    print(f'Test Accuracy : {t["accuracy"]:.4f}')
    print(f'Precision     : {t["precision_macro"]:.4f}')
    print(f'Recall        : {t["recall_macro"]:.4f}')
    print(f'F1 (macro)    : {t["f1_macro"]:.4f}')
    print(f'Waktu training: {train_secs/60:.1f} menit ({train_secs:.0f}s)')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
