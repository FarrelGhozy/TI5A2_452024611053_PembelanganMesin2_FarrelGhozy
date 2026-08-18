#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 — Prepare PlantVillage dataset: unzip + stratified split (80/10/10).
Hasil: <data_root>/plantvillage_split/{train,val,test}/<class>/*.jpg (symlink)
"""
import argparse
import os
import random
import zipfile
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_class_dirs(root, max_depth=4):
    """Cari rekursif semua folder yang berisi gambar langsung (= folder kelas)."""
    found = []
    queue = [(root, 0)]
    while queue:
        d, depth = queue.pop(0)
        if depth > max_depth or not os.path.isdir(d):
            continue
        for e in sorted(os.listdir(d)):
            full = os.path.join(d, e)
            if not os.path.isdir(full):
                continue
            children = os.listdir(full)
            has_imgs = any(f.lower().endswith(tuple(IMG_EXTS)) for f in children)
            if has_imgs:
                found.append(full)
            else:
                queue.append((full, depth + 1))
    return found


def collect_images(class_dirs):
    """{class_name: [abs paths]}"""
    mapping = defaultdict(list)
    for cd in class_dirs:
        name = os.path.basename(cd)
        for f in sorted(os.listdir(cd)):
            if f.lower().endswith(tuple(IMG_EXTS)):
                mapping[name].append(os.path.join(cd, f))
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=os.path.expanduser('~/pm2_v2/data/plantvillage.zip'))
    ap.add_argument('--data-root', default=os.path.expanduser('~/pm2_v2/data'))
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    extract_dir = os.path.join(args.data_root, 'plantvillage_raw')
    split_dir = os.path.join(args.data_root, 'plantvillage_split')

    # ---- unzip ----
    if not os.path.isdir(extract_dir) or not any(os.scandir(extract_dir)):
        print(f'[1/3] Unzip {args.zip} → {extract_dir}')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(extract_dir)
    else:
        print(f'[1/3] {extract_dir} sudah ada, skip unzip')

    # ---- cari folder kelas (rekursif, bisa beberapa level) ----
    class_dirs = find_class_dirs(extract_dir)
    if not class_dirs:
        raise SystemExit('Tidak menemukan folder kelas! Cek struktur zip.')
    # dataset ini punya 3 varian (color/grayscale/segmented) — pilih 'color'
    colored = [d for d in class_dirs if 'color' in d.replace(os.sep, '/').lower()]
    if colored and len(colored) < len(class_dirs):
        print(f'[2/3] Filter varian: {len(class_dirs)} folder kelas → {len(colored)} (color only)')
        class_dirs = colored

    mapping = collect_images(class_dirs)
    print(f'[2/3] Ditemukan {len(mapping)} kelas, total {sum(len(v) for v in mapping.values())} gambar')

    # ---- split stratified ----
    os.makedirs(split_dir, exist_ok=True)
    counts = {'train': 0, 'val': 0, 'test': 0}
    per_class = {}
    for cname, paths in sorted(mapping.items()):
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        splits = {'train': paths[:n_train],
                  'val': paths[n_train:n_train + n_val],
                  'test': paths[n_train + n_val:]}
        per_class[cname] = {k: len(v) for k, v in splits.items()}
        for split_name, spaths in splits.items():
            dest_dir = os.path.join(split_dir, split_name, cname)
            os.makedirs(dest_dir, exist_ok=True)
            for src in spaths:
                dst = os.path.join(dest_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass  # sudah ada
            counts[split_name] += len(spaths)

    print(f'[3/3] Split selesai: {counts}')
    print(f'Split dir: {split_dir}')
    for cname, c in list(per_class.items())[:5]:
        print(f'  {cname}: {c}')
    print(f'  ... ({len(per_class)} kelas total)')


if __name__ == '__main__':
    main()
