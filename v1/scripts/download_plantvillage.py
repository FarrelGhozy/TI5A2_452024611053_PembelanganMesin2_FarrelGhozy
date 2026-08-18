#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PlantVillage Dataset Download Script
# Project: TI5A2 - Pembelajaran Mesin 2
# Author: Farrel Ghozy Affifudin (452024611053)
# Class: TI5 A2
# Universitas Darussalam Gontor

import os
import requests
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/abdallahalidev/plantvillage-dataset"
OUTPUT_DIR = "plantvillage_dataset"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("="*60)
print("PlantVillage Dataset Downloader")
print("="*60)
print(f"Target directory: {Path.cwd() / OUTPUT_DIR}")
print("="*60)

try:
    print("\\nDownloading PlantVillage dataset...")
    print("This may take a while (approx. 2-3 GB)")
    print()
    
    response = requests.get(DOWNLOAD_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192
    
    with open("plantvillage.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\\rProgress: {percent:.1f}% ({downloaded / (1024**3):.2f} GB)", end="")
    
    print("\\n\\nDownload complete! Extracting...")
    
    with zipfile.ZipFile("plantvillage.zip", 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)
    
    print("Extraction complete!")
    
    os.remove("plantvillage.zip")
    
    extracted_dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    if len(extracted_dirs) == 1:
        old_dir = os.path.join(OUTPUT_DIR, extracted_dirs[0])
        new_dir = os.path.join(OUTPUT_DIR, "plantvillage")
        os.rename(old_dir, new_dir)
        print(f"Renamed to: {new_dir}")
    
    print("\\n" + "="*60)
    print("Dataset setup complete!")
    print("="*60)
    print(f"Location: {Path.cwd() / OUTPUT_DIR}")
    
    if os.path.exists(os.path.join(OUTPUT_DIR, "train")):
        train_count = len(os.listdir(os.path.join(OUTPUT_DIR, "train")))
        print(f"Training samples: {train_count}")
    else:
        print("Warning: Training directory not found. Please check the structure.")
        
except Exception as e:
    print(f"\\nError: {e}")
    print("\\nAlternative: Download manually from Kaggle")
    print("URL: https://www.kaggle.com/abdallahalidev/plantvillage-dataset")
