#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Plant Disease Detection - Training Script
# Project: TI5A2 - Pembelajaran Mesin 2
# Author: Farrel Ghozy Affifudin (452024611053)
# Class: TI5 A2
# Universitas Darussalam Gontor

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import json

class Config:
    DATA_DIR = 'plantvillage_dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_DIR = 'models'
    MODEL_NAME = 'plant_disease_mobilenetv3_small'
    TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.tflite')
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    AUGMENTATION = True
    USE_TRANSFER_LEARNING = True
    FREEZE_BASE = True

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_model(model_type='mobilenetv3_small'):
    if model_type == 'mobilenetv2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif model_type == 'mobilenetv3_small':
        base_model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif model_type == 'efficientnet_b0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if config.USE_TRANSFER_LEARNING and config.FREEZE_BASE:
        base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model_type='mobilenetv3_small'):
    print(f"\\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}\\n")
    
    train_generator, val_generator, test_generator = create_data_generators()
    num_classes = train_generator.num_classes
    class_names = train_generator.class_indices
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {list(class_names.keys())}")
    
    model = build_model(model_type)
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_{model_type}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    start_time = datetime.datetime.now()
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\\nTraining completed in {training_time:.2f} seconds")
    
    with open(os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_{model_type}_history.json'), 'w') as f:
        json.dump(history.history, f)
    
    return model, history, test_generator, class_names

def evaluate_model(model, test_generator, class_names):
    print(f"\\n{'='*60}")
    print("Evaluating Model")
    print(f"{'='*60}\\n")
    
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    
    print("\\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=list(class_names.keys()),
        digits=4
    ))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(class_names.keys()),
                yticklabels=list(class_names.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'confusion_matrix.png'), dpi=300)
    print(f"\\nConfusion matrix saved to: {os.path.join(config.MODEL_DIR, 'confusion_matrix.png')}")
    
    plot_training_history(history)
    
    return y_true, y_pred

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'training_history.png'), dpi=300)
    print(f"Training history plot saved to: {os.path.join(config.MODEL_DIR, 'training_history.png')}")

def convert_to_tflite(model, class_names):
    print(f"\\n{'='*60}")
    print("Converting Model to TensorFlow Lite")
    print(f"{'='*60}\\n")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {config.TFLITE_MODEL_PATH}")
    
    labels_path = os.path.join(config.MODEL_DIR, 'labels.txt')
    with open(labels_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\\n")
    
    print(f"Labels saved to: {labels_path}")
    
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model size: {model_size_mb:.2f} MB")
    
    return tflite_model

def analyze_inference_time(model, test_generator):
    print(f"\\n{'='*60}")
    print("Analyzing Inference Time")
    print(f"{'='*60}\\n")
    
    import time
    inference_times = []
    
    for i in range(100):
        test_generator.reset()
        x_batch, y_batch = next(test_generator)
        
        start_time = time.time()
        predictions = model.predict(x_batch[:1], verbose=0)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        inference_times.append(inference_time)
    
    inference_times = np.array(inference_times)
    
    print(f"Average inference time: {inference_times.mean():.2f} ms")
    print(f"Median inference time: {np.median(inference_times):.2f} ms")
    print(f"Min inference time: {inference_times.min():.2f} ms")
    print(f"Max inference time: {inference_times.max():.2f} ms")
    
    return inference_times

if __name__ == '__main__':
    models_to_train = ['mobilenetv3_small', 'mobilenetv2', 'efficientnet_b0']
    
    best_model = None
    best_accuracy = 0
    
    for model_type in models_to_train:
        try:
            model, history, test_generator, class_names = train_model(model_type)
            y_true, y_pred = evaluate_model(model, test_generator, class_names)
            inference_times = analyze_inference_time(model, test_generator)
            tflite_model = convert_to_tflite(model, class_names)
            
            test_accuracy = history.history['val_accuracy'][-1]
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = model_type
                
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    print(f"\\n{'='*60}")
    print(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}\\n")
