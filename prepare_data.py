"""Data Preparation Script for Animal Classification.

This script organizes the animal images into train/val/test splits.
"""
import os
import shutil
import json
import random
from pathlib import Path

def prepare_dataset(data_dir='data', output_dir='data/dataset', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Organize images into train/val/test splits.

    Args:
        data_dir: Directory containing animal folders.
        output_dir: Output directory for organized dataset.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    # Load translation dictionary for scientific to common names.
    translation_path = os.path.join(data_dir, 'translation.json')
    with open(translation_path, 'r') as f:
        translation = json.load(f)
    # Create output directories for each split.
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    # Get all animal directories excluding dataset folder.
    animal_dirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d != 'dataset']
    print(f"Found {len(animal_dirs)} animal categories")
    # Process each animal category.
    stats = {'train': 0, 'val': 0, 'test': 0}
    class_info = {}
    
    for idx, animal_name in enumerate(sorted(animal_dirs)):
        animal_path = os.path.join(data_dir, animal_name)
        common_name = translation.get(animal_name, animal_name)
        # Get all images for this animal category.
        images = [f for f in os.listdir(animal_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        # Shuffle images for random split with fixed seed for reproducibility.
        random.seed(42)
        random.shuffle(images)
        # Calculate split indices based on ratios.
        n_images = len(images)
        train_end = int(n_images * train_ratio)
        val_end = train_end + int(n_images * val_ratio)
        # Split images into train, validation, and test sets.
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        # Create class directory in each split.
        for split in ['train', 'val', 'test']:
            class_dir = os.path.join(output_dir, split, animal_name)
            os.makedirs(class_dir, exist_ok=True)
        # Copy training images to train directory.
        for img in train_images:
            src = os.path.join(animal_path, img)
            dst = os.path.join(output_dir, 'train', animal_name, img)
            shutil.copy2(src, dst)
            stats['train'] += 1
        # Copy validation images to val directory.
        for img in val_images:
            src = os.path.join(animal_path, img)
            dst = os.path.join(output_dir, 'val', animal_name, img)
            shutil.copy2(src, dst)
            stats['val'] += 1
        # Copy test images to test directory.
        for img in test_images:
            src = os.path.join(animal_path, img)
            dst = os.path.join(output_dir, 'test', animal_name, img)
            shutil.copy2(src, dst)
            stats['test'] += 1
        # Store class information for later use.
        class_info[animal_name] = {
            'index': idx,
            'common_name': common_name,
            'total_images': n_images
        }
        print(f"Processed {animal_name} ({common_name}): {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    # Save class information to JSON file.
    class_info_path = os.path.join(output_dir, 'class_info.json')
    with open(class_info_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print(f"Train images: {stats['train']}")
    print(f"Validation images: {stats['val']}")
    print(f"Test images: {stats['test']}")
    print(f"Total images: {sum(stats.values())}")
    print(f"Number of classes: {len(class_info)}")
    print(f"\nClass information saved to: {class_info_path}")

if __name__ == "__main__":
    prepare_dataset()
