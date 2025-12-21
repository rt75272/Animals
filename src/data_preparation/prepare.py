"""Data Preparation Script for Animal Classification.

This script organizes the animal images into train/val/test splits.
"""
import os
import shutil
import json
import random
from pathlib import Path

import sys
from pathlib import Path

# Add root directory to sys.path to allow imports from config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import paths, params


_IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


def _maybe_nested_dir(parent: Path, child_name: str) -> Path:
    """Handle layouts like 'Training Data/Training Data/<classes>'."""
    direct = parent / child_name
    nested = direct / child_name
    if nested.exists() and nested.is_dir():
        return nested
    return direct


def _load_translation_if_present(translation_path: Path) -> dict:
    if translation_path.exists():
        with open(translation_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    return {}


def _iter_class_dirs(root: Path) -> list[Path]:
    """Return immediate child directories representing classes."""
    if not root.exists() or not root.is_dir():
        return []
    return [d for d in root.iterdir() if d.is_dir()]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer symlinks to avoid duplicating large datasets; fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_from_presplit_dirs(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    output_dir: Path = paths.PROCESSED_DATA_DIR,
    translation_path: Path = paths.TRANSLATION_FILE,
) -> None:
    """Prepare dataset when data already comes with train/val/test splits.

    Expected input:
      train_dir/<class_name>/*images*
      val_dir/<class_name>/*images*
      test_dir/<class_name>/*images*

    Output:
      output_dir/train/<class_name>/...
      output_dir/val/<class_name>/...
      output_dir/test/<class_name>/...
    """
    translation = _load_translation_if_present(translation_path)
    split_sources = {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
    }

    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'test': 0}
    class_info: dict[str, dict] = {}

    classes = sorted({d.name for d in _iter_class_dirs(train_dir)} | {d.name for d in _iter_class_dirs(val_dir)} | {d.name for d in _iter_class_dirs(test_dir)})
    print(f"Found {len(classes)} classes across pre-split folders")

    for idx, class_name in enumerate(classes):
        common_name = translation.get(class_name, class_name.replace('-', ' ').replace('_', ' ').title())
        total_images = 0
        for split, src_root in split_sources.items():
            src_class_dir = src_root / class_name
            if not src_class_dir.exists():
                continue
            dst_class_dir = output_dir / split / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            images = [p for p in src_class_dir.rglob('*') if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]
            for img_path in images:
                # Preserve filename; do not preserve nested structure.
                dst = dst_class_dir / img_path.name
                _link_or_copy(img_path, dst)
                stats[split] += 1
                total_images += 1

        if total_images == 0:
            continue

        class_info[class_name] = {
            'index': idx,
            'common_name': common_name,
            'total_images': total_images,
        }
        print(f"Prepared {class_name} ({common_name}): total {total_images}")

    class_info_path = output_dir / 'class_info.json'
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

def prepare_dataset(data_dir=paths.DATA_DIR, 
                    output_dir=paths.PROCESSED_DATA_DIR, 
                    train_ratio=params.TRAIN_RATIO, 
                    val_ratio=params.VAL_RATIO, 
                    test_ratio=params.TEST_RATIO):
    """Organize images into train/val/test splits.

    Args:
        data_dir: Directory containing animal folders.
        output_dir: Output directory for organized dataset.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # If the dataset is already split into separate folders (your current layout),
    # normalize it into output_dir/train|val|test for the training code.
    train_src = _maybe_nested_dir(data_dir, 'Training Data')
    val_src = _maybe_nested_dir(data_dir, 'Validation Data')
    test_src = _maybe_nested_dir(data_dir, 'Testing Data')
    if train_src.exists() and val_src.exists() and test_src.exists():
        print("Detected pre-split dataset under data/. Building data/dataset...")
        return prepare_from_presplit_dirs(
            train_dir=train_src,
            val_dir=val_src,
            test_dir=test_src,
            output_dir=output_dir,
            translation_path=paths.TRANSLATION_FILE,
        )

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    translation = _load_translation_if_present(paths.TRANSLATION_FILE)
        
    # Create output directories for each split.
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
    # Get all animal directories excluding dataset folder.
    animal_dirs = [d for d in data_dir.iterdir()
                   if d.is_dir() and d.name != 'dataset']
    print(f"Found {len(animal_dirs)} animal categories")
    
    # Process each animal category.
    stats = {'train': 0, 'val': 0, 'test': 0}
    class_info = {}
    
    for idx, animal_path in enumerate(sorted(animal_dirs)):
        animal_name = animal_path.name
        common_name = translation.get(animal_name, animal_name.replace('-', ' ').replace('_', ' ').title())
        
        # Get all images for this animal category.
        images = [f for f in animal_path.glob('*') if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES]
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
            class_dir = output_dir / split / animal_name
            class_dir.mkdir(exist_ok=True)
            
        # Copy images to their respective directories.
        for img_list, split_name in [(train_images, 'train'), (val_images, 'val'), (test_images, 'test')]:
            for img_path in img_list:
                dst = output_dir / split_name / animal_name / img_path.name
                shutil.copy2(img_path, dst)
                stats[split_name] += 1
                
        # Store class information for later use.
        class_info[animal_name] = {
            'index': idx,
            'common_name': common_name,
            'total_images': n_images
        }
        print(f"Processed {animal_name} ({common_name}): {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
    # Save class information to JSON file.
    class_info_path = output_dir / 'class_info.json'
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
