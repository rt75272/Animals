"""Animal Classification Model Training Script.

This script trains a ResNet-based model for animal species classification with
advanced augmentation techniques and label smoothing for improved accuracy.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add root directory to sys.path to allow imports from config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import paths, params


class AnimalClassifier:
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, freeze_layers=True):
        """Initialize the animal classifier with improved architecture.

        Args:
            num_classes: Number of animal classes to predict.
            model_name: Base model architecture (resnet18, resnet50, etc.).
            pretrained: Whether to use pretrained weights.
            freeze_layers: Whether to freeze early layers for fine-tuning.
        """
        self.num_classes = num_classes
        # Check if CUDA is available for GPU acceleration.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pretrained model based on specified architecture.
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_features = self.model.fc.in_features
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_features = self.model.fc.in_features
        else:
            raise ValueError(f"Unknown model: {model_name}")
        # Freeze early layers to retain pretrained features and prevent overfitting.
        if freeze_layers and pretrained:
            for name, param in self.model.named_parameters():
                # Freeze all layers except layer4 and fc for fine-tuning.
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        # Replace final layer with enhanced classifier head with regularization.
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self.model = self.model.to(self.device)
        print(f"Model initialized: {model_name}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {num_classes}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train_epoch(self, dataloader, criterion, optimizer, scheduler=None):
        """Train for one epoch with gradient accumulation for stability.

        Args:
            dataloader: Training data loader.
            criterion: Loss function.
            optimizer: Optimizer for weight updates.
            scheduler: Learning rate scheduler (optional).
        Returns:
            Tuple of (epoch_loss, epoch_accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc='Training')
        for inputs, labels in pbar:
            # Move data to device (GPU or CPU).
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Zero gradients from previous iteration.
            optimizer.zero_grad()
            # Forward pass through the network.
            outputs = self.model(inputs)
            # Calculate loss between predictions and true labels.
            loss = criterion(outputs, labels)
            # Backward pass to compute gradients.
            loss.backward()
            # Clip gradients to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Update model weights.
            optimizer.step()
            # Step scheduler if using OneCycleLR.
            if scheduler is not None:
                scheduler.step()
            # Accumulate metrics for this batch.
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # Update progress bar with current metrics.
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        # Calculate average loss and accuracy for the epoch.
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader, criterion):
        """Validate the model on validation dataset.

        Args:
            dataloader: Validation data loader.
            criterion: Loss function.
        Returns:
            Tuple of (epoch_loss, epoch_accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        # Disable gradient computation for validation.
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass only (no gradient computation).
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # Accumulate validation metrics.
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, path, epoch, optimizer, train_acc, val_acc):
        """Save model checkpoint with training state.

        Args:
            path: File path to save checkpoint.
            epoch: Current epoch number.
            optimizer: Optimizer state.
            train_acc: Training accuracy.
            val_acc: Validation accuracy.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'num_classes': self.num_classes
        }, path)
        print(f"Checkpoint saved to {path}")
    def load_checkpoint(self, path):
        """Load model checkpoint from file.

        Args:
            path: File path to load checkpoint from.
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint

def get_data_transforms():
    """Define enhanced data augmentation and normalization transforms.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    # Training transformations with moderate augmentation to prevent overfitting.
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Moderate color jitter for better generalization without distortion.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Random erasing helps model learn robust features.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])
    # Validation transformations without augmentation for consistent evaluation.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training and validation metrics as line graphs.

    Args:
        history: Dictionary containing training metrics.
        save_path: Path to save the plot image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Plot loss curves to visualize training progression.
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    # Plot accuracy curves to monitor model performance.
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history plot saved to {save_path}")

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing for better generalization.

    Label smoothing prevents overconfident predictions and improves accuracy.
    """
    def __init__(self, smoothing=0.1):
        """Initialize label smoothing loss.

        Args:
            smoothing: Amount of smoothing (0.0 to 1.0).
        """
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        """Calculate smoothed cross entropy loss.
        
        Args:
            pred: Model predictions (logits).
            target: Ground truth labels.
        Returns:
            Smoothed loss value.
        """
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

def train_model(data_dir=paths.PROCESSED_DATA_DIR,
                model_name=params.MODEL_NAME,
                num_epochs=params.NUM_EPOCHS,
                batch_size=params.BATCH_SIZE,
                learning_rate=params.LEARNING_RATE,
                save_dir=paths.MODELS_DIR,
                finetune=params.FINETUNE,
                finetune_layers=params.FINETUNE_LAYERS,
                finetune_epochs=params.FINETUNE_EPOCHS,
                finetune_lr_factor=params.FINETUNE_LR_FACTOR):
    """Main training function with improved training strategy.

    Args:
        data_dir: Directory containing train/val/test splits.
        model_name: Model architecture to use.
        num_epochs: Number of training epochs for initial training.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        save_dir: Directory to save model checkpoints.
        finetune: Whether to run a second fine-tuning stage.
        finetune_layers: Backbone layers to unfreeze for fine-tuning.
        finetune_epochs: Number of fine-tuning epochs.
        finetune_lr_factor: Factor to reduce learning rate during fine-tuning.
    """
    # Create directory to save model checkpoints.
    save_dir.mkdir(exist_ok=True)
    
    train_dir = data_dir / 'train'
    train_dataset = datasets.ImageFolder(train_dir)
    num_classes = len(train_dataset.classes)

    # Save class-to-index mapping for later use in inference.
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    class_map_path = save_dir / 'idx_to_class.json'
    with open(class_map_path, 'w') as f:
        json.dump(idx_to_class, f, indent=4)
    print(f"Class mapping saved to {class_map_path}")

    # Get data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Set transforms for datasets
    train_dataset.transform = train_transform

    # Load validation dataset without augmentation.
    val_dataset = datasets.ImageFolder(
        data_dir / 'val',
        transform=val_transform
    )
    # Create data loaders for batch processing with larger batch size.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    # Don't shuffle validation data for consistent evaluation.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    # Initialize model with pretrained weights and frozen early layers.
    classifier = AnimalClassifier(num_classes, model_name=model_name, freeze_layers=True)
    # Use label smoothing with reduced smoothing factor.
    criterion = LabelSmoothingCrossEntropy(smoothing=params.LABEL_SMOOTHING)
    # Use AdamW optimizer with lower learning rate and weight decay.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.model.parameters()),
        lr=learning_rate,
        weight_decay=params.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    # OneCycleLR scheduler for better convergence and accuracy.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    # Initialize training history tracking.
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0
    # Training loop for specified number of epochs with early stopping.
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    patience = params.PATIENCE
    patience_counter = 0
    min_delta = params.MIN_DELTA
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # Train for one epoch with scheduler stepping per batch.
        train_loss, train_acc = classifier.train_epoch(train_loader, criterion, optimizer, scheduler)
        # Validate on validation set.
        val_loss, val_acc = classifier.validate(val_loader, criterion)
        # Record metrics for this epoch.
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        # Calculate overfitting gap.
        gap = train_acc - val_acc
        print(f"Train-Val Gap: {gap:.2f}%")
        # Save model if it achieves best validation accuracy.
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = save_dir / 'best_model.pth'
            classifier.save_checkpoint(
                best_model_path, epoch, optimizer, train_acc, val_acc
            )
            print(f"New best model! Validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
        # Early stopping if validation accuracy doesn't improve.
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
        # Save periodic checkpoints for recovery.
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            classifier.save_checkpoint(
                checkpoint_path, epoch, optimizer, train_acc, val_acc
            )
    # Save final model after initial training completes.
    final_model_path = save_dir / 'final_model.pth'
    classifier.save_checkpoint(
        final_model_path, num_epochs-1, optimizer,
        history['train_acc'][-1], history['val_acc'][-1]
    )
    print(f"Final model saved to {final_model_path}")
    # Optionally run second-stage fine-tuning by unfreezing more layers.
    if finetune:
        print("\nStarting fine-tuning stage with additional layers unfrozen...")
        for name, param in classifier.model.named_parameters():
            for layer_name in finetune_layers:
                if layer_name in name:
                    param.requires_grad = True
        finetune_lr = learning_rate * finetune_lr_factor
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.model.parameters()),
            lr=finetune_lr,
            weight_decay=0.02,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=finetune_lr * 10,
            epochs=finetune_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        patience_counter = 0
        print(f"Fine-tuning for up to {finetune_epochs} epochs with lr={finetune_lr:.6f}...")
        for ft_epoch in range(finetune_epochs):
            print(f"\nFine-tune Epoch {ft_epoch+1}/{finetune_epochs}")
            train_loss, train_acc = classifier.train_epoch(train_loader, criterion, optimizer, scheduler)
            val_loss, val_acc = classifier.validate(val_loader, criterion)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(f"Fine-tune Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Fine-tune Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            gap = train_acc - val_acc
            print(f"Fine-tune Train-Val Gap: {gap:.2f}%")
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_path = save_dir / 'best_model_finetuned.pth'
                classifier.save_checkpoint(
                    best_model_path,
                    ft_epoch,
                    optimizer,
                    train_acc,
                    val_acc
                )
                print(f"New best fine-tuned model! Validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"No fine-tune improvement for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f"\nFine-tuning early stopped after {patience} epochs without improvement.")
                print(f"Best validation accuracy after fine-tuning: {best_val_acc:.2f}%")
                break
    # Save training history as JSON for later analysis.
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    # Generate and save training visualization plots.
    plot_training_history(history, save_dir / 'training_history.png')
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {save_dir}")

if __name__ == "__main__":
    train_model()
