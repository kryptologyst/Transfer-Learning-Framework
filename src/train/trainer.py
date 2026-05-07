"""Training utilities and loss functions."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from ..utils.core import EarlyStopping, save_checkpoint, load_checkpoint
from ..metrics.evaluation import calculate_metrics

logger = logging.getLogger("transfer_learning")


class Trainer:
    """Main trainer class for transfer learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        experiment_dir: Optional[str] = None,
        use_mixed_precision: bool = False
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            early_stopping: Early stopping utility
            experiment_dir: Directory to save checkpoints
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optimizer or optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = scheduler
        
        # Early stopping
        self.early_stopping = early_stopping
        
        # Experiment directory
        self.experiment_dir = experiment_dir
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = 10) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                if self.experiment_dir:
                    checkpoint_path = f"{self.experiment_dir}/checkpoints/best_model.pth"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        {"val_acc": val_acc, "train_acc": train_acc},
                        checkpoint_path
                    )
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        return self.history
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader to evaluate on
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Get predictions
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            self.model.num_classes if hasattr(self.model, "num_classes") else 10
        )
        
        metrics["loss"] = total_loss / len(data_loader)
        
        return metrics


class ProgressiveUnfreezing:
    """Progressive unfreezing strategy for transfer learning."""
    
    def __init__(
        self,
        model: nn.Module,
        unfreeze_schedule: List[int],
        optimizer: optim.Optimizer,
        lr_multiplier: float = 0.1
    ):
        """Initialize progressive unfreezing.
        
        Args:
            model: Model to unfreeze progressively
            unfreeze_schedule: List of epochs to unfreeze layers
            optimizer: Optimizer to update
            lr_multiplier: Learning rate multiplier for unfrozen layers
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.optimizer = optimizer
        self.lr_multiplier = lr_multiplier
        
        # Get layer groups
        self.layer_groups = self._get_layer_groups()
        
        # Initialize with all layers frozen except classifier
        self._freeze_all_except_classifier()
    
    def _get_layer_groups(self) -> List[List[nn.Parameter]]:
        """Get layer groups for progressive unfreezing."""
        groups = []
        
        if hasattr(self.model, "backbone"):
            # Transfer learning model
            backbone = self.model.backbone
            
            if hasattr(backbone, "layer4"):
                # ResNet
                groups = [
                    list(backbone.layer4.parameters()),
                    list(backbone.layer3.parameters()),
                    list(backbone.layer2.parameters()),
                    list(backbone.layer1.parameters()),
                    list(backbone.conv1.parameters()) + list(backbone.bn1.parameters())
                ]
            elif hasattr(backbone, "features"):
                # VGG
                features = backbone.features
                groups = [
                    list(features[24:].parameters()),  # Last conv block
                    list(features[17:24].parameters()),  # Second to last
                    list(features[10:17].parameters()),  # Third block
                    list(features[3:10].parameters()),   # Second block
                    list(features[:3].parameters())      # First block
                ]
        
        return groups
    
    def _freeze_all_except_classifier(self):
        """Freeze all layers except classifier."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def step(self, epoch: int):
        """Update unfreezing based on current epoch.
        
        Args:
            epoch: Current epoch
        """
        if epoch in self.unfreeze_schedule:
            group_idx = self.unfreeze_schedule.index(epoch)
            
            if group_idx < len(self.layer_groups):
                # Unfreeze this group
                for param in self.layer_groups[group_idx]:
                    param.requires_grad = True
                
                # Update optimizer with new parameters
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                
                # Create new optimizer with different learning rates
                base_lr = self.optimizer.param_groups[0]["lr"]
                
                # Reset optimizer
                self.optimizer = optim.Adam([
                    {"params": self.model.classifier.parameters(), "lr": base_lr},
                    {"params": trainable_params, "lr": base_lr * self.lr_multiplier}
                ])
                
                logger.info(f"Unfroze layer group {group_idx} at epoch {epoch}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == "adam":
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "step",
    **kwargs
) -> Any:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
