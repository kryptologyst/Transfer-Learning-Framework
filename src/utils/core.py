"""Core utilities for the Transfer Learning Framework."""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from rich.console import Console
from rich.logging import RichHandler

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging with rich console output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("transfer_learning")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
    
    return logger


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]Using CUDA device: {torch.cuda.get_device_name()}[/green]")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]Using MPS device (Apple Silicon)[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]Using CPU device[/yellow]")
    
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    console.print(f"[blue]Random seed set to: {seed}[/blue]")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string with suffix (K, M, B)
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    **kwargs: Any
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        **kwargs
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    console.print(f"[green]Checkpoint saved to: {filepath}[/green]")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        filepath: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    console.print(f"[green]Checkpoint loaded from: {filepath}[/green]")
    return checkpoint


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    
    console.print(f"[blue]Experiment directory created: {exp_dir}[/blue]")
    return exp_dir


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for calculating FLOPs
    """
    param_counts = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    console.print("\n[bold blue]Model Summary[/bold blue]")
    console.print(f"Total parameters: {format_number(param_counts['total'])}")
    console.print(f"Trainable parameters: {format_number(param_counts['trainable'])}")
    console.print(f"Frozen parameters: {format_number(param_counts['frozen'])}")
    console.print(f"Model size: {model_size:.2f} MB")
    
    # Print model architecture
    console.print("\n[bold blue]Model Architecture[/bold blue]")
    console.print(str(model))


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min"
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score to monitor
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
