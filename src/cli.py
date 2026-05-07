"""Command-line interface for the Transfer Learning Framework."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .utils.core import setup_logging, get_device, set_seed, create_experiment_dir
from .data.datasets import create_data_loaders, get_dataset_info
from .models.architectures import create_model, get_available_models
from .train.trainer import Trainer, create_optimizer, create_scheduler
from .metrics.evaluation import evaluate_model, create_metrics_table, plot_confusion_matrix

console = Console()
app = typer.Typer(help="Transfer Learning Framework CLI")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


@app.command()
def train(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Experiment name (overrides config)"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of epochs (overrides config)"
    ),
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate (overrides config)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Batch size (overrides config)"
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model name (overrides config)"
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset name (overrides config)"
    ),
    freeze_backbone: Optional[bool] = typer.Option(
        None,
        "--freeze-backbone",
        help="Freeze backbone parameters"
    ),
    mixed_precision: Optional[bool] = typer.Option(
        None,
        "--mixed-precision",
        help="Use mixed precision training"
    )
):
    """Train a transfer learning model."""
    
    # Load configuration
    config = load_config(config)
    
    # Override config with command line arguments
    if experiment_name:
        config["experiment"]["name"] = experiment_name
    if epochs:
        config["training"]["num_epochs"] = epochs
    if lr:
        config["training"]["lr"] = lr
    if batch_size:
        config["dataset"]["batch_size"] = batch_size
    if model_name:
        config["model"]["name"] = model_name
    if dataset_name:
        config["dataset"]["name"] = dataset_name
    if freeze_backbone is not None:
        config["model"]["freeze_backbone"] = freeze_backbone
    if mixed_precision is not None:
        config["training"]["use_mixed_precision"] = mixed_precision
    
    # Setup logging
    logger = setup_logging(config["experiment"]["log_level"])
    
    # Set seed
    set_seed(config["experiment"]["seed"])
    
    # Get device
    device = get_device()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(
        config["experiment"]["base_dir"],
        config["experiment"]["name"]
    )
    
    # Save configuration
    save_config(config, os.path.join(experiment_dir, "config.yaml"))
    
    # Display configuration
    console.print(Panel.fit(
        f"[bold blue]Starting Training[/bold blue]\n"
        f"Model: {config['model']['name']}\n"
        f"Dataset: {config['dataset']['name']}\n"
        f"Epochs: {config['training']['num_epochs']}\n"
        f"Batch Size: {config['dataset']['batch_size']}\n"
        f"Learning Rate: {config['training']['lr']}\n"
        f"Device: {device}\n"
        f"Experiment Dir: {experiment_dir}",
        title="Configuration"
    ))
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        val_split=config["dataset"]["val_split"],
        input_size=config["dataset"]["input_size"],
        augmentation_strength=config["dataset"]["augmentation_strength"],
        use_albumentations=config["dataset"]["use_albumentations"],
        seed=config["experiment"]["seed"]
    )
    
    # Get dataset info
    dataset_info = get_dataset_info(config["dataset"]["name"])
    
    # Create model
    logger.info(f"Creating model: {config['model']['name']}")
    model = create_model(
        model_name=config["model"]["name"],
        num_classes=dataset_info["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        dropout_rate=config["model"]["dropout_rate"],
        use_custom_head=config["model"]["use_custom_head"]
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create scheduler
    scheduler = None
    if config["training"]["scheduler"]:
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name=config["training"]["scheduler"],
            **config["training"]["scheduler_params"]
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders.get("val"),
        test_loader=data_loaders["test"],
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_dir=experiment_dir,
        use_mixed_precision=config["training"]["use_mixed_precision"]
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(config["training"]["num_epochs"])
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(data_loaders["test"])
    
    # Save results
    results_path = os.path.join(experiment_dir, "results", "test_results.yaml")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        yaml.dump(test_results, f, default_flow_style=False, indent=2)
    
    # Create plots if requested
    if config["evaluation"]["create_plots"]:
        logger.info("Creating evaluation plots...")
        
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Training history plot
        if config["evaluation"]["plot_training_history"]:
            from .metrics.evaluation import plot_training_history
            plot_training_history(
                history,
                save_path=os.path.join(plots_dir, "training_history.png")
            )
        
        # Confusion matrix
        if config["evaluation"]["plot_confusion_matrix"]:
            plot_confusion_matrix(
                test_results["targets"],
                test_results["predictions"],
                class_names=dataset_info["class_names"],
                save_path=os.path.join(plots_dir, "confusion_matrix.png")
            )
    
    # Display final results
    console.print(Panel.fit(
        f"[bold green]Training Completed![/bold green]\n"
        f"Test Accuracy: {test_results['accuracy']:.4f}\n"
        f"Test F1 (Macro): {test_results['f1_macro']:.4f}\n"
        f"Best Val Accuracy: {trainer.best_val_acc:.2f}%\n"
        f"Results saved to: {experiment_dir}",
        title="Results"
    ))


@app.command()
def evaluate(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset name (overrides config)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Batch size (overrides config)"
    )
):
    """Evaluate a trained model."""
    
    # Load configuration
    config = load_config(config)
    
    # Override config
    if dataset_name:
        config["dataset"]["name"] = dataset_name
    if batch_size:
        config["dataset"]["batch_size"] = batch_size
    
    # Setup logging
    logger = setup_logging(config["experiment"]["log_level"])
    
    # Get device
    device = get_device()
    
    # Create data loaders
    data_loaders = create_data_loaders(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        val_split=0.0,  # No validation split for evaluation
        input_size=config["dataset"]["input_size"],
        augmentation_strength="light",  # Minimal augmentation for evaluation
        use_albumentations=False,
        seed=config["experiment"]["seed"]
    )
    
    # Get dataset info
    dataset_info = get_dataset_info(config["dataset"]["name"])
    
    # Create model
    model = create_model(
        model_name=config["model"]["name"],
        num_classes=dataset_info["num_classes"],
        pretrained=False  # Don't load pretrained weights
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(
        model=model,
        data_loader=data_loaders["test"],
        device=device,
        class_names=dataset_info["class_names"],
        return_probabilities=True
    )
    
    # Display results
    metrics_table = create_metrics_table(
        results["metrics"],
        class_names=dataset_info["class_names"]
    )
    
    console.print(Panel(metrics_table, title="Evaluation Results"))


@app.command()
def list_models():
    """List available models."""
    
    models = get_available_models()
    
    table = Table(title="Available Models")
    table.add_column("Category", style="cyan")
    table.add_column("Models", style="green")
    
    for category, model_list in models.items():
        table.add_row(category, ", ".join(model_list))
    
    console.print(table)


@app.command()
def list_datasets():
    """List available datasets."""
    
    datasets = ["cifar10"]
    
    table = Table(title="Available Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Description", style="green")
    
    for dataset in datasets:
        info = get_dataset_info(dataset)
        table.add_row(dataset, info["description"])
    
    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
