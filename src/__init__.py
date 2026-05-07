"""Init file for the transfer learning framework."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__description__ = "A comprehensive transfer learning framework for computer vision tasks"

from .utils.core import (
    setup_logging,
    get_device,
    set_seed,
    count_parameters,
    format_number,
    save_checkpoint,
    load_checkpoint,
    create_experiment_dir,
    get_model_size_mb,
    print_model_summary,
    EarlyStopping
)

from .data.datasets import (
    CIFAR10Dataset,
    CustomDataset,
    create_data_loaders,
    get_dataset_info,
    get_cifar10_transforms
)

from .models.architectures import (
    SimpleCNN,
    TransferLearningModel,
    EnsembleModel,
    create_model,
    get_available_models
)

from .train.trainer import (
    Trainer,
    ProgressiveUnfreezing,
    create_optimizer,
    create_scheduler
)

from .metrics.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_classification_report,
    plot_training_history,
    plot_calibration_curve,
    plot_roc_curve,
    create_metrics_table,
    evaluate_model
)

__all__ = [
    # Core utilities
    "setup_logging",
    "get_device", 
    "set_seed",
    "count_parameters",
    "format_number",
    "save_checkpoint",
    "load_checkpoint",
    "create_experiment_dir",
    "get_model_size_mb",
    "print_model_summary",
    "EarlyStopping",
    
    # Data utilities
    "CIFAR10Dataset",
    "CustomDataset",
    "create_data_loaders",
    "get_dataset_info",
    "get_cifar10_transforms",
    
    # Model architectures
    "SimpleCNN",
    "TransferLearningModel", 
    "EnsembleModel",
    "create_model",
    "get_available_models",
    
    # Training utilities
    "Trainer",
    "ProgressiveUnfreezing",
    "create_optimizer",
    "create_scheduler",
    
    # Evaluation utilities
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_classification_report", 
    "plot_training_history",
    "plot_calibration_curve",
    "plot_roc_curve",
    "create_metrics_table",
    "evaluate_model"
]
