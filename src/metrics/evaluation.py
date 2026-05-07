"""Evaluation metrics and utilities."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    top_k_accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("transfer_learning")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: Optional class names
        y_prob: Optional predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"class_{i}"
        metrics[f"precision_{class_name}"] = precision[i]
        metrics[f"recall_{class_name}"] = recall[i]
        metrics[f"f1_{class_name}"] = f1[i]
        metrics[f"support_{class_name}"] = support[i]
    
    # Top-k accuracy
    if y_prob is not None:
        for k in [2, 3, 5]:
            if k <= num_classes:
                try:
                    top_k_acc = top_k_accuracy_score(y_true, y_prob, k=k)
                    metrics[f"top_{k}_accuracy"] = top_k_acc
                except Exception as e:
                    logger.warning(f"Could not calculate top-{k} accuracy: {e}")
    
    # AUC metrics (for binary and multiclass)
    if y_prob is not None:
        if num_classes == 2:
            # Binary classification
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                metrics["average_precision"] = average_precision_score(y_true, y_prob[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        else:
            # Multiclass classification
            try:
                metrics["auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                metrics["auc_weighted"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            except Exception as e:
                logger.warning(f"Could not calculate multiclass AUC: {e}")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title("Confusion Matrix (Normalized)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot classification report as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract metrics for heatmap
    metrics = ["precision", "recall", "f1-score"]
    classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    
    data = np.zeros((len(metrics), len(classes)))
    
    for i, metric in enumerate(metrics):
        for j, class_name in enumerate(classes):
            data[i, j] = report[class_name][metric]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        xticklabels=classes,
        yticklabels=metrics,
        ax=ax
    )
    
    ax.set_title("Classification Report")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Metrics")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate plot
    if "learning_rate" in history:
        axes[2].plot(epochs, history["learning_rate"], "g-")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_yscale("log")
        axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    
    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Optional class names
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    else:
        # Multiclass classification
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            class_name = class_names[i] if class_names else f"Class {i}"
            ax.plot(fpr[i], tpr[i], lw=2, label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
    
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_metrics_table(
    metrics: Dict[str, float],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> str:
    """Create a formatted metrics table.
    
    Args:
        metrics: Dictionary of metrics
        class_names: Optional class names
        save_path: Optional path to save table
        
    Returns:
        Formatted table string
    """
    import pandas as pd
    
    # Separate different types of metrics
    basic_metrics = {}
    per_class_metrics = {}
    
    for key, value in metrics.items():
        if any(suffix in key for suffix in ["precision_", "recall_", "f1_", "support_"]):
            per_class_metrics[key] = value
        else:
            basic_metrics[key] = value
    
    # Create basic metrics table
    basic_df = pd.DataFrame(list(basic_metrics.items()), columns=["Metric", "Value"])
    basic_df["Value"] = basic_df["Value"].round(4)
    
    # Create per-class metrics table
    if per_class_metrics:
        per_class_data = []
        for key, value in per_class_metrics.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                metric_type, class_name = parts
                per_class_data.append({
                    "Class": class_name,
                    "Metric": metric_type,
                    "Value": round(value, 4)
                })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_df = per_class_df.pivot(index="Class", columns="Metric", values="Value")
    
    # Format output
    output = "=== BASIC METRICS ===\n"
    output += basic_df.to_string(index=False)
    
    if per_class_metrics:
        output += "\n\n=== PER-CLASS METRICS ===\n"
        output += per_class_df.to_string()
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(output)
    
    return output


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    return_probabilities: bool = False
) -> Dict[str, Any]:
    """Evaluate model and return comprehensive results.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run on
        class_names: Optional class names
        return_probabilities: Whether to return probabilities
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Calculate metrics
    num_classes = len(class_names) if class_names else len(np.unique(y_true))
    metrics = calculate_metrics(y_true, y_pred, num_classes, class_names, y_prob)
    
    results = {
        "metrics": metrics,
        "predictions": y_pred,
        "targets": y_true
    }
    
    if return_probabilities:
        results["probabilities"] = y_prob
    
    return results
