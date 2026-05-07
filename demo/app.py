"""Streamlit demo app for Transfer Learning Framework."""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import io

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import get_device, set_seed, count_parameters, format_number
from src.data.datasets import create_data_loaders, get_dataset_info, get_cifar10_transforms
from src.models.architectures import create_model, get_available_models
from src.train.trainer import Trainer, create_optimizer, create_scheduler
from src.metrics.evaluation import calculate_metrics, plot_confusion_matrix, plot_training_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transfer_learning")

# Page config
st.set_page_config(
    page_title="Transfer Learning Framework",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "device" not in st.session_state:
    st.session_state.device = get_device()
if "seed" not in st.session_state:
    st.session_state.seed = 42

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Transfer Learning Framework</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Educational Use Only</h4>
    <p>This framework is designed for research and educational purposes. 
    <strong>Not intended for production decisions or control systems.</strong></p>
    <p>Always validate results with domain experts and consider ethical implications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        st.subheader("📊 Dataset")
        dataset_name = st.selectbox(
            "Select Dataset",
            ["cifar10"],
            help="Choose the dataset for transfer learning experiments"
        )
        
        # Model selection
        st.subheader("🤖 Model")
        available_models = get_available_models()
        
        model_category = st.selectbox(
            "Model Category",
            list(available_models.keys()),
            help="Choose between baseline and transfer learning models"
        )
        
        model_name = st.selectbox(
            "Model Architecture",
            available_models[model_category],
            help="Select specific model architecture"
        )
        
        # Training parameters
        st.subheader("🎯 Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.slider("Epochs", 1, 50, 10)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with col2:
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            freeze_backbone = st.checkbox("Freeze Backbone", value=False)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            optimizer_name = st.selectbox("Optimizer", ["adam", "adamw", "sgd", "rmsprop"])
            scheduler_name = st.selectbox("Scheduler", ["none", "step", "cosine", "plateau"])
            augmentation_strength = st.selectbox("Augmentation", ["light", "medium", "strong"])
            use_mixed_precision = st.checkbox("Mixed Precision", value=False)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.5)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🏃‍♂️ Quick Start", "📈 Training", "📊 Evaluation", "🔍 Analysis"])
    
    with tab1:
        quick_start_tab(dataset_name, model_name, freeze_backbone)
    
    with tab2:
        training_tab(
            dataset_name, model_name, num_epochs, batch_size, learning_rate,
            freeze_backbone, optimizer_name, scheduler_name, augmentation_strength,
            use_mixed_precision, dropout_rate
        )
    
    with tab3:
        evaluation_tab(dataset_name, model_name)
    
    with tab4:
        analysis_tab()


def quick_start_tab(dataset_name: str, model_name: str, freeze_backbone: bool):
    """Quick start tab with basic functionality."""
    
    st.header("🚀 Quick Start Transfer Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Overview")
        
        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        st.write(f"**Dataset:** {dataset_info['name']}")
        st.write(f"**Classes:** {dataset_info['num_classes']}")
        st.write(f"**Samples:** {dataset_info['num_train_samples']:,} train, {dataset_info['num_test_samples']:,} test")
        st.write(f"**Description:** {dataset_info['description']}")
        
        # Display class names
        st.subheader("🏷️ Class Labels")
        cols = st.columns(3)
        for i, class_name in enumerate(dataset_info["class_names"]):
            with cols[i % 3]:
                st.write(f"• {class_name}")
    
    with col2:
        st.subheader("🤖 Model Info")
        
        # Create model to get info
        try:
            model = create_model(
                model_name=model_name,
                num_classes=dataset_info["num_classes"],
                pretrained=True,
                freeze_backbone=freeze_backbone
            )
            
            param_counts = count_parameters(model)
            
            st.metric("Total Parameters", format_number(param_counts["total"]))
            st.metric("Trainable Parameters", format_number(param_counts["trainable"]))
            st.metric("Frozen Parameters", format_number(param_counts["frozen"]))
            
            # Model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            st.metric("Model Size (MB)", f"{model_size:.1f}")
            
        except Exception as e:
            st.error(f"Error creating model: {e}")
    
    # Sample images
    st.subheader("🖼️ Sample Images")
    
    if st.button("Load Sample Images", key="load_samples"):
        try:
            # Create data loaders
            data_loaders = create_data_loaders(
                dataset_name=dataset_name,
                batch_size=8,
                num_workers=0,
                val_split=0.0,
                input_size=224,
                augmentation_strength="light",
                use_albumentations=False,
                seed=st.session_state.seed
            )
            
            # Get a batch
            data_iter = iter(data_loaders["test"])
            images, labels = next(data_iter)
            
            # Display images
            cols = st.columns(4)
            for i in range(min(8, len(images))):
                with cols[i % 4]:
                    # Convert tensor to PIL
                    img = images[i].permute(1, 2, 0).numpy()
                    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    img = np.clip(img, 0, 1)
                    
                    st.image(img, caption=f"Class: {dataset_info['class_names'][labels[i]]}")
        
        except Exception as e:
            st.error(f"Error loading images: {e}")


def training_tab(
    dataset_name: str, model_name: str, num_epochs: int, batch_size: int,
    learning_rate: float, freeze_backbone: bool, optimizer_name: str,
    scheduler_name: str, augmentation_strength: str, use_mixed_precision: bool,
    dropout_rate: float
):
    """Training tab with interactive training."""
    
    st.header("🏋️‍♂️ Interactive Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Training Configuration")
        
        config_display = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Epochs": num_epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Freeze Backbone": freeze_backbone,
            "Optimizer": optimizer_name,
            "Scheduler": scheduler_name,
            "Augmentation": augmentation_strength,
            "Mixed Precision": use_mixed_precision,
            "Dropout Rate": dropout_rate
        }
        
        for key, value in config_display.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("📊 Training Status")
        
        if "training_progress" not in st.session_state:
            st.session_state.training_progress = {
                "running": False,
                "epoch": 0,
                "total_epochs": 0,
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": 0.0,
                "val_acc": 0.0
            }
    
    # Training controls
    st.subheader("🎮 Training Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Training", disabled=st.session_state.training_progress["running"]):
            start_training(
                dataset_name, model_name, num_epochs, batch_size, learning_rate,
                freeze_backbone, optimizer_name, scheduler_name, augmentation_strength,
                use_mixed_precision, dropout_rate
            )
    
    with col2:
        if st.button("⏹️ Stop Training", disabled=not st.session_state.training_progress["running"]):
            st.session_state.training_progress["running"] = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.training_progress = {
                "running": False,
                "epoch": 0,
                "total_epochs": 0,
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": 0.0,
                "val_acc": 0.0
            }
            st.rerun()
    
    # Training progress
    if st.session_state.training_progress["running"]:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        epoch = st.session_state.training_progress["epoch"]
        total_epochs = st.session_state.training_progress["total_epochs"]
        
        if total_epochs > 0:
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{total_epochs}")
    
    # Training metrics
    if "training_history" in st.session_state and st.session_state.training_history:
        st.subheader("📈 Training Metrics")
        
        history = st.session_state.training_history
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            "Epoch": range(1, len(history["train_loss"]) + 1),
            "Train Loss": history["train_loss"],
            "Train Accuracy": history["train_acc"],
            "Val Loss": history["val_loss"],
            "Val Accuracy": history["val_acc"]
        })
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(metrics_df.set_index("Epoch")[["Train Loss", "Val Loss"]])
        
        with col2:
            st.line_chart(metrics_df.set_index("Epoch")[["Train Accuracy", "Val Accuracy"]])


def start_training(
    dataset_name: str, model_name: str, num_epochs: int, batch_size: int,
    learning_rate: float, freeze_backbone: bool, optimizer_name: str,
    scheduler_name: str, augmentation_strength: str, use_mixed_precision: bool,
    dropout_rate: float
):
    """Start training process."""
    
    try:
        # Set seed
        set_seed(st.session_state.seed)
        
        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for Streamlit
            val_split=0.2,
            input_size=224,
            augmentation_strength=augmentation_strength,
            use_albumentations=False,
            seed=st.session_state.seed
        )
        
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=dataset_info["num_classes"],
            pretrained=True,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
        
        # Create optimizer
        optimizer = create_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            lr=learning_rate
        )
        
        # Create scheduler
        scheduler = None
        if scheduler_name != "none":
            scheduler = create_scheduler(
                optimizer=optimizer,
                scheduler_name=scheduler_name,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=data_loaders["train"],
            val_loader=data_loaders.get("val"),
            test_loader=data_loaders["test"],
            device=st.session_state.device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_mixed_precision=use_mixed_precision
        )
        
        # Update session state
        st.session_state.training_progress["running"] = True
        st.session_state.training_progress["total_epochs"] = num_epochs
        
        # Train model
        history = trainer.train(num_epochs)
        
        # Save results
        st.session_state.training_history = history
        st.session_state.trained_model = model
        st.session_state.training_progress["running"] = False
        
        st.success("✅ Training completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.session_state.training_progress["running"] = False


def evaluation_tab(dataset_name: str, model_name: str):
    """Evaluation tab with model testing."""
    
    st.header("📊 Model Evaluation")
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ No trained model available. Please train a model first.")
        return
    
    st.subheader("🎯 Evaluation Results")
    
    try:
        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        # Create test data loader
        data_loaders = create_data_loaders(
            dataset_name=dataset_name,
            batch_size=32,
            num_workers=0,
            val_split=0.0,
            input_size=224,
            augmentation_strength="light",
            use_albumentations=False,
            seed=st.session_state.seed
        )
        
        # Evaluate model
        model = st.session_state.trained_model
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loaders["test"]:
                data, target = data.to(st.session_state.device), target.to(st.session_state.device)
                
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        metrics = calculate_metrics(
            y_true, y_pred, dataset_info["num_classes"],
            dataset_info["class_names"], y_prob
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
        with col3:
            st.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
        with col4:
            st.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.4f}")
        
        # Confusion matrix
        st.subheader("🔍 Confusion Matrix")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = np.array([[0] * dataset_info["num_classes"] for _ in range(dataset_info["num_classes"])])
        
        for i in range(len(y_true)):
            cm[y_true[i]][y_pred[i]] += 1
        
        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=dataset_info["class_names"],
            yticklabels=dataset_info["class_names"],
            ax=ax
        )
        
        ax.set_title("Confusion Matrix (Normalized)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        st.pyplot(fig)
        
        # Per-class metrics
        st.subheader("📈 Per-Class Metrics")
        
        per_class_data = []
        for i, class_name in enumerate(dataset_info["class_names"]):
            per_class_data.append({
                "Class": class_name,
                "Precision": metrics.get(f"precision_{class_name}", 0),
                "Recall": metrics.get(f"recall_{class_name}", 0),
                "F1 Score": metrics.get(f"f1_{class_name}", 0),
                "Support": metrics.get(f"support_{class_name}", 0)
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        st.dataframe(per_class_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Evaluation failed: {e}")


def analysis_tab():
    """Analysis tab with model comparison and insights."""
    
    st.header("🔍 Model Analysis")
    
    st.subheader("📊 Model Comparison")
    
    # Placeholder for model comparison
    st.info("Model comparison features will be available after training multiple models.")
    
    st.subheader("🎯 Transfer Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Transfer Learning Benefits:**
        - 🚀 Faster convergence with pretrained features
        - 📈 Better performance with limited data
        - 🔧 Reduced computational requirements
        - 🎯 Domain adaptation capabilities
        """)
    
    with col2:
        st.markdown("""
        **Best Practices:**
        - 🧊 Start with frozen backbone
        - 🔄 Gradually unfreeze layers
        - 📊 Use appropriate learning rates
        - 🎨 Apply domain-specific augmentations
        - 📏 Monitor validation metrics closely
        """)
    
    st.subheader("📚 Resources")
    
    st.markdown("""
    - 📖 [Transfer Learning Paper](https://arxiv.org/abs/1411.1792)
    - 🎓 [Deep Learning Course](https://www.deeplearning.ai/)
    - 🔬 [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
    - 🏗️ [Model Zoo](https://pytorch.org/hub/)
    """)


if __name__ == "__main__":
    main()
