# Transfer Learning Framework

A comprehensive transfer learning framework for computer vision tasks, designed for research and educational purposes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Important Disclaimer

**This framework is designed for research and educational purposes only.**

- **NOT intended for production decisions or control systems**
- **Research and educational use only**
- **Always validate results with domain experts**
- **Consider ethical implications before deployment**

## Features

### **Modern Architecture**
- Clean, modular codebase with type hints
- Comprehensive error handling and logging
- Device-agnostic (CUDA, MPS, CPU) with automatic fallback
- Deterministic seeding for reproducibility

### **Model Support**
- **Baseline Models**: Simple CNN for comparison
- **Transfer Learning**: ResNet, VGG, DenseNet, EfficientNet, Vision Transformers
- **Advanced Features**: Progressive unfreezing, ensemble methods
- **Custom Architectures**: Easy to extend and modify

### **Comprehensive Evaluation**
- Multiple metrics: Accuracy, Precision, Recall, F1, AUC
- Top-k accuracy for ranking tasks
- Confusion matrices and classification reports
- Training history visualization
- ROC curves and calibration analysis

### **Interactive Demo**
- Streamlit-based web interface
- Real-time training visualization
- Interactive model comparison
- Sample image exploration

### **Developer Experience**
- CLI interface with rich console output
- YAML configuration management
- Comprehensive logging with Rich
- Pre-commit hooks and code formatting

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kryptologyst/Transfer-Learning-Framework.git
cd Transfer-Learning-Framework
```

2. **Install dependencies:**
```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With advanced features
pip install -e ".[advanced]"
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Basic Usage

#### Command Line Interface

```bash
# Train a ResNet50 model on CIFAR-10
python -m src.cli train --model transfer_resnet50 --epochs 20

# Evaluate a trained model
python -m src.cli evaluate checkpoints/best_model.pth

# List available models
python -m src.cli list-models

# List available datasets
python -m src.cli list-datasets
```

#### Python API

```python
from src.utils.core import get_device, set_seed
from src.data.datasets import create_data_loaders
from src.models.architectures import create_model
from src.train.trainer import Trainer, create_optimizer

# Set up
set_seed(42)
device = get_device()

# Create data loaders
data_loaders = create_data_loaders(
    dataset_name="cifar10",
    batch_size=32,
    val_split=0.2
)

# Create model
model = create_model(
    model_name="transfer_resnet50",
    num_classes=10,
    pretrained=True,
    freeze_backbone=False
)

# Create trainer
optimizer = create_optimizer(model, "adam", lr=0.001)
trainer = Trainer(
    model=model,
    train_loader=data_loaders["train"],
    val_loader=data_loaders["val"],
    device=device,
    optimizer=optimizer
)

# Train
history = trainer.train(num_epochs=20)

# Evaluate
test_results = trainer.evaluate(data_loaders["test"])
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
```

#### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## 📁 Project Structure

```
transfer-learning-framework/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   └── datasets.py          # Dataset classes and utilities
│   ├── models/                   # Model architectures
│   │   └── architectures.py     # Model definitions
│   ├── train/                    # Training utilities
│   │   └── trainer.py           # Training loop and utilities
│   ├── metrics/                  # Evaluation metrics
│   │   └── evaluation.py        # Metrics and visualization
│   ├── utils/                    # Utility functions
│   │   └── core.py              # Core utilities
│   └── cli.py                   # Command-line interface
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit application
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks
├── assets/                       # Generated assets
├── data/                         # Data storage
├── experiments/                  # Experiment outputs
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Configuration

The framework uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
dataset:
  name: "cifar10"
  batch_size: 32
  augmentation_strength: "medium"

model:
  name: "transfer_resnet50"
  pretrained: true
  freeze_backbone: false

training:
  num_epochs: 20
  optimizer: "adam"
  lr: 0.001
  scheduler: "step"
```

## Available Models

### Baseline Models
- **Simple CNN**: Custom CNN architecture for comparison

### Transfer Learning Models
- **ResNet**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **VGG**: VGG16, VGG19
- **DenseNet**: DenseNet121, DenseNet161, DenseNet201
- **EfficientNet**: EfficientNet-B0, EfficientNet-B1, EfficientNet-B2
- **Vision Transformers**: ViT-Base, ViT-Large

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro and weighted averages
- **Top-k Accuracy**: Ranking performance
- **AUC**: Area under ROC curve
- **Average Precision**: For imbalanced datasets

### Visualization
- **Confusion Matrix**: Normalized and raw counts
- **Classification Report**: Per-class metrics heatmap
- **Training History**: Loss and accuracy curves
- **ROC Curves**: Binary and multiclass
- **Calibration Curves**: Model calibration analysis

## Advanced Features

### Progressive Unfreezing
Gradually unfreeze layers during training for better fine-tuning:

```python
from src.train.trainer import ProgressiveUnfreezing

progressive_unfreezing = ProgressiveUnfreezing(
    model=model,
    unfreeze_schedule=[5, 10, 15],
    optimizer=optimizer,
    lr_multiplier=0.1
)
```

### Ensemble Methods
Combine multiple models for improved performance:

```python
from src.models.architectures import EnsembleModel

ensemble = EnsembleModel(
    models=[model1, model2, model3],
    weights=[0.4, 0.3, 0.3],
    fusion_method="weighted"
)
```

### Mixed Precision Training
Accelerate training with automatic mixed precision:

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    device=device,
    use_mixed_precision=True
)
```

## Experiments

### Running Experiments

```bash
# Basic experiment
python -m src.cli train --config configs/default.yaml

# Custom experiment
python -m src.cli train \
    --model transfer_efficientnet_b0 \
    --epochs 30 \
    --lr 0.0005 \
    --batch-size 64 \
    --freeze-backbone
```

### Experiment Tracking

The framework automatically saves:
- Model checkpoints (best and last)
- Training history and metrics
- Configuration files
- Evaluation plots and reports

## Interactive Demo

Launch the Streamlit demo for an interactive experience:

```bash
streamlit run demo/app.py
```

Features:
- **Quick Start**: Model and dataset overview
- **Training**: Interactive training with real-time metrics
- **Evaluation**: Comprehensive model evaluation
- **Analysis**: Transfer learning insights and best practices

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_models.py
```

## Development

### Code Quality

The project uses modern Python development practices:

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding New Models

1. **Extend the model registry:**
```python
# In src/models/architectures.py
def _load_backbone(self, backbone_name: str, pretrained: bool):
    if backbone_name.lower() == "your_model":
        return your_model(pretrained=pretrained)
    # ... existing code
```

2. **Add to available models:**
```python
def get_available_models():
    return {
        "transfer_learning": [
            # ... existing models
            "transfer_your_model"
        ]
    }
```

### Adding New Datasets

1. **Create dataset class:**
```python
class YourDataset(Dataset):
    def __init__(self, ...):
        # Implementation
```

2. **Add to data loader factory:**
```python
def create_data_loaders(dataset_name: str, ...):
    if dataset_name.lower() == "your_dataset":
        # Implementation
```

## Examples

### Transfer Learning on Custom Dataset

```python
from src.data.datasets import CustomDataset
from src.models.architectures import TransferLearningModel

# Create custom dataset
custom_dataset = CustomDataset(
    images=["path/to/image1.jpg", "path/to/image2.jpg"],
    labels=[0, 1],
    transform=transforms
)

# Create model
model = TransferLearningModel(
    backbone_name="resnet50",
    num_classes=2,
    pretrained=True,
    freeze_backbone=True
)

# Train
trainer = Trainer(model=model, ...)
history = trainer.train(num_epochs=10)
```

### Model Comparison

```python
models = ["transfer_resnet50", "transfer_efficientnet_b0", "simple_cnn"]
results = {}

for model_name in models:
    model = create_model(model_name, num_classes=10)
    trainer = Trainer(model=model, ...)
    history = trainer.train(num_epochs=20)
    test_results = trainer.evaluate(test_loader)
    results[model_name] = test_results

# Compare results
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics['accuracy']:.4f}")
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/kryptologyst/Transfer-Learning-Framework.git

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Author**: [kryptologyst](https://github.com/kryptologyst)
- **GitHub**: https://github.com/kryptologyst
- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework
- **Open Source Community**: For inspiration and contributions

## Related Projects

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [TIMM: PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
- [Albumentations](https://github.com/albumentations-team/albumentations)

---

**Remember**: This framework is for research and educational purposes only. Always validate results and consider ethical implications before any real-world application.
# Transfer-Learning-Framework
