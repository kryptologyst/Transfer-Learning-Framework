#!/usr/bin/env python3
"""Simple demo script for the Transfer Learning Framework."""

import sys
import os
sys.path.append('src')

from src.utils.core import get_device, set_seed
from src.data.datasets import create_data_loaders, get_dataset_info
from src.models.architectures import create_model
from src.train.trainer import Trainer, create_optimizer

def main():
    """Run a simple transfer learning demo."""
    
    print("🚀 Transfer Learning Framework Demo")
    print("=" * 50)
    
    # Configuration
    config = {
        'dataset_name': 'cifar10',
        'model_name': 'transfer_resnet50',
        'batch_size': 32,
        'num_epochs': 3,  # Quick demo
        'learning_rate': 0.001,
        'seed': 42
    }
    
    # Setup
    set_seed(config['seed'])
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # Get dataset info
    dataset_info = get_dataset_info(config['dataset_name'])
    print(f"📊 Dataset: {dataset_info['name']}")
    print(f"📈 Classes: {dataset_info['num_classes']}")
    
    # Create data loaders
    print("📥 Creating data loaders...")
    data_loaders = create_data_loaders(
        dataset_name=config['dataset_name'],
        batch_size=config['batch_size'],
        num_workers=0,  # Use 0 for compatibility
        val_split=0.2,
        input_size=224,
        augmentation_strength='medium',
        use_albumentations=False,
        seed=config['seed']
    )
    
    print(f"✅ Train batches: {len(data_loaders['train'])}")
    print(f"✅ Val batches: {len(data_loaders['val'])}")
    print(f"✅ Test batches: {len(data_loaders['test'])}")
    
    # Create model
    print(f"🤖 Creating {config['model_name']}...")
    model = create_model(
        model_name=config['model_name'],
        num_classes=dataset_info['num_classes'],
        pretrained=True,
        freeze_backbone=False
    )
    
    # Show model info
    from src.utils.core import count_parameters
    param_counts = count_parameters(model)
    print(f"📊 Total parameters: {param_counts['total']:,}")
    print(f"🔧 Trainable parameters: {param_counts['trainable']:,}")
    
    # Create optimizer and trainer
    optimizer = create_optimizer(
        model=model,
        optimizer_name='adam',
        lr=config['learning_rate']
    )
    
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        device=device,
        optimizer=optimizer
    )
    
    # Train model
    print(f"🏋️‍♂️ Training for {config['num_epochs']} epochs...")
    history = trainer.train(config['num_epochs'])
    
    # Evaluate
    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(data_loaders['test'])
    
    # Display results
    print("\n" + "=" * 50)
    print("📈 RESULTS")
    print("=" * 50)
    print(f"✅ Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"✅ Test Precision (Macro): {test_results['precision_macro']:.4f}")
    print(f"✅ Test Recall (Macro): {test_results['recall_macro']:.4f}")
    print(f"✅ Test F1 Score (Macro): {test_results['f1_macro']:.4f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n⚠️ Remember: This framework is for research and educational purposes only.")
    print("Not intended for production decisions or control systems.")

if __name__ == "__main__":
    main()
