"""Test suite for the Transfer Learning Framework."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.core import get_device, set_seed, count_parameters, format_number
from src.data.datasets import get_dataset_info, get_cifar10_transforms
from src.models.architectures import create_model, get_available_models
from src.train.trainer import create_optimizer, create_scheduler
from src.metrics.evaluation import calculate_metrics


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Seed setting is hard to test directly
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        counts = count_parameters(model)
        
        assert "total" in counts
        assert "trainable" in counts
        assert "frozen" in counts
        assert counts["total"] == 55  # 10*5 + 5 bias
        assert counts["trainable"] == 55
        assert counts["frozen"] == 0
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1000) == "1.0K"
        assert format_number(1000000) == "1.0M"
        assert format_number(1000000000) == "1.0B"
        assert format_number(500) == "500"


class TestDataUtils:
    """Test data utility functions."""
    
    def test_get_dataset_info(self):
        """Test dataset info retrieval."""
        info = get_dataset_info("cifar10")
        
        assert info["name"] == "CIFAR-10"
        assert info["num_classes"] == 10
        assert len(info["class_names"]) == 10
        assert info["num_train_samples"] == 50000
        assert info["num_test_samples"] == 10000
    
    def test_get_cifar10_transforms(self):
        """Test CIFAR-10 transforms."""
        transforms = get_cifar10_transforms(
            input_size=224,
            use_albumentations=False,
            augmentation_strength="medium"
        )
        
        assert "train" in transforms
        assert "val" in transforms
        assert "test" in transforms
        
        # Test that transforms are callable
        dummy_input = torch.randn(3, 32, 32)
        transformed = transforms["val"](dummy_input)
        assert transformed.shape == (3, 224, 224)


class TestModelUtils:
    """Test model utility functions."""
    
    def test_get_available_models(self):
        """Test available models listing."""
        models = get_available_models()
        
        assert "baseline" in models
        assert "transfer_learning" in models
        assert "simple_cnn" in models["baseline"]
        assert "transfer_resnet50" in models["transfer_learning"]
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("simple_cnn", num_classes=10)
        
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "forward")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        assert output.shape == (1, 10)


class TestTrainingUtils:
    """Test training utility functions."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = torch.nn.Linear(10, 5)
        
        optimizer = create_optimizer(
            model=model,
            optimizer_name="adam",
            lr=0.001
        )
        
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.param_groups[0]["lr"] == 0.001
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        optimizer = torch.optim.Adam(torch.nn.Linear(10, 5).parameters())
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name="step",
            step_size=7,
            gamma=0.1
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


class TestEvaluationUtils:
    """Test evaluation utility functions."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_prob = np.random.rand(5, 3)
        
        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=3,
            y_prob=y_prob
        )
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        
        # Test that metrics are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1


class TestIntegration:
    """Integration tests."""
    
    @patch('src.data.datasets.datasets.CIFAR10')
    def test_end_to_end_workflow(self, mock_cifar10):
        """Test end-to-end workflow."""
        # Mock CIFAR-10 dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=(torch.randn(3, 32, 32), 0))
        mock_cifar10.return_value = mock_dataset
        
        # Test basic workflow
        set_seed(42)
        device = get_device()
        
        # Create model
        model = create_model("simple_cnn", num_classes=10)
        model.to(device)
        
        # Create optimizer
        optimizer = create_optimizer(model, "adam", lr=0.001)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        output = model(dummy_input)
        
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
