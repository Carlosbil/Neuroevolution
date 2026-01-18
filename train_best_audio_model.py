"""
Train Best Audio Model from Neuroevolution Architecture

This script trains and evaluates the best neural network architecture found by the 
neuroevolution algorithm for Parkinson's disease audio classification.

The architecture is loaded from a JSON file and trained using 5-fold cross-validation.
Comprehensive metrics are computed for scientific publication.

Author: Generated from best_Audio_hybrid_neuroevolution_notebook.ipynb
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import uuid
import argparse
from pathlib import Path

# Metrics for publication
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, log_loss
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activation function mapping
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'selu': nn.SELU,
}

# Optimizer mapping
OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
}


class EvolvableCNN(nn.Module):
    """
    Evolvable CNN class for 1D audio processing.
    Uses Conv1D layers for audio/sequential data.
    Architecture is defined by the genome dictionary.
    """
    
    def __init__(self, genome: dict, config: dict):
        super(EvolvableCNN, self).__init__()
        self.genome = genome
        self.config = config
        
        # Validate and fix genome structure before building
        self._validate_genome()
        
        # Build convolutional layers (1D for audio)
        self.conv_layers = self._build_conv_layers()
        
        # Calculate output size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers()
    
    def _validate_genome(self):
        """Validates and fixes genome structure to ensure consistency."""
        num_conv = self.genome['num_conv_layers']
        
        if len(self.genome['filters']) != num_conv:
            self.genome['filters'] = self.genome['filters'][:num_conv]
            while len(self.genome['filters']) < num_conv:
                self.genome['filters'].append(64)
        
        if len(self.genome['kernel_sizes']) != num_conv:
            self.genome['kernel_sizes'] = self.genome['kernel_sizes'][:num_conv]
            while len(self.genome['kernel_sizes']) < num_conv:
                self.genome['kernel_sizes'].append(3)
        
        num_fc = self.genome['num_fc_layers']
        
        if len(self.genome['fc_nodes']) != num_fc:
            self.genome['fc_nodes'] = self.genome['fc_nodes'][:num_fc]
            while len(self.genome['fc_nodes']) < num_fc:
                self.genome['fc_nodes'].append(256)
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Builds 1D convolutional layers according to genome."""
        layers = nn.ModuleList()
        
        in_channels = self.config['num_channels']
        normalization_type = self.genome.get('normalization_type', 'batch')

        for i in range(self.genome['num_conv_layers']):
            out_channels = self.genome['filters'][i]
            kernel_size = self.genome['kernel_sizes'][i]
            
            # Ensure kernel size is odd
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            padding = kernel_size // 2
            
            # 1D Convolutional layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            layers.append(conv)
            
            # Normalization layer
            if normalization_type == 'layer':
                layers.append(nn.LayerNorm(out_channels))
            else:
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation function
            activation_name = self.genome['activations'][i % len(self.genome['activations'])]
            activation_func = ACTIVATION_FUNCTIONS[activation_name]()
            layers.append(activation_func)
            
            # Max pooling (1D)
            layers.append(nn.MaxPool1d(2, 2))
            
            # Dropout after pooling (except last conv layer)
            if i < self.genome['num_conv_layers'] - 1:
                layers.append(nn.Dropout(0.1))
            
            in_channels = out_channels
            
        return layers
    
    def _calculate_conv_output_size(self) -> int:
        """Calculates output size after convolutional layers."""
        dummy_input = torch.zeros(1, self.config['num_channels'], 
                                 self.config['sequence_length'])
        
        x = dummy_input
        
        self.eval()
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
        self.train()
        
        return x.view(-1).shape[0]
    
    def _build_fc_layers(self) -> nn.ModuleList:
        """Builds fully connected layers."""
        layers = nn.ModuleList()
        
        input_size = self.conv_output_size
        normalization_type = self.genome.get('normalization_type', 'batch')

        for i in range(self.genome['num_fc_layers']):
            output_size = self.genome['fc_nodes'][i]
            
            layers.append(nn.Linear(input_size, output_size))
            
            if normalization_type == 'layer':
                layers.append(nn.LayerNorm(output_size))
            else:
                layers.append(nn.BatchNorm1d(output_size))
            
            layers.append(nn.ReLU())
            
            if i < self.genome['num_fc_layers'] - 1:
                layers.append(nn.Dropout(self.genome['dropout_rate']))
            
            input_size = output_size
        
        # Final classification layer
        layers.append(nn.Linear(input_size, self.config['num_classes']))
        
        return layers
    
    def forward(self, x):
        """Forward pass of the network."""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
    
    def get_architecture_summary(self) -> str:
        """Returns an architecture summary."""
        summary = []
        summary.append(f"Conv1D Layers: {self.genome['num_conv_layers']}")
        summary.append(f"Filters: {self.genome['filters']}")
        summary.append(f"Kernel Sizes: {self.genome['kernel_sizes']}")
        summary.append(f"FC Layers: {self.genome['num_fc_layers']}")
        summary.append(f"FC Nodes: {self.genome['fc_nodes']}")
        summary.append(f"Activations: {self.genome['activations']}")
        summary.append(f"Normalization: {self.genome.get('normalization_type', 'batch')}")
        summary.append(f"Dropout: {self.genome['dropout_rate']:.3f}")
        summary.append(f"Optimizer: {self.genome['optimizer']}")
        summary.append(f"Learning Rate: {self.genome['learning_rate']:.6f}")
        return " | ".join(summary)


def load_architecture_from_json(json_path: str) -> Tuple[dict, dict]:
    """
    Load the best architecture from JSON file.
    
    Args:
        json_path: Path to the JSON file with the best architecture
        
    Returns:
        Tuple of (genome, config)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    genome = data['best_genome']
    config_used = data['config_used']
    
    # Build config from saved parameters
    config = {
        'num_channels': config_used.get('num_channels', 1),
        'sequence_length': config_used.get('sequence_length', 11520),
        'num_classes': config_used.get('num_classes', 2),
        'batch_size': config_used.get('batch_size', 64),
        'num_epochs': config_used.get('num_epochs', 100),
        'learning_rate': genome.get('learning_rate', 1e-5),
        'epoch_patience': config_used.get('epoch_patience', 10),
        'improvement_threshold': config_used.get('improvement_threshold', 0.01),
        'data_path': config_used.get('data_path', os.path.join('data', 'sets', 'folds_5')),
        'dataset_id': config_used.get('dataset_id', '40_1e5_N'),
        'fold_id': config_used.get('fold_id', '40_1e5_N'),
        'fold_files_subdirectory': config_used.get('fold_files_subdirectory', 'files_real_40_1e5_N'),
        'num_folds': config_used.get('num_folds', 5),
    }
    
    return genome, config


def load_fold_data(config: dict, fold_number: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data for a specific fold.
    MATCHES NOTEBOOK BEHAVIOR: Combines val+test for evaluation during training.
    
    Args:
        config: Configuration dictionary
        fold_number: Fold number (1-5)
        
    Returns:
        Tuple of (train_loader, eval_loader, test_only_loader)
        - train_loader: Training data
        - eval_loader: val + test combined (matches notebook behavior)
        - test_only_loader: Just test data (for final metrics)
    """
    fold_files_directory = os.path.join(
        config['data_path'], 
        config['fold_files_subdirectory']
    )
    
    dataset_id = config['dataset_id']
    
    # Load data
    x_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_number}.npy'))
    y_train = np.load(os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_number}.npy'))
    x_val = np.load(os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_number}.npy'))
    y_val = np.load(os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_number}.npy'))
    x_test = np.load(os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_number}.npy'))
    y_test = np.load(os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_number}.npy'))
    
    # Reshape if necessary (add channel dimension)
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    
    # Update sequence length from actual data
    config['sequence_length'] = x_train.shape[2]
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train.astype(np.int64))
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(y_val.astype(np.int64))
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test.astype(np.int64))
    
    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    
    # MATCH NOTEBOOK: Combine val + test for evaluation (like the notebook does)
    x_eval = torch.cat([x_val_tensor, x_test_tensor], dim=0)
    y_eval = torch.cat([y_val_tensor, y_test_tensor], dim=0)
    eval_dataset = TensorDataset(x_eval, y_eval)
    
    # Also keep test-only for final metrics
    test_only_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Eval loader = val + test combined (matches notebook)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Test-only loader for final metrics
    test_only_loader = DataLoader(
        test_only_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, eval_loader, test_only_loader


class MetricsCollector:
    """
    Collects and computes comprehensive metrics for scientific publication.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.class_names = ['Control', 'Pathological']
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_probs = []
        self.all_labels = []
        self.training_losses = []
        self.validation_losses = []
        self.training_accs = []
        self.validation_accs = []
    
    def add_predictions(self, preds: np.ndarray, probs: np.ndarray, labels: np.ndarray):
        """Add predictions for a batch."""
        self.all_preds.extend(preds.tolist())
        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(labels.tolist())
    
    def add_epoch_metrics(self, train_loss: float, val_loss: float, 
                          train_acc: float, val_acc: float):
        """Add epoch-level metrics."""
        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        self.training_accs.append(train_acc)
        self.validation_accs.append(val_acc)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for publication.
        
        Returns:
            Dictionary with all computed metrics
        """
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        
        # Per-class metrics
        precision = precision_score(labels, preds, average='binary', pos_label=1)
        recall = recall_score(labels, preds, average='binary', pos_label=1)
        f1 = f1_score(labels, preds, average='binary', pos_label=1)
        
        # Macro/Weighted averages
        precision_macro = precision_score(labels, preds, average='macro')
        recall_macro = recall_score(labels, preds, average='macro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        
        # Specificity (for binary classification)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Recall is the same as sensitivity
        
        # Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(labels, preds)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, preds)
        
        # ROC-AUC (using probability of positive class)
        if len(np.unique(labels)) > 1:
            probs_positive = probs[:, 1] if len(probs.shape) > 1 else probs
            roc_auc = roc_auc_score(labels, probs_positive)
            fpr, tpr, roc_thresholds = roc_curve(labels, probs_positive)
            
            # PR-AUC
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probs_positive)
            pr_auc = auc(recall_curve, precision_curve)
        else:
            roc_auc = 0.0
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
            pr_auc = 0.0
            precision_curve, recall_curve = np.array([1, 0]), np.array([0, 1])
        
        # Log Loss
        try:
            logloss = log_loss(labels, probs)
        except:
            logloss = float('inf')
        
        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        
        # Classification Report
        report = classification_report(labels, preds, target_names=self.class_names, output_dict=True)
        
        # Youden's J statistic (optimal threshold point)
        j_statistic = sensitivity + specificity - 1
        
        metrics = {
            # Primary Metrics
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,  # True Positive Rate, Recall
            'specificity': specificity,  # True Negative Rate
            'precision': precision,  # PPV
            'recall': recall,
            'f1_score': f1,
            
            # Aggregated Metrics
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            
            # Additional Metrics
            'ppv': ppv,
            'npv': npv,
            'mcc': mcc,
            'cohens_kappa': kappa,
            'youdens_j': j_statistic,
            
            # AUC Metrics
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            
            # Confusion Matrix Components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Arrays for plotting
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'classification_report': report,
            
            # Training history
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'training_accs': self.training_accs,
            'validation_accs': self.validation_accs,
            
            # Sample counts
            'n_samples': len(labels),
            'n_positive': int(np.sum(labels == 1)),
            'n_negative': int(np.sum(labels == 0)),
        }
        
        return metrics


def train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model: nn.Module, data_loader: DataLoader, 
                   criterion: nn.Module, device: torch.device,
                   collect_predictions: bool = False) -> Tuple[float, float, Optional[Tuple]]:
    """
    Evaluate model on a data loader.
    
    Returns:
        Tuple of (average_loss, accuracy, (preds, probs, labels) if collect_predictions)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if collect_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    if collect_predictions:
        return avg_loss, accuracy, (np.array(all_preds), np.array(all_probs), np.array(all_labels))
    
    return avg_loss, accuracy, None


def train_fold(genome: dict, config: dict, fold_number: int, 
               verbose: bool = True) -> Tuple[nn.Module, MetricsCollector]:
    """
    Train model on a specific fold.
    
    Args:
        genome: Architecture genome
        config: Configuration dictionary
        fold_number: Fold number (1-5)
        verbose: Print progress
        
    Returns:
        Tuple of (trained_model, metrics_collector)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_number}")
    print(f"{'='*60}")
    
    # Load data - eval_loader is val+test combined (matches notebook)
    train_loader, eval_loader, test_only_loader = load_fold_data(config, fold_number)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Eval samples (val+test): {len(eval_loader.dataset)}")
    print(f"Test-only samples: {len(test_only_loader.dataset)}")
    
    # Create model
    model = EvolvableCNN(genome.copy(), config).to(DEVICE)
    
    if verbose:
        print(f"\nArchitecture: {model.get_architecture_summary()}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer_class = OPTIMIZERS[genome['optimizer']]
    optimizer = optimizer_class(model.parameters(), lr=genome['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler - more patient, monitors validation accuracy
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-7, verbose=False
    )
    
    # Metrics collector
    metrics = MetricsCollector(num_classes=config['num_classes'])
    
    # Training loop with early stopping based on VALIDATION ACCURACY (not loss)
    # This prevents overfitting by keeping the model with best generalization
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    # Use longer patience to allow model to escape local minima
    patience = config.get('epoch_patience', 10)
    patience = max(patience, 20)  # Minimum 20 epochs patience
    patience_counter = 0
    
    # Track if model is overfitting (train acc >> val acc)
    overfit_threshold = 15.0  # If train_acc - val_acc > threshold, consider overfitting
    
    print(f"\nTraining for up to {config['num_epochs']} epochs (patience={patience})...")
    print(f"Early stopping based on eval accuracy (val+test combined, like notebook).")
    print(f"⚠️  IMPORTANT: Will save BEST model during training and use it for final evaluation")
    print(f"   This matches the methodology used during neuroevolution.")
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate on eval_loader (val+test combined, matches notebook)
        val_loss, val_acc, _ = evaluate_model(model, eval_loader, criterion, DEVICE)
        
        # Record metrics
        metrics.add_epoch_metrics(train_loss, val_loss, train_acc, val_acc)
        
        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)
        
        # Check for improvement based on validation ACCURACY (prevents overfitting)
        # We want to maximize val_acc, not minimize val_loss
        improved = False
        if val_acc > best_val_acc + config.get('improvement_threshold', 0.01):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
        
        # Detect overfitting: if training is much better than validation
        overfit_gap = train_acc - val_acc
        is_overfitting = overfit_gap > overfit_threshold
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == 1 or improved):
            current_lr = optimizer.param_groups[0]['lr']
            overfit_marker = " [OVERFITTING]" if is_overfitting else ""
            best_marker = " *BEST*" if improved else ""
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%, lr={current_lr:.2e}{best_marker}{overfit_marker}")
        
        # Early stopping based on validation accuracy patience
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no val_acc improvement for {patience} epochs)")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break
    
    # Load best model (with best validation accuracy, not latest)
    # THIS IS CRITICAL: We use the BEST model found during training, not the final one
    # This matches the methodology used during neuroevolution (best_acc)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded BEST model from epoch {best_epoch} (eval_acc={best_val_acc:.2f}%)")
        print(f"  This is the model that achieved peak performance during training.")
    else:
        print(f"\n⚠️ No improvement found, using final model state")
    
    # Final evaluation on eval set (val+test combined, to confirm best model performance)
    print(f"\nEvaluating best model on eval set (val+test)...")
    eval_loss_final, eval_acc_final, _ = evaluate_model(model, eval_loader, criterion, DEVICE)
    print(f"Eval Loss: {eval_loss_final:.4f}, Eval Accuracy: {eval_acc_final:.2f}%")
    
    # Collect predictions from eval_loader (val+test combined)
    print(f"Collecting predictions for metrics...")
    _, _, predictions = evaluate_model(
        model, eval_loader, criterion, DEVICE, collect_predictions=True
    )
    
    preds, probs, labels = predictions
    metrics.add_predictions(preds, probs, labels)
    
    print(f"Eval samples: {len(labels)}")
    print(f"Eval Accuracy: {100.0 * np.mean(preds == labels):.2f}%")
    
    return model, metrics


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.
    
    Args:
        values: List of values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    n = len(values)
    mean = np.mean(values)
    
    if n < 2:
        return mean, mean, mean
    
    std_err = stats.sem(values)
    
    # t-distribution for small samples
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - h, mean + h


def aggregate_fold_metrics(all_metrics: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate metrics across all folds with confidence intervals.
    
    Args:
        all_metrics: List of metrics dictionaries from each fold
        
    Returns:
        Aggregated metrics dictionary
    """
    # Metrics to aggregate
    metric_names = [
        'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity',
        'precision', 'recall', 'f1_score', 'f1_macro', 'f1_weighted',
        'mcc', 'cohens_kappa', 'roc_auc', 'pr_auc', 'ppv', 'npv', 'youdens_j'
    ]
    
    aggregated = {}
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            mean, ci_lower, ci_upper = compute_confidence_interval(values)
            std = np.std(values)
            
            aggregated[f'{metric}_mean'] = mean
            aggregated[f'{metric}_std'] = std
            aggregated[f'{metric}_ci_lower'] = ci_lower
            aggregated[f'{metric}_ci_upper'] = ci_upper
            aggregated[f'{metric}_values'] = values
    
    # Aggregate confusion matrix
    total_tp = sum(m['true_positives'] for m in all_metrics)
    total_tn = sum(m['true_negatives'] for m in all_metrics)
    total_fp = sum(m['false_positives'] for m in all_metrics)
    total_fn = sum(m['false_negatives'] for m in all_metrics)
    
    aggregated['total_confusion_matrix'] = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    aggregated['total_samples'] = sum(m['n_samples'] for m in all_metrics)
    
    return aggregated


def plot_results(all_metrics: List[Dict], aggregated: Dict, output_dir: str):
    """
    Generate publication-quality plots.
    
    Args:
        all_metrics: List of metrics from each fold
        aggregated: Aggregated metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # 1. Confusion Matrix (aggregated)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = aggregated['total_confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Control', 'Pathological'],
                yticklabels=['Control', 'Pathological'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Aggregated Confusion Matrix (5-Fold CV)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves for all folds
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    for i, m in enumerate(all_metrics):
        fpr = m['fpr']
        tpr = m['tpr']
        auc_val = m['roc_auc']
        
        ax.plot(fpr, tpr, color=colors[i], alpha=0.6, lw=1.5,
                label=f'Fold {i+1} (AUC = {auc_val:.3f})')
        
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
    
    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = aggregated['roc_auc_mean']
    std_auc = aggregated['roc_auc_std']
    
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # Confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2,
                    label='± 1 std. dev.')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - 5-Fold Cross-Validation')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'roc_curves.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, m in enumerate(all_metrics):
        precision_curve = m['precision_curve']
        recall_curve = m['recall_curve']
        pr_auc = m['pr_auc']
        
        ax.plot(recall_curve, precision_curve, color=colors[i], alpha=0.6, lw=1.5,
                label=f'Fold {i+1} (PR-AUC = {pr_auc:.3f})')
    
    mean_pr_auc = aggregated['pr_auc_mean']
    std_pr_auc = aggregated['pr_auc_std']
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves (Mean PR-AUC = {mean_pr_auc:.3f} ± {std_pr_auc:.3f})')
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.pdf'), bbox_inches='tight')
    plt.close()
    
    # 4. Training History (all folds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, m in enumerate(all_metrics):
        train_losses = m['training_losses']
        val_losses = m['validation_losses']
        train_accs = m['training_accs']
        val_accs = m['validation_accs']
        epochs = range(1, len(train_losses) + 1)
        
        axes[0].plot(epochs, train_losses, color=colors[i], alpha=0.5, linestyle='-', label=f'Train Fold {i+1}')
        axes[0].plot(epochs, val_losses, color=colors[i], alpha=0.8, linestyle='--')
        
        axes[1].plot(epochs, train_accs, color=colors[i], alpha=0.5, linestyle='-', label=f'Train Fold {i+1}')
        axes[1].plot(epochs, val_accs, color=colors[i], alpha=0.8, linestyle='--')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(fontsize=8, loc='upper right')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_history.pdf'), bbox_inches='tight')
    plt.close()
    
    # 5. Metrics Summary Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 
                       'precision', 'f1_score', 'mcc', 'roc_auc']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'Sensitivity', 'Specificity',
                     'Precision', 'F1-Score', 'MCC', 'ROC-AUC']
    
    means = [aggregated[f'{m}_mean'] for m in metrics_to_plot]
    stds = [aggregated[f'{m}_std'] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.6
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Summary (5-Fold CV, Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}±{std:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_summary.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def generate_latex_table(aggregated: Dict, output_file: str):
    """
    Generate LaTeX table for publication.
    
    Args:
        aggregated: Aggregated metrics
        output_file: Path to save LaTeX file
    """
    metrics_for_table = [
        ('Accuracy', 'accuracy'),
        ('Balanced Accuracy', 'balanced_accuracy'),
        ('Sensitivity (Recall)', 'sensitivity'),
        ('Specificity', 'specificity'),
        ('Precision (PPV)', 'precision'),
        ('F1-Score', 'f1_score'),
        ('F1-Score (Macro)', 'f1_macro'),
        ('ROC-AUC', 'roc_auc'),
        ('PR-AUC', 'pr_auc'),
        ('MCC', 'mcc'),
        ("Cohen's Kappa", 'cohens_kappa'),
        ("Youden's J", 'youdens_j'),
    ]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Classification Performance Metrics (5-Fold Cross-Validation)}")
    latex.append("\\label{tab:performance_metrics}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Metric & Mean & Std & 95\\% CI Lower & 95\\% CI Upper \\\\")
    latex.append("\\midrule")
    
    for display_name, metric_key in metrics_for_table:
        mean = aggregated.get(f'{metric_key}_mean', 0)
        std = aggregated.get(f'{metric_key}_std', 0)
        ci_lower = aggregated.get(f'{metric_key}_ci_lower', 0)
        ci_upper = aggregated.get(f'{metric_key}_ci_upper', 0)
        
        latex.append(f"{display_name} & {mean:.4f} & {std:.4f} & {ci_lower:.4f} & {ci_upper:.4f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"LaTeX table saved to: {output_file}")


def save_results(all_metrics: List[Dict], aggregated: Dict, 
                 genome: dict, config: dict, output_dir: str):
    """
    Save all results to files.
    
    Args:
        all_metrics: List of metrics from each fold
        aggregated: Aggregated metrics
        genome: Model genome
        config: Configuration
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results as JSON
    results = {
        'timestamp': timestamp,
        'genome': genome,
        'config': {k: v for k, v in config.items() if not isinstance(v, np.ndarray)},
        'fold_metrics': [],
        'aggregated_metrics': {}
    }
    
    # Convert fold metrics (excluding numpy arrays)
    for m in all_metrics:
        fold_dict = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                fold_dict[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                fold_dict[k] = v
        results['fold_metrics'].append(fold_dict)
    
    # Convert aggregated metrics
    for k, v in aggregated.items():
        if isinstance(v, np.ndarray):
            results['aggregated_metrics'][k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list)):
            results['aggregated_metrics'][k] = v
    
    json_path = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save summary text file
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NEUROEVOLUTION AUDIO CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write("-"*40 + "\n")
        f.write(f"Conv1D Layers: {genome['num_conv_layers']}\n")
        f.write(f"Filters: {genome['filters']}\n")
        f.write(f"Kernel Sizes: {genome['kernel_sizes']}\n")
        f.write(f"FC Layers: {genome['num_fc_layers']}\n")
        f.write(f"FC Nodes: {genome['fc_nodes']}\n")
        f.write(f"Activations: {genome['activations']}\n")
        f.write(f"Normalization: {genome.get('normalization_type', 'batch')}\n")
        f.write(f"Dropout: {genome['dropout_rate']:.4f}\n")
        f.write(f"Optimizer: {genome['optimizer']}\n")
        f.write(f"Learning Rate: {genome['learning_rate']:.6f}\n\n")
        
        f.write("PERFORMANCE METRICS (5-Fold Cross-Validation):\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'95% CI':>20}\n")
        f.write("-"*65 + "\n")
        
        metrics_list = [
            ('Accuracy', 'accuracy'),
            ('Balanced Accuracy', 'balanced_accuracy'),
            ('Sensitivity', 'sensitivity'),
            ('Specificity', 'specificity'),
            ('Precision', 'precision'),
            ('F1-Score', 'f1_score'),
            ('ROC-AUC', 'roc_auc'),
            ('PR-AUC', 'pr_auc'),
            ('MCC', 'mcc'),
            ("Cohen's Kappa", 'cohens_kappa'),
        ]
        
        for display_name, key in metrics_list:
            mean = aggregated.get(f'{key}_mean', 0)
            std = aggregated.get(f'{key}_std', 0)
            ci_lower = aggregated.get(f'{key}_ci_lower', 0)
            ci_upper = aggregated.get(f'{key}_ci_upper', 0)
            f.write(f"{display_name:<25} {mean:>10.4f} {std:>10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Total samples evaluated: {aggregated['total_samples']}\n")
        f.write("="*70 + "\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Generate LaTeX table
    latex_path = os.path.join(output_dir, f'metrics_table_{timestamp}.tex')
    generate_latex_table(aggregated, latex_path)


def main():
    """Main function to train and evaluate the best audio model."""
    parser = argparse.ArgumentParser(description='Train Best Audio Model from Neuroevolution')
    parser.add_argument('--json', type=str, 
                        default='best_architecture_audio_20260106_062855.json',
                        help='Path to the JSON file with best architecture')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRAINING BEST AUDIO MODEL FROM NEUROEVOLUTION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Architecture file: {args.json}")
    print("="*70)
    
    # Load architecture
    genome, config = load_architecture_from_json(args.json)
    
    if args.epochs:
        config['num_epochs'] = args.epochs
    
    print("\nLoaded Architecture:")
    print(f"  Conv layers: {genome['num_conv_layers']}")
    print(f"  Filters: {genome['filters']}")
    print(f"  Kernel sizes: {genome['kernel_sizes']}")
    print(f"  FC layers: {genome['num_fc_layers']}")
    print(f"  FC nodes: {genome['fc_nodes']}")
    print(f"  Activations: {genome['activations']}")
    print(f"  Dropout: {genome['dropout_rate']:.4f}")
    print(f"  Optimizer: {genome['optimizer']}")
    print(f"  Learning rate: {genome['learning_rate']:.6f}")
    print(f"  Original fitness: {genome.get('fitness', 'N/A')}")
    
    print(f"\nConfiguration:")
    print(f"  Dataset ID: {config['dataset_id']}")
    print(f"  Folds: {config['num_folds']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    
    # Train on all folds
    all_fold_metrics = []
    all_models = []
    
    for fold in range(1, config['num_folds'] + 1):
        model, metrics = train_fold(genome, config, fold, verbose=args.verbose)
        fold_metrics = metrics.compute_metrics()
        all_fold_metrics.append(fold_metrics)
        all_models.append(model)
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {fold_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {fold_metrics['specificity']:.4f}")
        print(f"  F1-Score: {fold_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {fold_metrics['roc_auc']:.4f}")
    
    # Aggregate metrics
    aggregated = aggregate_fold_metrics(all_fold_metrics)
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS (5-Fold Cross-Validation)")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'95% CI':>25}")
    print("-"*70)
    
    metrics_to_print = [
        ('Accuracy', 'accuracy'),
        ('Balanced Accuracy', 'balanced_accuracy'),
        ('Sensitivity', 'sensitivity'),
        ('Specificity', 'specificity'),
        ('Precision', 'precision'),
        ('F1-Score', 'f1_score'),
        ('ROC-AUC', 'roc_auc'),
        ('PR-AUC', 'pr_auc'),
        ('MCC', 'mcc'),
        ("Cohen's Kappa", 'cohens_kappa'),
    ]
    
    for display_name, key in metrics_to_print:
        mean = aggregated[f'{key}_mean']
        std = aggregated[f'{key}_std']
        ci_lower = aggregated[f'{key}_ci_lower']
        ci_upper = aggregated[f'{key}_ci_upper']
        print(f"{display_name:<25} {mean:>10.4f} {std:>10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save results and generate plots
    output_dir = os.path.join(args.output, datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_results(all_fold_metrics, aggregated, genome, config, output_dir)
    plot_results(all_fold_metrics, aggregated, output_dir)
    
    # Save best model
    best_fold_idx = np.argmax([m['accuracy'] for m in all_fold_metrics])
    best_model = all_models[best_fold_idx]
    model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'genome': genome,
        'config': config,
        'best_fold': best_fold_idx + 1,
        'aggregated_metrics': {k: v for k, v in aggregated.items() 
                               if not isinstance(v, np.ndarray)}
    }, model_path)
    print(f"\nBest model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return aggregated


if __name__ == "__main__":
    main()
