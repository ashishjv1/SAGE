#!/usr/bin/env python3
"""
SAGE: Scalable Agreement-based Gradient Estimation
End-to-end training script with full command-line interface
"""

import argparse
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Optional
import logging

# Import our modules
from model_factory import create_model
from sage_core import (
    FDStreamer, 
    gpu_fd_stream,
    class_balanced_agreeing_subset_fast,
    agreeing_subset_fast,
    _project_single_grad,
    per_sample_grads_slow,
    per_sample_grads_last_layer_only,
    cutmix_criterion,
    CutMixDataLoader
)
from data_utils import get_dataset, apply_label_corruption, apply_minority_downsampling

# Import baselines separately (not included in main repo)
try:
    from baselines import gradmatch_selection, random_selection, uncertainty_selection
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    logger.warning("Baselines module not available. Only SAGE methods will work.")

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SAGE Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100', 
                       choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to store/load dataset')
    
    # SAGE subset selection arguments
    parser.add_argument('--per_class', action='store_true', default=True,
                       help='Use per-class subset selection')
    parser.add_argument('--subset_fraction', type=float, default=0.05,
                       help='Fraction of training data to select (e.g., 0.05 = 5 percent)')
    parser.add_argument('--sketch_size', type=int, default=256,
                       help='Sketch size (ell) for frequent directions')
    parser.add_argument('--selection_method', type=str, default='sage',
                       choices=['sage', 'gradmatch', 'random'],
                       help='Subset selection method')
    parser.add_argument('--fd_method', type=str, default='fd',
                       choices=['fd', 'random_projection'],
                       help='Dimensionality reduction method: FD vs random projection')
    parser.add_argument('--scoring_method', type=str, default='agreement',
                       choices=['agreement', 'gradient_norm'],
                       help='Scoring method: agreement-based vs gradient-norm')
    
    # Training hyperparameters
    parser.add_argument('--model', type=str, default='resnext',
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    
    # SAGE refresh and warmup
    parser.add_argument('--refresh_epochs', type=int, default=50,
                       help='Epochs between subset refreshes')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Warmup epochs before first selection')
    parser.add_argument('--first_selection_epoch', type=int, default=1,
                       help='Epoch for first subset selection')
    parser.add_argument('--single_selection', action='store_true',
                       help='Perform subset selection only once (no iterative refresh)')
    parser.add_argument('--no_warmup', action='store_true',
                       help='Skip warmup and start with subset selection immediately')
    parser.add_argument('--no_selection', action='store_true',
                       help='Train on full dataset without any subset selection')
    
    # Compression schedule
    parser.add_argument('--compression_schedule', type=str, default='cpu',
                       choices=['cpu', 'gpu'],
                       help='Where to perform FD compression')
    
    # Data corruption experiments
    parser.add_argument('--corrupt_labels', type=float, default=0.0,
                       help='Fraction of labels to corrupt (0.0-1.0)')
    parser.add_argument('--minority_downsample', type=float, default=0.0,
                       help='Fraction to downsample minority classes (0.0-1.0)')
    
    # Technical parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--batch_size_fd', type=int, default=32,
                       help='Batch size for FD streaming')
    parser.add_argument('--chunk_size_grad', type=int, default=32,
                       help='Chunk size for gradient computation')
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (epochs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # CutMix augmentation
    parser.add_argument('--cutmix', action='store_true',
                       help='Enable CutMix augmentation during training')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                       help='CutMix alpha parameter (beta distribution)')
    parser.add_argument('--cutmix_prob', type=float, default=0.5,
                       help='Probability of applying CutMix per batch')
    
    # Optional TensorBoard logging
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                       help='TensorBoard log directory')
    
    # Checkpointing and warm start
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--save_checkpoint_freq', type=int, default=50,
                       help='Save checkpoint every N epochs (0 to disable)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_arg)


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer based on arguments"""
    if args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, 
                         weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, args):
    """Create learning rate scheduler"""
    if args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_sketch(model: nn.Module, loader: DataLoader, args, device: torch.device) -> torch.Tensor:
    """Build FD sketch or random projection matrix"""
    if args.fd_method == 'random_projection':
        # Create random projection matrix
        # First, get the gradient dimension
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            grad_sample = per_sample_grads_last_layer_only(model, x[:1], y[:1])
            D = grad_sample.shape[1]
            break
        
        # Create random Gaussian matrix and orthogonalize
        P = torch.randn(args.sketch_size, D, dtype=torch.float32)
        P = torch.nn.functional.normalize(P, dim=1)
        return P
    
    else:  # FD method
        fd = FDStreamer(args.sketch_size, batch_size=args.batch_size_fd, dtype=torch.float16)
        
        for xb, yb in tqdm(loader, desc="🔨 Building FD sketch (optimized)"):
            if args.compression_schedule == 'gpu':
                xb, yb = xb.to(device), yb.to(device)
                rows = per_sample_grads_last_layer_only(model, xb, yb)
            else:  # CPU compression
                # Move data to device for gradient computation
                xb, yb = xb.to(device), yb.to(device)
                rows = per_sample_grads_last_layer_only(model, xb, yb)
                # Convert to CPU for FD streaming
                if isinstance(rows, torch.Tensor):
                    rows = rows.cpu().numpy()
            
            fd.add(rows)
        
        print("✅ Sketch building complete")
        return torch.from_numpy(fd.finalize())


def select_subset(model: nn.Module, dataset, args, device: torch.device, 
                 proj_matrix: torch.Tensor) -> List[int]:
    """Select subset using specified method"""
    
    if args.selection_method == 'random':
        # Random baseline
        if BASELINES_AVAILABLE:
            num_classes = get_num_classes(args.dataset)
            total_size = int(args.subset_fraction * len(dataset))
            return random_selection(dataset, total_size, args.per_class, num_classes)
        else:
            # Fallback random selection
            total_size = int(args.subset_fraction * len(dataset))
            return np.random.choice(len(dataset), total_size, replace=False).tolist()
    
    elif args.selection_method == 'gradmatch':
        # GradMatch: gradient norm based selection
        if not BASELINES_AVAILABLE:
            raise ValueError("Baselines module required for GradMatch. Use SAGE instead.")
        
        num_classes = get_num_classes(args.dataset)
        criterion = nn.CrossEntropyLoss()
        total_size = int(args.subset_fraction * len(dataset))
        
        return gradmatch_selection(
            model, dataset, total_size, criterion, device, proj_matrix,
            per_class=args.per_class, num_classes=num_classes
        )
    
    else:  # SAGE
        num_classes = get_num_classes(args.dataset)
        
        if args.per_class:
            k_per_cls = int(args.subset_fraction * len(dataset) / num_classes)
            return class_balanced_agreeing_subset_fast(
                model, dataset, num_classes, k_per_cls,
                nn.CrossEntropyLoss(), device, proj_matrix.to(device),
                batch_size_data=args.batch_size, 
                chunk_size_grad=args.chunk_size_grad
            )
        else:
            subset_size = int(args.subset_fraction * len(dataset))
            return agreeing_subset_fast(
                model, dataset, subset_size,
                nn.CrossEntropyLoss(), device, proj_matrix.to(device),
                batch_size_data=args.batch_size,
                chunk_size_grad=args.chunk_size_grad
            )


def gradmatch_subset_selection(model: nn.Module, dataset, args, device: torch.device, 
                              proj_matrix: torch.Tensor) -> List[int]:
    """GradMatch baseline: select based on gradient norms"""
    from sage_core import compute_gradient_norms
    
    num_classes = get_num_classes(args.dataset)
    if args.per_class:
        k_per_cls = int(args.subset_fraction * len(dataset) / num_classes)
        return compute_gradient_norms(
            model, dataset, num_classes, k_per_cls,
            nn.CrossEntropyLoss(), device, proj_matrix.to(device),
            batch_size_data=args.batch_size,
            chunk_size_grad=args.chunk_size_grad
        )
    else:
        subset_size = int(args.subset_fraction * len(dataset))
        return compute_gradient_norms(
            model, dataset, subset_size,
            nn.CrossEntropyLoss(), device, proj_matrix.to(device),
            batch_size_data=args.batch_size,
            chunk_size_grad=args.chunk_size_grad,
            per_class=False
        )


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for dataset"""
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'tiny_imagenet':
        return 200
    elif dataset_name == 'imagenet':
        return 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device, 
               scaler: Optional[torch.cuda.amp.GradScaler] = None,
               use_cutmix: bool = False) -> tuple:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_data in tqdm(loader, desc="Training", leave=False):
        if use_cutmix and len(batch_data) == 5:
            # CutMix batch: inputs, targets_a, targets_b, lam, cutmix_applied
            inputs, targets_a, targets_b, lam, cutmix_applied = batch_data
            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
        else:
            # Regular batch: inputs, targets
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            cutmix_applied = False
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if cutmix_applied:
                    loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if cutmix_applied:
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        
        if cutmix_applied:
            # For CutMix, accuracy is weighted combination
            correct += (lam * (pred == targets_a).float() + 
                       (1 - lam) * (pred == targets_b).float()).sum().item()
        else:
            correct += (pred == targets).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> tuple:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        correct += (pred == targets).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total


def save_results(results: Dict[str, Any], args):
    """Save training results"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save configuration
    config = vars(args).copy()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler, results: Dict[str, Any], subset_indices: List[int],
                   proj_matrix: torch.Tensor, args, checkpoint_dir: str):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'results': results,
        'subset_indices': subset_indices,
        'proj_matrix': proj_matrix,
        'args': vars(args),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler, device: torch.device):
    """Load training checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore RNG states for reproducibility
    torch.set_rng_state(checkpoint['rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state']:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    return (checkpoint['epoch'], checkpoint['results'], checkpoint['subset_indices'],
            checkpoint['proj_matrix'], checkpoint['args'])


def main():
    args = parse_args()
    
    # Validate argument combinations
    if args.no_warmup and args.warmup_epochs > 0:
        logger.warning("--no_warmup conflicts with --warmup_epochs > 0. Setting warmup_epochs to 0.")
        args.warmup_epochs = 0
    
    if args.single_selection and args.refresh_epochs < args.epochs:
        logger.info(f"Single selection mode: subset will not be refreshed every {args.refresh_epochs} epochs")
    
    if args.no_warmup and args.single_selection:
        logger.info("No-warmup + single selection mode: subset selected at epoch 1, no refreshes")
    elif args.no_warmup:
        logger.info("No-warmup mode: subset selected at epoch 1, refreshed every N epochs")
    elif args.single_selection:
        logger.info("Single selection mode: subset selected once, no refreshes")
    elif args.no_selection:
        logger.info("No selection mode: training on full dataset")
        args.subset_fraction = 1.0  # Override to use full dataset
    
    # Set seed and device
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, test_dataset = get_dataset(args.dataset, args.data_path)
    
    # Apply data corruptions if specified
    if args.corrupt_labels > 0:
        train_dataset = apply_label_corruption(train_dataset, args.corrupt_labels)
        logger.info(f"Applied {args.corrupt_labels*100:.1f}% label corruption")
    
    if args.minority_downsample > 0:
        train_dataset = apply_minority_downsampling(train_dataset, args.minority_downsample)
        logger.info(f"Applied {args.minority_downsample*100:.1f}% minority downsampling")
    
    # Create model
    num_classes = get_num_classes(args.dataset)
    model = create_model(args.model, num_classes=num_classes).to(device)
    
    # Create model for gradient computation (potentially different from training model)
    model_for_grad = create_model(args.model, num_classes=num_classes).to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    
    # Initialize checkpoint directory
    if args.save_checkpoint_freq > 0 or args.resume:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize training state
    start_epoch = 1
    subset_indices = []
    proj_matrix = None
    results = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'selection_times': [],
        'subset_sizes': [],
        'epoch_times': [],
        'learning_rates': [],
        'selection_epochs': [],
        'memory_usage': [],
        'config': vars(args)
    }
    
    # Load checkpoint if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        
        start_epoch, results, subset_indices, proj_matrix, loaded_args = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        
        logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Restore subset if it exists
        if subset_indices:
            subset_dataset = Subset(train_dataset, subset_indices)
            base_loader = DataLoader(subset_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)
            if args.cutmix:
                train_loader = CutMixDataLoader(base_loader, args.cutmix_alpha, args.cutmix_prob)
            else:
                train_loader = base_loader
            logger.info(f"Restored subset with {len(subset_indices)} samples")
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE:
        run_name = f"{args.dataset}_{args.selection_method}_{args.subset_fraction}_{int(time.time())}"
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, run_name))
        logger.info(f"TensorBoard logging enabled: {args.tensorboard_dir}/{run_name}")
    elif args.tensorboard and not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
    
    # Initial data loaders (if not resuming with subset)
    if not subset_indices:
        full_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_loader = full_loader
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Training loop
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Subset selection logic
        do_selection = False
        
        # Handle no_selection mode - never perform subset selection
        if args.no_selection:
            do_selection = False
        
        # Handle no_warmup mode - immediate selection at epoch 1
        elif args.no_warmup and epoch == 1 and not subset_indices:
            do_selection = True
            logger.info(f"No-warmup subset selection at epoch {epoch}")
        
        # Handle single_selection mode - only select once
        elif args.single_selection and not subset_indices:
            if args.warmup_epochs == 0 and epoch == args.first_selection_epoch:
                do_selection = True
                logger.info(f"Single subset selection at epoch {epoch}")
            elif args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1:
                do_selection = True
                logger.info(f"Single subset selection after warmup at epoch {epoch}")
        
        # Handle iterative selection mode (default behavior)
        elif not args.single_selection and not args.no_warmup:
            if epoch == args.first_selection_epoch and args.warmup_epochs == 0 and not subset_indices:
                do_selection = True
                logger.info(f"Initial subset selection at epoch {epoch}")
            elif epoch == args.warmup_epochs + 1 and args.warmup_epochs > 0 and not subset_indices:
                do_selection = True
                logger.info(f"Subset selection after warmup at epoch {epoch}")
            elif (subset_indices and epoch > max(args.first_selection_epoch, args.warmup_epochs + 1) and 
                  (epoch - max(args.first_selection_epoch, args.warmup_epochs + 1)) % args.refresh_epochs == 0):
                do_selection = True
                logger.info(f"Subset refresh at epoch {epoch}")
        
        if do_selection:
            selection_start = time.time()
            
            # Build sketch/projection matrix
            logger.info("Building projection matrix...")
            full_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True)
            proj_matrix = build_sketch(model_for_grad, full_loader, args, device)
            
            # Select subset
            logger.info("Selecting subset...")
            subset_indices = select_subset(model_for_grad, train_dataset, args, device, proj_matrix)
            
            # Create new data loader
            subset_dataset = Subset(train_dataset, subset_indices)
            base_loader = DataLoader(subset_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)
            
            # Wrap with CutMix if enabled
            if args.cutmix:
                train_loader = CutMixDataLoader(base_loader, args.cutmix_alpha, args.cutmix_prob)
            else:
                train_loader = base_loader
            
            selection_time = time.time() - selection_start
            results['selection_times'].append(selection_time)
            results['subset_sizes'].append(len(subset_indices))
            
            logger.info(f"Selected {len(subset_indices)} samples in {selection_time:.2f}s")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.cutmix)
        
        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Record results
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        results['epoch_times'].append(epoch_time)
        results['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Record memory usage if CUDA available
        if torch.cuda.is_available():
            results['memory_usage'].append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
        
        # Record if subset selection happened this epoch
        if do_selection:
            results['selection_epochs'].append(epoch)
        
        # Logging
        if epoch % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
                       f"lr {lr:.5f} | "
                       f"train_loss {train_loss:.4f} | train_acc {train_acc*100:.2f}% | "
                       f"test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}% | "
                       f"time {epoch_time:.2f}s")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Time/Epoch', epoch_time, epoch)
            
            if do_selection and 'selection_time' in locals():
                writer.add_scalar('SAGE/Selection_Time', selection_time, epoch)
                writer.add_scalar('SAGE/Subset_Size', len(subset_indices), epoch)
            
            if torch.cuda.is_available():
                writer.add_scalar('Memory/GPU_Usage_GB', torch.cuda.max_memory_allocated() / 1024**3, epoch)
        
        # Save checkpoint
        if args.save_checkpoint_freq > 0 and epoch % args.save_checkpoint_freq == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, results, subset_indices,
                          proj_matrix, args, args.checkpoint_dir)
        
        # Save model if requested
        if args.save_model and epoch == args.epochs:
            model_path = os.path.join(args.output_dir, 'final_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Final model saved: {model_path}")
    
    # Save results
    save_results(results, args)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Final test accuracy: {results['test_accs'][-1]*100:.2f}%")
    logger.info(f"Best test accuracy: {max(results['test_accs'])*100:.2f}%")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if args.tensorboard and writer is not None:
        logger.info(f"TensorBoard logs: {args.tensorboard_dir}")
        logger.info("View with: tensorboard --logdir " + args.tensorboard_dir)


if __name__ == '__main__':
    main()

