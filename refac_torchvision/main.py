import argparse
from data.custom_dataset import CustomDataset
from data.transforms import get_transform
from torch.utils.data import DataLoader
from train.utils import collate_fn, set_seed
from train.train import run_fold_training, run_standard_training
from models.model import create_model
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train object detection model with configurable parameters")
    
    # General configurations
    parser.add_argument('--annotation_path', type=str, default="D:/AI_Tech_파일들/Level_2_Project/data/dataset/train.json", help='Path to the annotation file (JSON)')
    parser.add_argument('--data_dir', type=str, default="D:/AI_Tech_파일들/Level_2_Project/data/dataset", help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='fasterrcnn_resnet50_fpn', help='Model name to use for training')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--base_dir', type=str, default='./results', help='Base directory for saving results')

    # Training configurations
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for K-Fold cross-validation')
    parser.add_argument('--training_mode', type=str, choices=['standard', 'fold'], required=True, help='Training mode: "standard" for normal training, "fold" for K-Fold cross-validation')

    # Optimizer configurations
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW'], default='SGD', help='Optimizer type: SGD or Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0009, help='Weight decay for the optimizer')

    # Scheduler configurations
    parser.add_argument('--scheduler_t_max', type=int, default=40, help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--scheduler_eta_min', type=float, default=0, help='Eta_min for CosineAnnealingLR scheduler')

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = CustomDataset(args.annotation_path, args.data_dir, get_transform(train=True), filter_bbox=True)
    val_dataset = CustomDataset(args.annotation_path, args.data_dir, get_transform(train=False))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = create_model(args.model_name, num_classes=11).to(device)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                lr=args.learning_rate, weight_decay=args.weight_decay)

    # Scheduler Setup
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max, eta_min=args.scheduler_eta_min)

    # Training Mode Selection
    if args.training_mode == 'standard':
        run_standard_training(
            train_data_loader, val_data_loader, model, optimizer, scheduler, device, args.base_dir, args.num_epochs
        )
    elif args.training_mode == 'fold':
        run_fold_training(
            train_dataset, args.model_name, device, args.base_dir, args.num_epochs, args.n_splits, args.batch_size,
            args.learning_rate, args.momentum, args.weight_decay
        )

if __name__ == '__main__':
    main()
