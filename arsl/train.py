import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from arsl.data import get_dataloaders
from arsl.models.baseline_model import BaselineModel 
from arsl.models.pretrained_model import PretrainedModel
from arsl.utils import get_device, save_checkpoint, load_checkpoint


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        accuracy = correct / size

        if batch % 100 == 0:
            loss, current = loss.item(), batch * args.batch_size + len(X)
            print(f"loss: {loss:>7f}  accuracy: {accuracy * 100:>0.1f}% [{current:>5d}/{size:>5d}]")

        del X, y, pred, loss

        if device == 'cuda':
            torch.cuda.empty_cache()


def test_loop(dataloader, model, loss_fn, device):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            del X, y, pred

            if device == 'cuda':
                torch.cuda.empty_cache()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main(args):
    device = get_device()
    print(f"[INFO] Running on device: {device}")

    print(f"[INFO] Creating dataset loaders")
    train_loader, test_loader = get_dataloaders(
        args.root_dir,
        args.training_mode,
        args.model_type,
        args.labels_path,
        args.batch_size
    )

    print(f"[INFO] Creating {args.model_type} model")
    if args.model_type == 'baseline':
        model = BaselineModel().to(device)
    elif args.model_type == 'pretrained':
        model = PretrainedModel().to(device)
    else:
        raise ValueError(f'Model type "{args.model_type}" is not supported.'
                         ' Supported types are: ["baseline", "pretrained"].')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    epochs_res = [os.path.join(args.checkpoints_dir, res) for res in sorted(os.listdir(args.checkpoints_dir))]

    if epochs_res:
        start_epoch = load_checkpoint(model, optimizer, epochs_res[-1]) + 1
        print(f"[INFO] Loaded checkpoint from {epochs_res[-1]}")
    else:
        start_epoch = 0
        print("[INFO] No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        save_checkpoint(model, optimizer, epoch, args.checkpoints_dir)

        test_loop(test_loader, model, loss_fn, device)
        scheduler.step()


    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition Model")

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for dataloaders')
    parser.add_argument('--step_size', type=int, default=5, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.01, help='Scheduler learning rate decay factor')

    parser.add_argument('--root_dir', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to labels file')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Directory where checkpoints will be saved')
    parser.add_argument('--model_type', type=str, required=True, help='Model type. Supported types are: ["baseline", "pretrained"]')
    parser.add_argument('--training_mode', type=str, required=True, choices=['dependent', 'independent'], help='Choose "dependent" or "independent" for signer dependency')

    args = parser.parse_args()

    main(args)
