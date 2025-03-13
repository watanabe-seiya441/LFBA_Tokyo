import os
import random
import logging
import numpy as np
import shutil
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
import multiprocessing
import threading
import queue

# -------------------------
# Logging setup
logger = logging.getLogger(__name__)


def set_seed(seed=57):
    """Fixes random seed for reproducibility.

    Args:
        seed (int): Seed value. Defaults to 57.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(data_dir, img_size, batch_size):
    """Loads dataset, prepares DataLoaders, and retrieves the number of classes.

    Args:
        data_dir (str): Path to dataset directory.
        img_size (int): Image size for resizing.
        batch_size (int): Batch size for training.

    Returns:
        tuple: (dataloaders, image_datasets, num_classes)
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(6, cpu_count)
        logger.info(f"[TRAIN] Using {num_workers} workers for DataLoader (CPU cores available: {cpu_count})")

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, transform=data_transforms['val'])
        }

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        }

        num_classes = len(image_datasets['train'].classes)
        logger.info(f"[TRAIN] Dataset loaded successfully with {num_classes} classes.")
        return dataloaders, image_datasets, num_classes

    except Exception as e:
        logger.error(f"[Error] loading dataset: {e}")
        raise


def initialize_model(num_classes, model_dir, device):
    """Initializes the MobileNetV3-Small model and loads existing weights if available.

    Args:
        num_classes (int): Number of output classes.
        model_dir (str): Directory to save model.
        device (torch.device): Computation device.

    Returns:
        tuple: (model, model_path, is_finetune)
    """
    try:
        model_path = os.path.join(model_dir, "mobilenetv3_small_latest.pth")

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model = model.to(device)

        if os.path.exists(model_path):
            logger.info(f"[TRAIN] Loading existing model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model, model_path, True
        else:
            logger.info("[TRAIN] No existing model found. Starting new training.")
            return model, model_path, False

    except Exception as e:
        logger.error(f"[Error] initializing model: {e}")
        raise


def train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs, stop_event):
    """Trains and validates the model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloaders (dict): Data loaders for training and validation.
        image_datasets (dict): Image datasets.
        device (torch.device): Device to run the model on.
        model_path (str): Path to save the model.
        learning_rate (float): Learning rate.
        epochs (int): Number of epochs.
        stop_event (threading.Event): Event to signal stopping.
    """
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        logger.info("[TRAIN] Training started.")

        for epoch in range(epochs):
            logger.info(f"[TRAIN] Epoch {epoch+1}/{epochs} started.")

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()

                running_loss, running_corrects = 0.0, 0
                total_batches = len(dataloaders[phase])

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    if stop_event.is_set():
                        logger.warning("[STOP] Stop event detected during batch! Saving model and terminating training.")
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"[TRAIN] Model saved at {model_path}")
                        return

                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    if batch_idx % 10 == 0:
                        logger.info(f"[TRAIN] {phase} Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f}")

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                logger.info(f"[TRAIN] {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            torch.save(model.state_dict(), model_path)
            logger.info(f"[TRAIN] Model saved at {model_path}")
            scheduler.step(epoch_loss)

    except Exception as e:
        logger.error(f"[Error] during training: {e}")
        torch.save(model.state_dict(), model_path)
        logger.info(f"[TRAIN] Model saved due to error at {model_path}")
        raise



def train_controller(stop_event, start_train, batch_size, epochs, img_size, learning_rate, data_dir, model_dir, gpu):
    """Watches for training start signals and runs training when triggered.

    Args:
        stop_event (threading.Event): Event to stop training.
        start_train (queue.Queue): Queue receiving 'start' signals.
    """
    set_seed()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"[TRAIN] Using device: {device}")

    while not stop_event.is_set():
        if start_train.empty():
            continue

        task = start_train.get()
        if task == "start":
            logger.info("[TRAIN] Training triggered.")
            dataloaders, image_datasets, num_classes = prepare_data(data_dir, img_size, batch_size)
            model, model_path, is_finetune = initialize_model(num_classes, model_dir, device)

            if is_finetune:
                learning_rate *= 0.1
                logger.info("[TRAIN] Fine-tuning detected. Reducing learning rate.")

            train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs, stop_event)

            logger.info("[TRAIN] Training completed.")
            start_train.task_done()