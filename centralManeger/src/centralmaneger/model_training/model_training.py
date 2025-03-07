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

# -------------------------
# Set random seed for reproducibility
def set_seed(seed=57):
    """
    Fixes random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Load configuration from config.toml
def load_config(config_path="config.toml"):
    """
    Loads configuration settings from config.toml.
    """
    with open(config_path, "rb") as f:
        return tomllib.load(f)

# -------------------------
# Prepare dataset & get number of classes
def prepare_data(data_dir, img_size, batch_size):
    """
    Loads dataset, prepares DataLoaders, and retrieves the number of classes.

    Args:
        data_dir (str): Path to dataset directory.
        img_size (int): Image size for resizing.
        batch_size (int): Batch size for training.

    Returns:
        tuple: (dataloaders, image_datasets, num_classes)
    """
    try:
        # Get system's available CPU cores
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(6, cpu_count)

        logger.info(f"Using {num_workers} workers for DataLoader (CPU cores available: {cpu_count})")

        # Define data transformations
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

        num_classes = len(image_datasets['train'].classes)  # Get number of classes

        logger.info(f"Dataset loaded successfully with {num_classes} classes.")
        return dataloaders, image_datasets, num_classes

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# -------------------------
# Initialize model (Load if exists, create new otherwise)
def initialize_model(num_classes, result_dir, device):
    """
    Initializes the MobileNetV3-Small model and loads existing weights if available.
    """
    try:
        model_path = os.path.join(result_dir, "mobilenetv3_small_latest.pth")

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model = model.to(device)

        if os.path.exists(model_path):
            logger.info(f"Loading existing model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model, model_path, True  # Fine-tuning
        else:
            logger.info("No existing model found. Starting new training.")
            return model, model_path, False  # New training

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

# -------------------------
# Train & Validate Model
def train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs, stop_event):
    """
    Trains and validates the model. If stop_event is set, training stops immediately,
    saves the model, and exits safely.
    
    Args:
        model (torch.nn.Module): The model to train.
        dataloaders (dict): Dictionary containing DataLoaders.
        image_datasets (dict): Dictionary containing datasets.
        device (torch.device): Target device (CPU/GPU).
        model_path (str): Path to save the trained model.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of epochs to train.
        stop_event (threading.Event): Event flag to trigger early termination.
    """
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        logger.info("Training started.")

        for epoch in range(epochs):
            if stop_event.is_set():
                logger.warning("Stop event detected! Saving model and terminating training immediately.")
                torch.save(model.state_dict(), model_path)
                logger.info(f"Model saved at {model_path}")
                return  # **Exit training immediately**

            logger.info(f"Epoch {epoch+1}/{epochs} started.")

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()

                running_loss, running_corrects = 0.0, 0
                dataloader = dataloaders[phase]
                total_batches = len(dataloader)

                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    if stop_event.is_set():
                        logger.warning("Stop event detected! Saving model and terminating training immediately.")
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"Model saved at {model_path}")
                        return  # **Exit training immediately**

                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if batch_idx % 10 == 0:
                        logger.info(f"{phase} Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f}")

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            torch.save(model.state_dict(), model_path)
            logger.info(f"Latest model saved: {model_path}")

            scheduler.step(epoch_loss)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        torch.save(model.state_dict(), model_path)  # Save model before exiting
        logger.info(f"Model saved due to error at {model_path}")
        raise

# -------------------------
# Train Controller Thread
def train_controller(stop_event: threading.Event, start_train: queue.Queue):
    """
    Watches for training start signals and runs training when triggered.
    Training will be stopped immediately if stop_event is set.
    
    Args:
        stop_event (threading.Event): Event to stop training.
        start_train (queue.Queue): Queue receiving "start" signals.
    """
    config = load_config()
    set_seed()

    batch_size = config["hyperparameters"]["batch_size"]
    epochs = config["hyperparameters"]["epochs"]
    img_size = config["hyperparameters"]["img_size"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    data_dir = config["directory"]["data_dir"]
    result_dir = config["directory"]["result_dir"]
    gpu = config["gpu"]["gpu_index"]

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    while not stop_event.is_set():
        if start_train.empty():
            continue

        task = start_train.get()
        if task == "start":
            logger.info("Training triggered.")
            dataloaders, image_datasets, num_classes = prepare_data(data_dir, img_size, batch_size)
            model, model_path, is_finetune = initialize_model(num_classes, result_dir, device)

            if is_finetune:
                learning_rate *= 0.1
                logger.info("Fine-tuning detected. Reducing learning rate.")

            train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs, stop_event)

            logger.info("Training completed.")
            start_train.task_done()

