import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import threading
import time
import queue
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

class ImageClassifier:
    """Image classifier that dynamically loads a model and classifies images."""

    def __init__(self, model_path):
        """
        Initialize the classifier.

        Args:
            model_path (str): Path to the model checkpoint.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """
        Load the model from a checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: Loaded model.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        model_arch = checkpoint.get("arch", self._infer_arch(checkpoint))
        num_classes = checkpoint.get("num_classes", self._infer_num_classes(checkpoint, model_arch))
        
        model = self._initialize_model(model_arch, num_classes)
        state_dict = self._adjust_state_dict_keys(checkpoint.get("state_dict", checkpoint), model)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        return model

    def _infer_arch(self, checkpoint):
        """
        Infer the architecture from the checkpoint.

        Args:
            checkpoint (dict): Loaded checkpoint dictionary.

        Returns:
            str: Model architecture name.
        """
        if "features.0.0.weight" in checkpoint:
            return "mobilenet"
        if "classifier.0.weight" in checkpoint:
            return "vgg"
        raise ValueError("Unknown model architecture in checkpoint.")

    def _infer_num_classes(self, checkpoint, model_arch):
        """
        Infer the number of classes from the checkpoint.

        Args:
            checkpoint (dict): Loaded checkpoint dictionary.
            model_arch (str): Model architecture name.

        Returns:
            int: Number of output classes.
        """
        state_dict = checkpoint.get("state_dict", checkpoint)
        try:
            return state_dict[f"classifier.{3 if model_arch == 'mobilenet' else 6}.weight"].shape[0]
        except KeyError as e:
            raise ValueError(f"Could not infer num_classes from checkpoint. Missing key: {e}")

    def _initialize_model(self, model_arch, num_classes):
        """
        Initialize the model architecture.

        Args:
            model_arch (str): Architecture name.
            num_classes (int): Number of output classes.

        Returns:
            torch.nn.Module: Initialized model.
        """
        if model_arch == "mobilenet":
            return models.mobilenet_v3_small(num_classes=num_classes)
        if model_arch == "vgg":
            return models.vgg16(num_classes=num_classes)
        raise ValueError("Unsupported model type. Use 'mobilenet' or 'vgg'.")

    def _adjust_state_dict_keys(self, state_dict, model):
        """
        Adjust state dictionary keys to match model keys.

        Args:
            state_dict (dict): State dictionary from the checkpoint.
            model (torch.nn.Module): Model instance.

        Returns:
            dict: Adjusted state dictionary.
        """
        model_keys = set(model.state_dict().keys())
        return {k.replace("module.", ""): v for k, v in state_dict.items() if k.replace("module.", "") in model_keys}

    def classify(self, image_data):
        """
        Classify an image and return the predicted class index.

        Args:
            image_data (np.ndarray or str): Image data or path to the image.

        Returns:
            int: Predicted class index.
        """
        image = self._prepare_image(image_data)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def _prepare_image(self, image_data):
        """
        Prepare image data for classification.

        Args:
            image_data (np.ndarray or str): Image data or path to the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data[..., ::-1])
        return self.transform(image_data.convert('RGB')).unsqueeze(0).to(self.device)

def process_images(stop_event, mode_train, frame_queue, write_queue, model_path, classes, classes_queue):
    """
    Process images in a background thread for classification.

    Args:
        stop_event (threading.Event): Event to signal stopping the process.
        mode_train (threading.Event): Event indicating training mode.
        frame_queue (queue.Queue): Queue to receive image frames.
        write_queue (queue.Queue): Queue to send classification results.
        model_path (str): Path to the model checkpoint.
        classes (list): List of class names.
        classes_queue (queue.Queue): Queue to receive class names.
    """
    classifier, previous_prediction, consecutive_count = None, None, 0
    last_model_update = None
    current_state = None

    while not stop_event.is_set():
        classifier, last_model_update, new_classes = _check_and_reload_model(classifier, model_path, last_model_update, classes_queue)
        if new_classes is not None:
            classes = new_classes

        if not classifier:
            # _switch_to_train_mode_if_needed(mode_train)
            continue

        if mode_train.is_set():
            previous_prediction, consecutive_count = None, 0
            time.sleep(1)
            continue

        image_data = frame_queue.get()
        if image_data is None:
            frame_queue.task_done()
            continue

        predicted_class, confidence = _classify_image(classifier, image_data)
        if predicted_class is None or confidence is None:
            logger.warning("[WARNING] Classification returned None. Skipping this frame.")
            continue  # Skip to the next iteration
        logger.debug(f"[DEBUG] Current prediction is {classes[predicted_class]}")

        consecutive_count = consecutive_count + 1 if predicted_class == previous_prediction else 1
        if consecutive_count >= 3 and current_state != classes[predicted_class]:
            write_queue.put(classes[predicted_class])
            logger.info(f"[PROCESS] Stable prediction confirmed: {classes[predicted_class]} (Confidence: {confidence:.2f})")
            current_state = classes[predicted_class]
            time.sleep(5)
        previous_prediction = predicted_class
        frame_queue.task_done()

def _check_and_reload_model(classifier, model_path, last_model_update, classes_queue):
    """
    Check for model updates and reload if necessary.

    Args:
        classifier (ImageClassifier): Current classifier instance.
        model_path (str): Path to the model checkpoint.
        last_model_update (float): Last model update timestamp.
        classes_queue (queue.Queue): Queue to receive class names.

    Returns:
        tuple: (Updated classifier, latest model update timestamp, updated classes or None)
    """
    if os.path.exists(model_path):
        current_update = os.path.getmtime(model_path)
        if not classifier or current_update > (last_model_update or 0):
            logger.info(f"[MODEL] Model update detected. Reloading model from {model_path}.")
            classifier = ImageClassifier(model_path)

            # Reload classes from the queue if available
            if not classes_queue.empty():
                classes = classes_queue.get()
                classes_queue.put(classes)  # Put it back for future access
                logger.info(f"[MODEL] Classes reloaded from queue: {classes}")
            else:
                classes = None
                logger.warning("[WARNING] Classes queue is empty during model reload!")

            return classifier, current_update, classes
    return classifier, last_model_update, None

def _switch_to_train_mode_if_needed(mode_train):
    """
    Switch to training mode if model is unavailable.

    Args:
        mode_train (threading.Event): Training mode event.
    """
    if not mode_train.is_set():
        logger.info("[MODEL] The model has not been generated yet. Switching to train mode.")
        mode_train.set()
    time.sleep(1)

def _classify_image(classifier, image_data):
    """
    Classify an image and return the prediction and confidence.

    Args:
        classifier (ImageClassifier): Classifier instance.
        image_data (np.ndarray or str): Image data.

    Returns:
        tuple: (Predicted class index, confidence score) or (None, None) if classification fails.
    """
    if image_data is None:
        return None, None
    
    try:
        with torch.no_grad():
            transformed_image = classifier._prepare_image(image_data)
            output = classifier.model(transformed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            sorted_indices = np.argsort(probabilities)[::-1]
            top_1_idx, top_2_idx = sorted_indices[:2]
            top_1_confidence = probabilities[top_1_idx]
            top_2_confidence = probabilities[top_2_idx]

            # Skip if confidence difference is too small
            if abs(top_1_confidence - top_2_confidence) < 0.1:
                logger.warning(f"[SKIP] Confidence too close: Top-1={top_1_confidence:.2f}, Top-2={top_2_confidence:.2f}")
                return None, None

            return top_1_idx, top_1_confidence
    except Exception as e:
        logger.error(f"[ERROR] Classification failed: {e}")
        return None, None

