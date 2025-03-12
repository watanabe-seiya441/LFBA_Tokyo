import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import threading
import time
import queue
import sys
import logging
import os

# ロギング設定
logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine model architecture
        if "arch" in checkpoint:
            model_arch = checkpoint["arch"]
        elif "features.0.0.weight" in checkpoint:
            model_arch = "mobilenet"
        elif "classifier.0.weight" in checkpoint:
            model_arch = "vgg"
        else:
            raise ValueError("Unknown model architecture in checkpoint.")

        self.model_type = model_arch.lower()

        # Load state_dict
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Dynamically determine num_classes
        num_classes = checkpoint.get("num_classes")
        if num_classes is None:
            try:
                if self.model_type == "mobilenet":
                    # Get from final layer weight shape
                    num_classes = state_dict["classifier.3.weight"].shape[0]
                elif self.model_type == "vgg":
                    num_classes = state_dict["classifier.6.weight"].shape[0]
                else:
                    raise ValueError("Unsupported model type for inferring num_classes.")
            except KeyError as e:
                raise ValueError(f"Could not infer num_classes from checkpoint. Missing key: {e}")

        # Load the correct model
        if self.model_type == "mobilenet":
            self.model = models.mobilenet_v3_small(num_classes=num_classes)
        elif self.model_type == "vgg":
            self.model = models.vgg16(num_classes=num_classes)
        else:
            raise ValueError("Unsupported model type. Use 'mobilenet' or 'vgg'.")

        # Adjust state_dict keys if necessary (remove 'module.' if present)
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        if model_keys != checkpoint_keys:
            print("[WARNING] Model and checkpoint keys do not match. Adjusting keys...")
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k.replace("module.", "") in model_keys}
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    def classify(self, image_data):
        """ 画像を分類し、クラスインデックスを返す """
        if isinstance(image_data, np.ndarray):
            # OpenCVの画像データはBGRなので、RGBに変換
            image = Image.fromarray(image_data[..., ::-1]).convert('RGB')
        else:
            image = Image.open(image_data).convert('RGB')

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

def predict_image(image_data, model_path):
    """ 指定した画像の分類を行い、結果を返す """
    classifier = ImageClassifier(model_path)
    return classifier.classify(image_data)

def process_images(stop_event: threading.Event, 
                   mode_train: threading.Event, 
                   frame_queue: queue.Queue, 
                   write_queue: queue.Queue, 
                   model_path, 
                   classes: list):
    """
    Continuously classifies images in a separate thread.
    - Skips classification if in training mode.
    - Reloads the model automatically if the file is updated or created.
    - Uses stabilization methods to prevent sending unstable predictions.
    - Logs every prediction result.
    """

    classifier = None
    latest_frame = None
    previous_prediction = None
    consecutive_count = 0

    # Track the last modified timestamp of the model file
    last_model_update = None

    while not stop_event.is_set():
        # Check if model exists and load if it's not already loaded
        if classifier is None and os.path.exists(model_path):
            try:
                classifier = ImageClassifier(model_path)
                last_model_update = os.path.getmtime(model_path)
                logger.info(f"[INFO] Model loaded from {model_path}.")
            except ValueError as e:
                logger.info("[INFO] Failed to load model. Continuing in train mode.")
                mode_train.set()
                time.sleep(1)
                continue

        # Check if model file has been updated
        if classifier and os.path.exists(model_path):
            current_model_update = os.path.getmtime(model_path)
            if last_model_update is None or current_model_update > last_model_update:
                logger.info(f"[INFO] Model update detected. Reloading model from {model_path}.")
                classifier = ImageClassifier(model_path)
                last_model_update = current_model_update

        # If model is still not available, continue in train mode
        if classifier is None:
            if not mode_train.is_set():
                logger.info("[INFO] The model has not been generated yet. Switching to train mode.")
                mode_train.set()
            time.sleep(1)
            continue  # Skip until model is available

        # Skip processing if in training mode
        if mode_train.is_set():
            previous_prediction = None
            consecutive_count = 0
            time.sleep(1)
            continue

        # Get image data from the queue
        image_data = frame_queue.get()
        if image_data is None:
            frame_queue.task_done()
            continue

        # Convert numpy.ndarray to PIL.Image if necessary
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data[..., ::-1])  # Convert BGR to RGB

        latest_frame = image_data

        if latest_frame is not None:
            with torch.no_grad():
                transformed_image = classifier.transform(latest_frame).unsqueeze(0).to(classifier.device)
                output = classifier.model(transformed_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                predicted_class = np.argmax(probabilities)

            # Sort probabilities & Get Top-2
            sorted_probs = np.sort(probabilities)[::-1]
            top_1, top_2 = sorted_probs[:2]

            # Log the current prediction result
            logger.info(f"[PREDICT] Current prediction: {classes[predicted_class]} (Confidence: {top_1:.2f})")

            # Skip if probabilities are too close (uncertain prediction)
            if abs(top_1 - top_2) < 0.1:
                logger.info(f"[SKIP] Close probability detected: {classes[predicted_class]} (Top-1: {top_1:.2f}, Top-2: {top_2:.2f})")
                continue

            # Count consecutive identical predictions
            if predicted_class == previous_prediction:
                consecutive_count += 1
            else:
                consecutive_count = 1

            logger.info(f"[INFO] Consecutive count: {consecutive_count}")

            # Only send result if the same prediction appears 3 times consecutively
            if consecutive_count >= 3:
                write_queue.put(classes[predicted_class])
                logger.info(f"[PROCESS] Stable prediction confirmed: {classes[predicted_class]} (Confidence: {top_1:.2f})")
                time.sleep(5)

            previous_prediction = predicted_class

        frame_queue.task_done()
