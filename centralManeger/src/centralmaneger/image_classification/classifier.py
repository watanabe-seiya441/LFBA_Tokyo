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

# ロギング設定
logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    A class for image classification.
    Loads the specified model (MobileNetV3 or VGG) and performs classification.
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine model type
        if "arch" in checkpoint:
            model_arch = checkpoint["arch"]
        elif "features.0.0.weight" in checkpoint:
            model_arch = "mobilenet"
        elif "classifier.0.weight" in checkpoint:
            model_arch = "vgg"
        else:
            raise ValueError("Unknown model architecture in checkpoint.")

        self.model_type = model_arch.lower()

        # Load correct model
        if self.model_type == "mobilenet":
            self.model = models.mobilenet_v3_small(num_classes=4)
        elif self.model_type == "vgg":
            self.model = models.vgg16(num_classes=4)
        else:
            raise ValueError("Unsupported model type. Use 'mobilenet' or 'vgg'.")

        # Extract state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            return
        else:
            raise ValueError("Invalid checkpoint format")

        # Adjust key names if necessary
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

def process_images(stop_event: threading.Event, mode_train: threading.Event, 
                   frame_queue: queue.Queue, write_queue: queue.Queue, model_path, classes: list):
    """
    画像を継続的に分類するスレッド処理。
    - 訓練モード中は処理をスキップ。
    - 最新のフレームを保持し、それを分類。
    - 結果を write_queue に格納（過去のpredictionと異なる場合のみ）。
    """
    classifier = ImageClassifier(model_path)
    latest_frame = None
    previous_prediction = None  # 過去の予測結果を保存

    while not stop_event.is_set():
        if mode_train.is_set():
            previous_prediction = None
            continue  # 訓練モード時は処理をスキップ
        
        image_data = frame_queue.get()
        if image_data is None:
            continue  # None が来たらスキップ
        
        latest_frame = image_data  # 最新のフレームを更新

        if latest_frame is not None:
            prediction = classifier.classify(latest_frame)

            # 過去の予測結果と異なる場合のみ write_queue に格納
            if prediction != previous_prediction:
                write_queue.put(classes[prediction])
                previous_prediction = prediction  # 過去の予測を更新
                logger.info(f"[PROCESS] New prediction: {classes[prediction]}")
                time.sleep(5)  # 5秒待機

        frame_queue.task_done()
