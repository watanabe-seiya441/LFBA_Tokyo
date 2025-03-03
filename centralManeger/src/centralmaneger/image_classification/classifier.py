import torch
import torchvision.transforms as transforms
from PIL import Image
import threading
import queue
import logging

# ロギング設定
logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    画像分類を行うクラス。
    指定されたモデルをロードし、画像の分類を行う。
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # モデルのロード（OrderedDict の場合は構築）
        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            from torchvision import models
            self.model = models.mobilenet_v3_small(num_classes=4)  # 必要に応じてモデル変更
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 画像の前処理設定
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify(self, image_path):
        """ 画像を分類し、クラスインデックスを返す """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        
        return predicted.item()

def predict_image(image_path, model_path):
    """ 指定した画像の分類を行い、結果を返す """
    classifier = ImageClassifier(model_path)
    return classifier.classify(image_path)

def process_images(stop_event: threading.Event, mode_train: threading.Event, 
                   frame_queue: queue.Queue, write_queue: queue.Queue, model_path, classes: list):
    """
    画像を継続的に分類するスレッド処理。
    - 訓練モード中は処理をスキップ。
    - 最新のフレームを保持し、それを分類。
    - 結果を write_queue に格納。
    """
    classifier = ImageClassifier(model_path)
    latest_frame = None
    
    while not stop_event.is_set():
        if mode_train.is_set():
            continue  # 訓練モード時は処理をスキップ
        
        image_path = frame_queue.get()
        if image_path is None:
            continue  # None が来たらスキップ
        
        latest_frame = image_path  # 最新のフレームを更新
        
        if latest_frame:
            prediction = classifier.classify(latest_frame)
            write_queue.put(classes[prediction])
        
        frame_queue.task_done()
        logger.debug(f"[PROCESS] Processed {latest_frame}, Prediction: {classes[prediction]}")
