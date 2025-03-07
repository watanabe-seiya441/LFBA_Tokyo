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
from tqdm import tqdm

# Logging setup
logger = logging.getLogger(__name__)

# -------------------------
# 乱数のシード固定
def set_seed(seed=57):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# 設定ファイル読み込み
def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        return tomllib.load(f)

# -------------------------
# データ準備
def prepare_data(data_dir, img_size, batch_size, num_workers=8):
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

    logging.info("データセットのロード完了")
    logging.info(f"学習データ: {len(image_datasets['train'])}枚")
    logging.info(f"検証データ: {len(image_datasets['val'])}枚")

    return dataloaders, image_datasets

# -------------------------
# モデルの初期化 & ロード
def initialize_model(num_classes, result_dir, device):
    model_path = os.path.join(result_dir, "mobilenetv3_small_latest.pth")

    # MobileNetV3-Small の初期化
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)

    # 既存モデルがあればロード
    if os.path.exists(model_path):
        logging.info(f"既存モデルをロード: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, model_path, True  # 継続学習
    else:
        logging.info("既存モデルなし、新規学習を開始")
        return model, model_path, False  # 新規学習

# -------------------------
# モデルの学習と検証
def train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    logging.info("学習開始")

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs} 開始")
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0.0, 0
            dataloader = dataloaders[phase]
            progress_bar = tqdm(dataloader, desc=phase)

            for inputs, labels in progress_bar:
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
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # **最後のエポックのモデルのみを保存**
        torch.save(model.state_dict(), model_path)
        print(f"Latest model saved: {model_path}")

        scheduler.step(epoch_loss)

# -------------------------
# メイン関数
def main():
    # 設定ファイル読み込み
    config = load_config()

    # 乱数のシード固定
    set_seed()

    num_classes = config["hyperparameters"]["num_classes"]
    batch_size = config["hyperparameters"]["batch_size"]
    epochs = config["hyperparameters"]["epochs"]
    img_size = config["hyperparameters"]["img_size"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    data_dir = config["directory"]["data_dir"]
    result_dir = config["directory"]["result_dir"]
    gpu = config["gpu"]["gpu_index"]

    # 結果保存ディレクトリの作成
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%dT%H%M%S")
    result_dir = os.path.join(result_dir, formatted_time)

    # ログ設定
    setup_logging(result_dir)

    logging.info(f"データディレクトリ: {data_dir}")
    logging.info(f"結果保存ディレクトリ: {result_dir}")
    logging.info(f"使用GPU: {gpu}")

    # データローダーの準備
    dataloaders, image_datasets = prepare_data(data_dir, img_size, batch_size)

    # デバイス設定
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用デバイス: {device}")

    # モデルの初期化（既存モデルがあればロード）
    model, model_path, is_finetune = initialize_model(num_classes, result_dir, device)

    # 継続学習の場合、学習率を小さく
    if is_finetune:
        learning_rate *= 0.1

    # 学習開始
    train_model(model, dataloaders, image_datasets, device, model_path, learning_rate, epochs)
