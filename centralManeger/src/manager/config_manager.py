"""
config_manager.py
================
設定ファイルの読み込みと管理を行うモジュール
"""

import os
import tomllib
from dataclasses import dataclass
from typing import List


@dataclass
class SerialConfig:
    """シリアル通信の設定"""
    port: str
    baudrate: int


@dataclass
class CameraConfig:
    """カメラの設定"""
    cameraID: int

@dataclass
class DirectoryConfig:
    """ディレクトリの設定"""
    dataset_dir: str
    model_dir: str
    image_dir: str


@dataclass
class ModelConfig:
    """モデルの設定"""
    classes: List[str]
    name: str
    arch: str
    is_update: bool


@dataclass
class MonitoringConfig:
    """監視の設定"""
    THRESHOLD: int
    CHECK_INTERVAL: int

@dataclass
class HyperParameters:
    """ハイパーパラメータの設定"""
    batch_size: int
    epochs: int
    img_size: int
    learning_rate: float

@dataclass
class GPUConfig:
    """GPU設定"""
    gpu_index: int


@dataclass
class SystemConfig:
    """システム全体の設定を統合"""
    serial: SerialConfig
    camera: CameraConfig
    model: ModelConfig
    hyperparameters: HyperParameters
    directory: DirectoryConfig
    monitoring: MonitoringConfig
    gpu: GPUConfig
    
    @property
    def model_path(self) -> str:
        """モデルファイルのパスを返す"""
        return os.path.join(self.directory.model_dir, self.model.name)


def load_config(config_path: str = "config.toml") -> SystemConfig:
    """
    設定ファイルを読み込んでSystemConfigオブジェクトを返す
    
    Args:
        config_path (str): 設定ファイルのパス
        
    Returns:
        SystemConfig: 設定オブジェクト
    """
    with open(config_path, "rb") as config_file:
        config = tomllib.load(config_file)
    
    return SystemConfig(
        serial=SerialConfig(**config["serial"]),
        camera=CameraConfig(**config["camera"]),
        model=ModelConfig(**config["model"]),
        hyperparameters=HyperParameters(**config["hyperparameters"]),
        directory=DirectoryConfig(**config["directory"]),
        monitoring=MonitoringConfig(**config["monitoring"]),
        gpu=GPUConfig(**config["gpu"])
    ) 