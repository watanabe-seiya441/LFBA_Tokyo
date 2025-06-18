"""
back.py
========
【リファクタリング済み】システム統合管理エントリーポイント

使用方法:
    poetry run python back.py

機能:
- システム全体のライフサイクル管理
- ハードウェア初期化（シリアル通信・カメラ）
- マネージャー初期化（スレッド・API）
- 段階的起動とクリーンアップ
"""

import sys
import os

# 各種マネージャーをインポート
from src.manager.config_manager import load_config
from src.manager.logger_setup import setup_logging, get_logger
from src.manager.thread_manager import ThreadManager
from src.manager.api_server import APIServer
from tool.camera.camera import Camera
from tool.serial.communication import SerialCommunication


class SystemManager:
    """システム全体を管理するメインクラス"""
    
    def __init__(self, config_path="config.toml"):
        """
        システムマネージャーを初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        # ログ設定
        setup_logging()
        self.logger = get_logger(__name__)
        self.logger.info("[SYSTEM] Initializing system...")
        
        # 設定の読み込み
        self.config = load_config(config_path)
        self.logger.info("[SYSTEM] Configuration loaded successfully")
        
        # コンポーネントの初期化
        self.serial_comm = None
        self.camera = None
        self.thread_manager = None
        self.api_server = None
        
        self.logger.info("[SYSTEM] System manager initialized")
    
    def initialize_hardware(self):
        """ハードウェアコンポーネントを初期化"""
        try:
            # シリアル通信の初期化
            self.serial_comm = SerialCommunication(
                self.config.serial.port, 
                self.config.serial.baudrate
            )
            self.logger.info(f"[SYSTEM] Serial communication initialized on {self.config.serial.port}")
            
            # カメラの初期化
            self.camera = Camera(
                camera_id=self.config.camera.cameraID, 
                capture_interval=1
            )
            self.logger.info(f"[SYSTEM] Camera initialized with ID {self.config.camera.cameraID}")
            
        except Exception as e:
            self.logger.error(f"[SYSTEM] Failed to initialize hardware: {e}")
            raise
    
    def initialize_managers(self):
        """マネージャークラスを初期化"""
        try:
            # スレッドマネージャーの初期化
            self.thread_manager = ThreadManager(self.config)
            self.logger.info("[SYSTEM] Thread manager initialized")
            
            # APIサーバーの初期化
            self.api_server = APIServer(self.thread_manager)
            self.logger.info("[SYSTEM] API server initialized")
            
        except Exception as e:
            self.logger.error(f"[SYSTEM] Failed to initialize managers: {e}")
            raise
    
    def start(self):
        """システムを開始"""
        try:
            self.logger.info("[SYSTEM] Starting system...")
            
            # 段階的初期化
            self.initialize_hardware()
            self.initialize_managers()
            
            # すべてのスレッドを開始
            self.thread_manager.start_all_threads(self.serial_comm, self.camera)
            
            # APIサーバーを開始
            self.api_server.start_server()
            
            self.logger.info("[SYSTEM] System started successfully")
            
            # メインスレッドの終了を待つ
            self.thread_manager.wait_for_completion()
            
        except KeyboardInterrupt:
            self.logger.info("[SYSTEM] Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"[SYSTEM] System error: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """システムをシャットダウン"""
        self.logger.info("[SYSTEM] Shutting down system...")
        
        if self.thread_manager:
            self.thread_manager.cleanup(self.serial_comm, self.camera)
        
        self.logger.info("[SYSTEM] System shutdown completed")


def main():
    """メイン関数"""
    print("[INFO] Start system...")
    print("[INFO] All packages loaded successfully.")
    
    try:
        system = SystemManager()
        system.start()
    except Exception as e:
        print(f"[ERROR] System failed to start: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


