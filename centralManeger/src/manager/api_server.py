"""
api_server.py
=============
FastAPIサーバー機能を提供するモジュール
"""

import threading
import queue
import logging
import cv2
import base64
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class APIServer:
    """FastAPIサーバーを管理するクラス"""
    
    def __init__(self, thread_manager):
        self.thread_manager = thread_manager
        self.app = FastAPI(title="Inference API")
        
        # CORS設定
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # ルートの設定
        self._setup_routes()
    
    def _setup_routes(self):
        """APIルートを設定"""
        
        @self.app.get("/api/inference")
        def api_inference():
            """最新フレームと推論結果を返すAPI"""
            logger.debug(f"[API] Request received, bits: {self.thread_manager.latest_inference_bits}")
            
            try:
                frame = self.thread_manager.api_frame_queue.get_nowait()
                logger.debug("[API] Frame retrieved from queue")
            except queue.Empty:
                logger.debug("[API] No frame available")
                return {"imageUrl": "", "bits": self.thread_manager.latest_inference_bits}
                
            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                logger.warning("[API] Failed to encode frame")
                return {"imageUrl": "", "bits": self.thread_manager.latest_inference_bits}
                
            data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
            logger.debug(f"[API] Response sent, bits: {self.thread_manager.latest_inference_bits}")
            return {"imageUrl": data_uri, "bits": self.thread_manager.latest_inference_bits}
        
        @self.app.get("/api/status")
        def api_status():
            """システム状態を返すAPI"""
            return {
                "mode_train": self.thread_manager.mode_train.is_set(),
                "mode_record": self.thread_manager.mode_record.is_set(),
                "frame_queue_size": self.thread_manager.frame_queue.qsize(),
                "api_frame_queue_size": self.thread_manager.api_frame_queue.qsize(),
                "latest_bits": self.thread_manager.latest_inference_bits
            }
    
    def start_server(self, host="127.0.0.1", port=8000):
        """サーバーを別スレッドで開始"""
        api_thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=host, port=port, log_level="info"),
            daemon=True
        )
        api_thread.start()
        logger.info(f"[API] Server started on http://{host}:{port}")
        return api_thread 