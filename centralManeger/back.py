# backend/server.py
import cv2, threading, queue, time, base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← 追加
from centralmaneger.camera.camera import Camera
from centralmaneger.camera.cameraApp import capture_latest_frame
import uvicorn

app = FastAPI()

# CORS の許可設定
origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] で全オリジン許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frame_queue: queue.Queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
mode_record = threading.Event()
mode_record.set()  # キャプチャを有効化しておく

# カメラ初期化
camera = Camera(camera_id=0)

# キャプチャスレッド起動
threading.Thread(
    target=capture_latest_frame,
    args=(camera, frame_queue, stop_event, mode_record),
    daemon=True
).start()

@app.get("/api/camera")
def get_camera():
    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        return {"imageUrl": "", "bits": "----"}

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_bytes = buffer.tobytes()
    data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()

    bits = "1101"  # ダミー値
    return {"imageUrl": data_uri, "bits": bits}

if __name__ == "__main__":
    # ローカルのみでよければ 127.0.0.1 に変更
    uvicorn.run(app, host="127.0.0.1", port=8000)
