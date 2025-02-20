from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from threading import Event, Thread
import cv2
import queue
import time
import serial

app = FastAPI()

# 静的ファイルの設定（HTMLやJavaScriptを配信）
app.mount("/static", StaticFiles(directory="static"), name="static")

# グローバル変数
stop_event = Event()
frame_queue = queue.Queue(maxsize=1)
label_queue = queue.Queue(maxsize=1)

# カメラ設定
camera = cv2.VideoCapture(0)

# シリアル通信設定
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

def capture_frames():
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            continue
        if not frame_queue.empty():
            frame_queue.get()
        frame_queue.put(frame)
        time.sleep(0.1)  # フレームレートの調整

def read_serial():
    while not stop_event.is_set():
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            if not label_queue.empty():
                label_queue.get()
            label_queue.put(data)
            print(f"Received: {data}")

def generate_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        time.sleep(0.1)

@app.on_event("startup")
def startup_event():
    Thread(target=capture_frames, daemon=True).start()
    Thread(target=read_serial, daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    stop_event.set()
    camera.release()
    ser.close()

@app.get("/start")
def start_system():
    stop_event.clear()
    startup_event()
    return JSONResponse(content={"message": "System started"})

@app.get("/stop")
def stop_system():
    stop_event.set()
    return JSONResponse(content={"message": "System stopped"})

@app.get("/camera")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/label")
def get_label():
    if not label_queue.empty():
        label = label_queue.get()
        return JSONResponse(content={"label": label})
    return JSONResponse(content={"label": "No data"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if not label_queue.empty():
                label = label_queue.get()
                await websocket.send_json({"label": label})
            time.sleep(1)
    except WebSocketDisconnect:
        pass
