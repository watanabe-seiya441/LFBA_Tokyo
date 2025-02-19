import cv2
import time

# カメラを開く (0 はデフォルトカメラ)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

start_time = time.time()
capture_interval = 1  # 1秒ごとに撮影

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした。")
        break
    
    # 画像を表示
    cv2.imshow("Camera", frame)
    
    # 現在の時間
    current_time = time.time()
    
    # 1秒ごとに画像を保存
    if current_time - start_time >= capture_interval:
        timestamp = int(current_time)
        filename = f"image/captured_image_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"画像を保存しました: {filename}")
        start_time = current_time
    
    # キー入力を待つ (1ms)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' キーで終了
    if key == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
