import cv2

# カメラを開く (0 はデフォルトカメラ)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした。")
        break
    
    # 画像を表示
    cv2.imshow("Camera", frame)
    
    # キー入力を待つ (1ms)
    key = cv2.waitKey(1) & 0xFF
    
    # スペースキーで画像を保存
    if key == ord(' '):
        cv2.imwrite("captured_image.jpg", frame)
        print("画像を保存しました: captured_image.jpg")
    
    # 'q' キーで終了
    elif key == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
