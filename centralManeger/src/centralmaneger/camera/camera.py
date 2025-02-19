import cv2

class Camera:
    def __init__(self, camera_id: int = 0, capture_interval: int = 1):
        """
        Initializes the camera with the specified camera ID and capture interval.

        Args:
            camera_id (int): The ID of the camera to use.
            capture_interval (int): Interval in seconds between captures.
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open the camera.")
        self.capture_interval = capture_interval

    def capture_frame(self):
        """Captures a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture a frame.")
            return None
        return frame

    def save_image(self, filename, frame):
        """Saves the captured frame as an image file."""
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    def release(self):
        """Releases the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()
