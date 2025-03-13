import cv2

class Camera:
    """
    A class to handle camera operations using OpenCV.

    Attributes:
        cap (cv2.VideoCapture): OpenCV video capture object.
        capture_interval (int): Interval in seconds between captures.
    """

    def __init__(self, camera_id: int = 0, capture_interval: int = 1):
        """
        Initializes the camera with the specified camera ID and capture interval.

        Args:
            camera_id (int): The ID of the camera to use.
            capture_interval (int): Interval in seconds between captures.

        Raises:
            ValueError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open the camera.")
        self.capture_interval = capture_interval

    def capture_frame(self):
        """
        Captures a frame from the camera.

        Returns:
            numpy.ndarray or None: The captured frame, or None if capture fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture a frame.")
            return None
        return frame

    def save_image(self, filename: str, frame) -> None:
        """
        Saves the captured frame as an image file.

        Args:
            filename (str): Path where the image will be saved.
            frame (numpy.ndarray): Frame to be saved.
        """
        cv2.imwrite(filename, frame)

    def release(self) -> None:
        """
        Releases the camera resource and closes any OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()
