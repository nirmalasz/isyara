class YoloRoiDetector:
    def __init__(self):
        self.available = False
        self.model = None
        try:
            from ultralytics import YOLO  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def detect(self, frame):
        if not self.available:
            return {"status": "bypassed", "regions": []}
        return {"status": "not_configured", "regions": []}
