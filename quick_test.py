from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/isyara_ai/terbisa_translator_50ep_yolo11/weights/best.pt")

cap = cv2.VideoCapture(0)

print(" Webcam started! Press 'q' to quit testing.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    results = model(frame, conf=0.5)

    
    annotated_frame = results[0].plot()

    
    cv2.imshow("Isyara BISINDO Translator - Live Test", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()