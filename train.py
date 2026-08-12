from roboflow import Roboflow
from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    rf = Roboflow(api_key="0dvUVaFLO4ckwVqPqhio")
    project = rf.workspace("jonathan-toga-sihotang").project("terbisa")
    
    dataset = project.version(1).download("yolov11")
    
    model = YOLO("yolo11n.pt")

    
    print("Starting training...")
    
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=8,
        device=0, 
        project="isyara_ai",
        name="terbisa_translator_50ep_yolo11"
    )

    print("\nTraining Complete!")
    print("Your trained model weights are saved at: isyara_ai/terbisa_translator/weights/best.pt")

if __name__ == '__main__':
    main()