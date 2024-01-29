from ultralytics import YOLO

device = "cuda"
print(f"Using device: {device}")

base_model = "yolov8l.pt"
model = YOLO(base_model).to(device)

if __name__ == "__main__":
    model.train()
    data="datasets\pantry-pal"
    imgsz=416,
    epochs=300,
    batch=8,
    name="yolov8l-pantry-pal"