from ultralytics import YOLO
import os

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)
results = model.predict(source=r"videos\The Best Corgi Compilation On The Internet 2023 Funny and Sassy Moments @FurryTails.mp4", show=True, stream=True)
