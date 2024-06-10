import os
from ultralytics import YOLO
import cv2


#VIDEOS_DIR = os.path.join('.')
#video_path = os.path.join(VIDEOS_DIR, r'videos\Corgi Are The Best - CUTEST Compilation.mp4')
#video_path_out = '{}_out.mp4'.format(r'videos\The Best Corgi Compilation On The Internet 2023 Funny and Sassy Moments @FurryTails.mp4')

cap = cv2.VideoCapture(r'videos\35 Corgis To Get You Through Your Day.mp4')
ret, frame = cap.read()
H, W, _ = frame.shape
#out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
#model.to('cuda')
threshold = 0.5

while ret:
    frame = cv2.resize(frame, (600,600))
    results = model.predict(frame)[0]
    

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #out.write(frame)
    ret, frame = cap.read()

cap.release()
#out.release()
cv2.destroyAllWindows()
