from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI()


model = YOLO("yolov8m.pt") 

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    
    video_path = Path(f"./temp_video/{file.filename}")
    video_path.parent.mkdir(parents=True, exist_ok=True) 
    with open(video_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(str(video_path))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video_path = Path(f"./output_videos/processed_{file.filename}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)  
    out = cv2.VideoWriter(str(output_video_path), fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(3)), int(cap.get(4))))  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
       
        results = model(frame) 

        for result in results:  
            for detection in result.boxes.data.numpy():  
                x1, y1, x2, y2, conf, cls = detection
                label = f"{model.names[int(cls)]}: {conf:.2f}" 

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()  
    cv2.destroyAllWindows()
    
    return HTMLResponse(content=f"""
    <html>
        <body>
            <h2>Video Processed</h2>
            <video width="640" height="480" controls>
                <source src="/output_videos/{output_video_path.name}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </body>
    </html>
    """)

@app.get("/output_videos/{video_name}", response_class=FileResponse)
async def get_video(video_name: str):
    video_path = Path(f"./output_videos/{video_name}")
    return FileResponse(video_path)  
