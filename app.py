from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import shutil
import os
import cv2
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static directory for result images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your path if needed

# Class names
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Ensure static folder exists
os.makedirs("static", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Face Mask Detection</title>
    </head>
    <body>
        <h2>Upload an image for detection:</h2>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Detect">
        </form>
    </body>
    </html>
    """


@app.post("/upload/")
async def upload(file: UploadFile):
    # Save uploaded file to disk
    image_path = f"static/{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO detection
    results = model(image_path)[0]

    # Load image for drawing
    img = cv2.imread(image_path)

    # Draw boxes if detections exist
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    else:
        # Write "No detections" on image
        cv2.putText(img, "No detections", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Save annotated result
    result_path = f"static/result_{file.filename}"
    cv2.imwrite(result_path, img)

    # Return result in HTML
    return HTMLResponse(content=f"""
    <html>
        <body>
            <h3>Detection Result:</h3>
            <img src="/{result_path}" width="640">
            <br><br>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
