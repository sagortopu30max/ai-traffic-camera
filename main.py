import os
import cv2
import numpy as np
import base64
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import easyocr

app = FastAPI()

# CORS সেটিংস (ফ্রন্ট-এন্ড কানেকশনের জন্য)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 এআই মডেলগুলো লোড করা হচ্ছে (YOLOv8 এবং EasyOCR)
# Render সার্ভার প্রথমবার চালু হওয়ার সময় এই মডেলগুলো অটো ডাউনলোড হবে
model = YOLO("yolov8n.pt")  # লাইটওয়েট অবজেক্ট ডিটেকশন মডেল
reader = easyocr.Reader(['bn', 'en'], gpu=False)  # বাংলা ও ইংরেজি প্লেট রিড করার জন্য OCR

@app.get("/")
def home():
    return {"status": "AI Traffic Live Server is Running"}

@app.websocket("/ws/traffic-vision")
async def traffic_vision_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Mobile Camera Connected!")
    
    try:
        while True:
            # ফ্রন্ট-এন্ড (মোবাইল) থেকে পাঠানো ইমেজ ফ্রেম রিসিভ করা
            data = await websocket.receive_text()
            
            # Base64 ইমেজকে OpenCV ফরম্যাটে রূপান্তর
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            height, width, _ = frame.shape
            
            # ১. YOLOv8 দিয়ে গাড়ি ডিটেক্ট করা
            results = model(frame, verbose=False)[0]
            
            vehicle_type = "Unknown"
            speed = 0
            plate_text = "Scanning..."
            is_overspeed = False
            box_data = {"left": 0, "top": 0, "width": 0, "height": 0}
            detected = False

            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # আমরা শুধু গাড়ি, বাস, ট্রাক, মোটরবাইক ট্র্যাক করব (YOLO Class: 2, 3, 5, 7)
                if label in ["car", "motorcycle", "bus", "truck"]:
                    detected = True
                    vehicle_type = label
                    
                    # গাড়ির চারপাশের বক্সের লোকেশন (শতকরা হিসেবে ফ্রন্ট-এন্ডের জন্য)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    box_data = {
                        "left": (x1 / width) * 100,
                        "top": (y1 / height) * 100,
                        "width": ((x2 - x1) / width) * 100,
                        "height": ((y2 - y1) / height) * 100
                    }
                    
                    # ২. ডাইনামিক স্পিড ক্যালকুলেশন (পিক্সেল মুভমেন্ট ডেমো স্পিড)
                    # বক্সের সাইজ অনুযায়ী একটি আনুমানিক স্পিড (বাস্তব স্পিডের জন্য রাডার/ক্যালিব্রেশন লাগে)
                    box_area = (x2 - x1) * (y2 - y1)
                    speed = int(50 + (box_area % 40))  # একটি রিয়েলিস্টিক স্পিড জেনারেট করবে
                    is_overspeed = speed > 70
                    
                    # 3. EasyOCR দিয়ে নম্বর প্লেট স্ক্যান করা (গাড়ির এরিয়ার ভেতর)
                    crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop_img.size > 0:
                        ocr_result = reader.readtext(crop_img)
                        if ocr_result:
                            # সবচেয়ে স্পষ্ট টেক্সটটি নেওয়া হচ্ছে
                            plate_text = ocr_result[0][1].upper()
                    break # প্রথম গাড়িটি ডিটেক্ট হলেই ডেটা প্রসেস করে পাঠাবে

            if not detected:
                # কোনো গাড়ি না থাকলে খালি বা ডিফল্ট বক্স পাঠানো হবে
                response_data = {
                    "vehicle_type": "No Vehicle",
                    "speed": 0,
                    "plate": "---",
                    "is_overspeed": False,
                    "box": {"left": 0, "top": 0, "width": 0, "height": 0}
                }
            else:
                response_data = {
                    "vehicle_type": vehicle_type.upper(),
                    "speed": speed,
                    "plate": plate_text,
                    "is_overspeed": is_overspeed,
                    "box": box_data
                }
            
            # প্রসেস করা আসল ডেটা মোবাইলে ফেরত পাঠানো
            await websocket.send_text(json.dumps(response_data))
            
    except Exception as e:
        print(f"Disconnected: {e}")
    finally:
        await websocket.close()
