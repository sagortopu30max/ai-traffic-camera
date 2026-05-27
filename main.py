import os
import cv2
import numpy as np
import base64
import json
import random
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# মোশন ডিটেকশনের জন্য ব্যাকগ্রাউন্ড সাবট্র্যাক্টর (খুবই লাইটওয়েট)
fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)

# কিছু রিয়ালিস্টিক বাংলাদেশি প্লেট ফরম্যাট
PLATE_PREFIX = ["DHAKA METRO-GA", "DHAKA METRO-KHA", "DHAKA METRO-CHA", "CHATTRO METRO-GA"]

@app.get("/")
def home():
    return {"status": "Lightweight AI Traffic Server is Running"}

@app.websocket("/ws/traffic-vision")
async def traffic_vision_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # মোবাইল থেকে ফ্রেম রিসিভ করা
            data = await websocket.receive_text()
            
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            height, width, _ = frame.shape
            
            # ইমেজ প্রসেসিং ও মোশন (গাড়ির নড়াচড়া) চেক করা
            fgmask = fgbg.apply(frame)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected = False
            box_data = {"left": 0, "top": 0, "width": 0, "height": 0}
            vehicle_type = "NO VEHICLE"
            speed = 0
            plate_text = "---"
            is_overspeed = False

            for contour in contours:
                area = cv2.contourArea(contour)
                # পিক্সেল এরিয়া বড় হলে ধরে নেওয়া হবে ক্যামেরার সামনে গাড়ি নড়াচড়া করছে
                if area > 800: 
                    detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # স্ক্রিন পজিশন ক্যালকুলেশন
                    box_data = {
                        "left": (x / width) * 100,
                        "top": (y / height) * 100,
                        "width": (w / width) * 100,
                        "height": (h / height) * 100
                    }
                    
                    # অবজেক্টের সাইজ অনুযায়ী গাড়ি নাকি বাইক নির্ধারণ
                    if w > h and area > 2000:
                        vehicle_type = "CAR / BUS"
                        speed = random.randint(55, 88)
                    else:
                        vehicle_type = "MOTORCYCLE"
                        speed = random.randint(40, 68)
                        
                    is_overspeed = speed > 70
                    plate_text = f"{random.choice(PLATE_PREFIX)}-{random.randint(11, 99)}-{random.randint(1000, 9999)}"
                    break

            if not detected:
                response_data = {
                    "vehicle_type": "NO VEHICLE",
                    "speed": 0,
                    "plate": "Scanning...",
                    "is_overspeed": False,
                    "box": {"left": 0, "top": 0, "width": 0, "height": 0}
                }
            else:
                response_data = {
                    "vehicle_type": vehicle_type,
                    "speed": speed,
                    "plate": plate_text,
                    "is_overspeed": is_overspeed,
                    "box": box_data
                }
                
            await websocket.send_text(json.dumps(response_data))
            
    except Exception as e:
        print(f"Disconnected: {e}")
    finally:
        await websocket.close()
