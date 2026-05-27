import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import random

app = FastAPI()

# CORS সেটিংস (যাতে যেকোনো ডোমেন থেকে ওয়েবসাইটটি কাজ করে)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "AI Traffic Server is Running"}

@app.websocket("/ws/traffic-vision")
async def traffic_vision_websocket(websocket: WebSocket):
    await websocket.accept()
    print("ক্যামেরা কানেক্টেড হয়েছে!")
    
    # ডেমো ডেটার জন্য কিছু গাড়ি ও লাইসেন্স প্লেট
    car_types = ["Sedan", "SUV", "Microbus", "Truck", "Bike"]
    plates = ["DHAKA METRO-GA-11-2222", "CHATTRO-METRO-HA-45-6789", "DHAKA METRO-KA-99-1234"]
    
    try:
        while True:
            # ফ্রন্ট-এন্ড থেকে ভিডিও ফ্রেমের ডেটা রিসিভ করা
            data = await websocket.receive_text()
            
            # সার্ভার সাইড স্পিড ও প্লেট ডিটেকশন লজিক (সিমুলেশন)
            # বাস্তব প্রজেক্টে এখানে OpenCV ও YOLOv8 ফ্রেমটি প্রসেস করবে
            speed = random.randint(40, 95)
            vehicle = random.choice(car_types)
            plate = random.choice(plates)
            
            # এআই ডিটেকশন ফলাফল তৈরি
            response_data = {
                "vehicle_type": vehicle,
                "speed": speed,
                "plate": plate,
                "is_overspeed": speed > 70,
                # স্ক্রিনে বক্স বসানোর জন্য ডাইনামিক পজিশন
                "box": {
                    "left": random.randint(10, 50),
                    "top": random.randint(30, 50),
                    "width": random.randint(25, 35),
                    "height": random.randint(35, 45)
                }
            }
            
            # ফলাফল আবার ফ্রন্ট-এন্ডে ফেরত পাঠানো
            await websocket.send_text(json.dumps(response_data))
            
    except Exception as e:
        print(f"কানেকশন বিচ্ছিন্ন হয়েছে: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)