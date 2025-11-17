import cv2
from PIL import Image
from transformers import pipeline
import torch
import time
 
print("\n" + "="*60)
print("JETSON WEBCAM IMAGE CAPTIONING")
print("="*60)
 
print("\nLoading image captioning model...")
print("(First run downloads ~500MB, please wait...)")
 
# Setup device
device = 0 if torch.cuda.is_available() else -1
print(f"Using: {'GPU (CUDA)' if device == 0 else 'CPU'}")
 
# Load model using pipeline (most reliable method)
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device
)
 
print("✓ Model loaded successfully!\n")
 
# Open webcam
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
print("="*60)
print("CONTROLS:")
print("  Press 'c' - Capture and describe current frame")
print("  Press 'a' - Auto mode (describe every 3 seconds)")
print("  Press 's' - Stop auto mode")
print("  Press 'q' - Quit")
print("="*60)
print("\nWebcam active. Waiting for commands...\n")
 
auto_mode = False
last_auto_time = 0
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame")
        break
    
    # Show live feed
    display_frame = frame.copy()
    if auto_mode:
        cv2.putText(display_frame, "AUTO MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Jetson Webcam - Press C/A/S/Q', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Manual capture
    if key == ord('c'):
        print("\n🔍 Analyzing image...")
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        result = captioner(pil_image, max_new_tokens=50)
        description = result[0]['generated_text']
        
        elapsed = time.time() - start_time
        print(f"📝 {description}")
        print(f"⏱️  ({elapsed:.2f}s)\n")
    
    # Toggle auto mode
    elif key == ord('a'):
        auto_mode = True
        print("▶️  AUTO MODE ENABLED (capturing every 3 seconds)\n")
    
    elif key == ord('s'):
        auto_mode = False
        print("⏸️  AUTO MODE STOPPED\n")
    
    # Auto capture every 3 seconds
    elif auto_mode and (time.time() - last_auto_time) > 3:
        print("🔍 Auto-analyzing...")
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        result = captioner(pil_image, max_new_tokens=50)
        description = result[0]['generated_text']
        
        elapsed = time.time() - start_time
        print(f"📝 {description} ({elapsed:.2f}s)\n")
        
        last_auto_time = time.time()
    
    # Quit
    elif key == ord('q'):
        break
 
cap.release()
 