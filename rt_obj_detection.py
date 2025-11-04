#import cv2
#import torch
#stream = cv2.VideoCapture(0)
#
#model = torch.hub.load('ultralytics/yolov5',
#                'yolov5s',
#                pretrained=True)
#
#def score_frame(frame, model):
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    model.to(device)
#    frame = [torch.tensor(frame)]
#    results = self.model(frame)
#    labels = results.xyxyn[0][:, -1].numpy()
#    cord = results.xyxyn[0][:, :-1].numpy()
#    return labels, cord
#
#def plot_boxes(self, results, frame):
#    labels, cord = results
#    n = len(labels)
#    x_shape, y_shape = frame.shape[1], frame.shape[0]
#    for i in range(n):
#        row = cord[i]
#        # If score is less than 0.2 we avoid making a prediction.
#        if row[4] < 0.2: 
#            continue
#        x1 = int(row[0]*x_shape)
#        y1 = int(row[1]*y_shape)
#        x2 = int(row[2]*x_shape)
#        y2 = int(row[3]*y_shape)
#        bgr = (0, 255, 0) # color of the box
#        classes = self.model.names # Get the name of label index
#        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
#        cv2.rectangle(frame, \
#                      (x1, y1), (x2, y2), \
#                       bgr, 2) #Plot the boxes
#        cv2.putText(frame,\
#                    classes[labels[i]], \
#                    (x1, y1), \
#                    label_font, 0.9, bgr, 2) #Put a label over box.
#        return frame
#
#def __call__(self):
#    player = self.get_video_stream() #Get your video stream.
#    assert player.isOpened() # Make sure that their is a stream. 
#    #Below code creates a new video writer object to write our
#    #output stream.
#    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
#    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    four_cc = cv2.VideoWriter_fourcc(*"MJPG") #Using MJPEG codex
#    out = cv2.VideoWriter(out_file, four_cc, 20, \
#                          (x_shape, y_shape)) 
#    ret, frame = player.read() # Read the first frame.
#    while rect: # Run until stream is out of frames
#        start_time = time() # We would like to measure the FPS.
#        results = self.score_frame(frame) # Score the Frame
#        frame = self.plot_boxes(results, frame) # Plot the boxes.
#        end_time = time()
#        fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
#        print(f"Frames Per Second : {fps}")
#        out.write(frame) # Write the frame onto the output.
#        ret, frame = player.read() # Read next frame.

# ...existing code...
import time
import cv2
import numpy as np
import torch

def load_model(device=None, model_name="yolov5s", conf_threshold=0.2):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    model.to(device)
    model.conf = conf_threshold  # confidence threshold for detection
    return model, device

def score_frame(model, frame):
    # model expects BGR numpy image; it will handle preprocessing internally
    results = model(frame)  # returns a Results object
    # results.xyxy[0]: (N,6) -> x1,y1,x2,y2,confidence,class
    detections = results.xyxy[0].cpu().numpy()
    return detections  # Nx6 array

def plot_boxes(detections, frame, names):
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.2:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[int(cls)]}: {conf:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main(source=0, out_file="output.avi"):
    model, device = load_model()
    names = model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: cannot open video source", source)
        return

    x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()
            detections = score_frame(model, frame)
            frame = plot_boxes(detections, frame, names)
            end_time = time.time()
            fps = 1 / max((end_time - start_time), 1e-4)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            out.write(frame)
            cv2.imshow("YOLOv5 Live", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# ...existing code...