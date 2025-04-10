import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the trained YOLO model
model = YOLO("path to your best.pt") # Replace with your model path

# Function to process image and return results
def process_image(image, iou_threshold=0.6, conf_threshold=0.5):
    # Convert PIL image to OpenCV format and store original dimensions
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    original_height, original_width = image_cv.shape[:2]
    
    # Resize to 640x640
    image_resized = cv2.resize(image_cv, (640, 640), interpolation=cv2.INTER_AREA)
    
    # Perform prediction
    results = model.predict(
        source=image_resized,
        save=False,
        imgsz=640,
        iou=iou_threshold,
        conf=conf_threshold
    )
    
    # Initialize detection data
    detection_data = []
    
    # Process results and store detections in 640x640 scale
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detection_data.append({
                "Class": class_name,
                "Confidence": confidence,
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })
    
    # Resize image back to original dimensions
    image_output = cv2.resize(image_resized, (original_width, original_height), interpolation=cv2.INTER_AREA)
    
    # Adjust bounding boxes to original scale
    width_scale = original_width / 640
    height_scale = original_height / 640
    
    # Draw adjusted bounding boxes
    for detection in detection_data:
        # Scale coordinates
        x1 = int(detection["X1"] * width_scale)
        y1 = int(detection["Y1"] * height_scale)
        x2 = int(detection["X2"] * width_scale)
        y2 = int(detection["Y2"] * height_scale)
        
        # Update detection data with adjusted coordinates
        detection["X1"] = x1
        detection["Y1"] = y1
        detection["X2"] = x2
        detection["Y2"] = y2
        
        # Draw bounding box and label
        color = (255, 0, 0)
        cv2.rectangle(image_output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['Class']} {detection['Confidence']:.2f}"
        cv2.putText(image_output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to RGB for display
    image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)
    return image_output, detection_data

# Streamlit app
def main():
    st.title("Tractor Detection")
    
    # Sidebar for parameters
    st.sidebar.header("Inference Parameters")
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.6, 0.1)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    # File uploader
    uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
    
    if uploaded_files:
        st.header("Inference Results")
        
        # Process each uploaded image
        all_detections = []
        for uploaded_file in uploaded_files:
            try:
                # Read image bytes and open with PIL
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Process image
                output_image, detections = process_image(image, iou_threshold, conf_threshold)
                
                # Display image with detections
                st.subheader(f"Processed Image: {uploaded_file.name}")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(output_image, caption="Image with Bounding Boxes", use_column_width=True)
                
                with col2:
                    st.write("**Detections:**")
                    if detections:
                        for det in detections:
                            st.write(f"- Class: {det['Class']}, Confidence: {det['Confidence']:.2f}")
                    else:
                        st.write("- No detections found")
                
                # Add detections to overall list
                for det in detections:
                    det["Image"] = uploaded_file.name
                all_detections.extend(detections)
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()