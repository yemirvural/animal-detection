import os
import time
import logging
import tempfile
import cv2 as cv
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

MODEL_DIR = './runs/detect/train/weights/best.pt'

logging.basicConfig(
    filename="./logs/log.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def main():
    # Load a model
    global model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("**Animal Classes**")

    class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra', "Cheetah", "Fox", "Jaguar", "Tiger", "Lion", "Panda"]

    for animal in class_names:
        st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Real-time Animal Species Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    # Load image or video
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file)
        
        if uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)

def apply_filters(image, canny_thresh1, canny_thresh2, kernel_size):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # Apply Canny Edge Detection
    edges = cv.Canny(blurred, canny_thresh1, canny_thresh2)
    
    # Apply Laplacian Edge Detection
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    return edges, laplacian

def convert_color(image, color_space):
    if color_space == 'RGB':
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        return cv.cvtColor(image, cv.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        return cv.cvtColor(image, cv.COLOR_BGR2LAB)
    elif color_space == 'YUV':
        return cv.cvtColor(image, cv.COLOR_BGR2YUV)
    elif color_space == 'XYZ':
        return cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    else:
        raise ValueError("Invalid color space. Supported color spaces are: RGB, HSV, LAB, YUV, XYZ")

def histogram_equalization(image):
    # Convert to YUV color space
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    # Equalize the histogram of the Y channel
    yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])
    # Convert back to BGR color space
    equalized_image = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
    return equalized_image

def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = convert_color(image_np, 'RGB')

    # User-configurable parameters
    canny_thresh1 = st.sidebar.slider("Canny Edge Detection Threshold 1", 0, 255, 100)
    canny_thresh2 = st.sidebar.slider("Canny Edge Detection Threshold 2", 0, 255, 200)
    kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2)

    # Apply filters
    edges, laplacian = apply_filters(image_cv, canny_thresh1, canny_thresh2, kernel_size)

    # Histogram equalization
    equalized_image = histogram_equalization(image_cv)

    # Predict the image
    predict = model.predict(image_cv)

    # Plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # Display the original image
    st.image(image_cv, caption="Original Image", width=600)
    
    # Display the edge detected image
    st.image(edges, caption="Edge Detected Image", width=600)
    
    # Display the Laplacian edge detected image
    st.image(laplacian, caption="Laplacian Edge Detected Image", width=600)
    
    # Display the histogram equalized image
    st.image(equalized_image, caption="Histogram Equalized Image", width=600)
    
    # Display the detection result
    st.image(plotted, caption="Detected Image", width=600)
    
    logging.info("Detected Image")

def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    if not cap.isOpened():
        st.error("Error opening video file.")
 
    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    canny_thresh1 = st.sidebar.slider("Canny Edge Detection Threshold 1", 0, 255, 100)
    canny_thresh2 = st.sidebar.slider("Canny Edge Detection Threshold 2", 0, 255, 200)
    kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2)

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # Apply filters
            edges, laplacian = apply_filters(frame, canny_thresh1, canny_thresh2, kernel_size)

            # Predict the frame
            predict = model.predict(frame, conf=0.75)
            
            # Plot boxes
            plotted = predict[0].plot()

            # Concatenate the original frame, the edge detected frame, and the Laplacian edge detected frame horizontally
            combined = cv.hconcat([frame, cv.cvtColor(edges, cv.COLOR_GRAY2BGR), cv.cvtColor(laplacian, cv.COLOR_GRAY2BGR), plotted])

            # Display the video
            frame_placeholder.image(combined, channels="BGR", caption="Video Frame")
        
        if stop_placeholder:
            break

    cap.release()
    os.unlink(temp_file.name)

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    st.write(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds ({fps:.2f} FPS)")

if __name__ == '__main__':
    main()
