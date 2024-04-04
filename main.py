# from io import StringIO
from pathlib import Path
import streamlit as st
import time
import detect 
import os
# import sys
import argparse
from PIL import Image
import torch
import cv2

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def main():

    st.title('Object Recognition Dashboard')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-visdrone.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    # parser.add_argument("--data", type=str, default="data/visdrone.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default= "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--soft",default=False, action="store_true", help="use Soft-NMS")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    st.sidebar.title("Settings")

    source = ("image", "video")
    source_index = st.sidebar.selectbox("input", range(
        len(source)), format_func=lambda x: source[x])
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "upload image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Uploading...'):
                # st.sidebar.image(uploaded_file)
                st.image(uploaded_file, caption="Selected Image")
                
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Uploading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    model_name_option = st.sidebar.selectbox("model", ("yolov5s", "yolov5-cus1", "yolov5-cus2", "orther"))
    
    model_weights = {
    "yolov5s": "weights/yolov5s-visdrone.pt",
    "yolov5-cus1": "weights/lam.pt",
    "yolov5-cus2": "weights/DSDyolov5s.pt",
    }
    
    if model_name_option in model_weights:
        opt.weights = model_weights[model_name_option]
    else:
        uploaded_model = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if uploaded_model is not None:
            is_valid = True
            with st.spinner(text="Uploading..."):
                model_path = os.path.join("weights", uploaded_model.name)
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
            opt.weights = model_path
        else:
            is_valid = False

    confidence = st.sidebar.slider('Confidence', min_value=0.001, max_value=1.0, value=.35)    
    opt.conf_thres = confidence
    
    iou = st.sidebar.slider('Iou', min_value=0.1, max_value=1.0, value=.45)
    opt.iou_thres = iou

    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        opt.device = '' if device_option == 'cuda' else 'cpu'
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    if st.sidebar.checkbox("Soft-NMS"):
        opt.soft = True

    if is_valid:
        print('valid')
        print(opt)
        if st.sidebar.button('detect'):
            if source_index == 0:
                detect.main(opt)
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
            else:
                opt.view_img
                detect.main(opt)
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

            # if source_index == 1 and uploaded_file is not None:
            #     cap = cv2.VideoCapture(opt.source)
            #     if not cap.isOpened():
            #         st.error("Error opening video file")
            #     else:
            #         with st.spinner(text='Performing real-time object detection...'):
            #             prev_time = 0
            #             while True:
            #                 ret, frame = cap.read()
            #                 if not ret:
            #                     break
                            
            #                 # Perform object detection on the frame
            #                 # ... (your existing object detection code goes here)

            #                 # Calculate and display FPS
            #                 curr_time = time.time()
            #                 fps = 1 / (curr_time - prev_time)
            #                 prev_time = curr_time
            #                 cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #                 # Display the frame with detected objects and FPS
            #                 st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="BGR")

            # cap.release()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass