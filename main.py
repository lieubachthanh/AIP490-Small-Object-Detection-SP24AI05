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
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
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
    "yolov5-cus1": "weights/DSDyolov5s.pt",
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

    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.35)    
    opt.conf_thres = confidence
    
    iou = st.sidebar.slider('Iou', min_value=0.1, max_value=1.0, value=.45)
    opt.iou_thres = iou

    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        opt.device = device_option
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    if st.sidebar.checkbox("Soft-NMS"):
        pass
        # sigma = st.sidebar.slider('Sigma', min_value=0.1, max_value=1.0, value=.5)
        # opt.soft = sigma

    if is_valid:
        print('valid')
        print(opt)
        if st.sidebar.button('detect'):
            detect.main(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass