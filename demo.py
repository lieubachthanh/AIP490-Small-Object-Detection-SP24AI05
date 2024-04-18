# from io import StringIO
import argparse
import glob
import os
import pandas as pd
import streamlit as st
import yaml
import torch
from pathlib import Path
from PIL import Image
# import cv2
# import shutil
# import time
# import sys

import detect 
import val

def edit_yaml_file(new_dir):
    """
    Edits the 'data/test.yaml' file with the new directory paths.

    Args:
        new_dir (str): The new directory path.
    """
    # Load the YAML file
    with open('data/test.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Replace the paths
    data['test'] = f'{new_dir}/images'
    data['val'] = f'{new_dir}/images'
    data['train'] = f'{new_dir}/images'

    # Write the changes back to the file
    with open('data/test1img.yaml', 'w') as file:
        yaml.dump(data, file)


def get_subdirs(directory='.'):
    """
    Returns a list of all sub-directories in the specified directory.

    Args:
        directory (str, optional): The directory to search. Defaults to the current directory.

    Returns:
        list: A list of sub-directories.
    """
    result = []
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            result.append(path)
    return result

def get_detection_folder():
    """
    Returns the path of the latest folder in the 'runs/detect' directory.

    Returns:
        str: The path of the latest 'runs/detect' folder.
    """
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def main():
    # path = "runs\detect"
    # if os.path.exists(path):
    #     shutil.rmtree(path) 

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
    opt.line_thickness = 1
    opt.hide_conf = True

    val_parser = argparse.ArgumentParser()
    val_parser.add_argument("--data", type=str, default="data/test1img.yaml", help="dataset.yaml path")
    val_parser.add_argument("--weights", nargs="+", type=str, default='weights/yolov5s-visdrone.pt', help="model path(s)")
    val_parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    val_parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    val_parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    val_parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    val_parser.add_argument("--task", default="test", help="train, val, test, speed or study")
    val_parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    val_parser.add_argument("--soft", action="store_true", default=None, help="use Soft-NMS")
    val_parser.add_argument("--save-hybrid", action="store_true", default=False, help="save label+prediction hybrid results to *.txt")
    val_opt = val_parser.parse_args()
    

    st.sidebar.title("Settings")
    
    source = ("image", "video")
    source_index = st.sidebar.selectbox("input", range(len(source)), format_func=lambda x: source[x])
    img_file = None
    vid_file = None

    if source_index == 0:
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
        if data_src == 'Sample data':
            img_path = glob.glob('data/images/sample_images/*')
            img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
            img_file = img_path[img_slider - 1]
            is_valid = True
        else:
            uploaded_file = st.sidebar.file_uploader(
                "upload image", type=['png', 'jpeg', 'jpg'])
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text='Uploading...'):
                    st.image(uploaded_file, caption="Selected Image")
                    img_name = str(uploaded_file.name)
                    img_name = img_name[:-4]
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'data/images/{uploaded_file.name}')
                    opt.source = f'data/images/{uploaded_file.name}'
                    edit_yaml_file(img_name)
            else:
                is_valid = False
    else:
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
        if data_src == 'Sample data':
            video_file = open('data/videos/result/demo2.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            is_valid = True
        else:
            uploaded_file = st.sidebar.file_uploader("upload video", type=['mp4'])
            if uploaded_file is not None:
                with st.spinner(text='Uploading...'):
                    with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    vid_file = f'data/videos/{uploaded_file.name}'
                    opt.source = vid_file
                is_valid = True
            else:
                is_valid = False

    model_name_option = st.sidebar.selectbox("model", ("our YOLO", "yolov5s",  "orther"))
    
    model_weights = {
    "our YOLO": "weights/OursYOLO.pt",
    "yolov5s": "weights/yolov5s-visdrone.pt",
    }
    
    if model_name_option in model_weights:
        opt.weights = model_weights[model_name_option]
        val_opt.weights = model_weights[model_name_option]
    else:
        uploaded_model = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if uploaded_model is not None:
            is_valid = True
            with st.spinner(text="Uploading..."):
                model_path = os.path.join("weights", uploaded_model.name)
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
            opt.weights = model_path
            val_opt.weights = model_path
        else:
            is_valid = False

    # confidence = st.sidebar.slider('Confidence', min_value=0.001, max_value=1.0, value=.35)    
    # opt.conf_thres = confidence
    # val_opt.conf_thres = confidence
    opt.conf_thres = 0.001
    val_opt.conf_thres = 0.001
    
    # iou = st.sidebar.slider('Iou', min_value=0.1, max_value=1.0, value=.45)
    # opt.iou_thres = iou
    # val_opt.iou_thres = iou
    opt.iou_thres = 0.5
    val_opt.iou_thres = 0.5

    if torch.cuda.is_available():
        dev = st.sidebar.text_input('DEVICE','cpu')
        opt.device = dev
        val_opt.device = dev
    else:
        opt.device = 'cpu'

    # if st.sidebar.checkbox("Soft-NMS"):
    opt.soft = True
    val_opt.soft = True

    if is_valid:
        print('valid')
        print(opt)
        # if st.sidebar.button('detect'):
        if source_index == 0:
            with st.spinner(text='Processing...'):
                if img_file:
                    st.image(img_file, caption="Selected Image")
                    opt.source = str(img_file)
                    detect.main(opt)
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}')/ img), caption="Model prediction")
                else:
                    detect.main(opt)
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                    try:    
                        val.main(val_opt)
                        df = pd.read_csv("data/result.csv")
                        st.write(df)
                    except:
                        pass
        else:
            if st.button('detect') and vid_file:
                with st.spinner(text='Processing...'):
                    detect.main(opt)
                    with st.spinner(text='Preparing Video'):
                        for vid in os.listdir(get_detection_folder()):
                            video = open(str(Path(f'{get_detection_folder()}') / vid), 'rb')
                            vid_bytes = video.read()
                            st.video(vid_bytes)

            
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass