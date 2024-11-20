from flask import Flask, render_template, request, redirect,  flash, abort, url_for, session
# from flask_session import Session
# from vadhyakalakshethra import app,db,bcrypt,mail
from pothole import app,mail

from flask_mail import Message
from pothole import app
from pothole.models import *
# from bewell.forms import *
from flask_login import login_user, current_user, logout_user, login_required
from random import randint
import os
from PIL import Image
# from flask_mail import Message


import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from datetime import datetime



from tensorflow import keras

import numpy as np
import cv2
import pygame
# Initialize Pygame
pygame.init()

# Initialize Pygame's mixer
pygame.mixer.init()



def preprocess_img(imgBGR, erode_dilate=True):
    if imgBGR is None:
        return None
    
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)  
    Bmin = np.array([100, 43,46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)
    
    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)
    
    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)
    
    if erode_dilate:
        kernelErosion = np.ones((3,3), np.uint8)
        kernelDilation = np.ones((3,3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)
    
    return img_bin

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects
    if max_area < 0:
        max_area = img_bin.shape[0] * img_bin.shape[1] 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def get_class(classNo):
    # Define your class labels here
    class_labels = ["label1", "label2", "label3", ...]  # Define your class labels here
    if 0 <= classNo < len(class_labels):
        return class_labels[classNo]
    else:
        return "Unknown"



def get_classe(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    # elif classNo == 13:
    #     return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'



@app.route('/sign_live',methods=['GET', 'POST'])
def sign_live():
    threshold = 0.95
    font = cv2.FONT_HERSHEY_SIMPLEX
    model_sign_video = keras.models.load_model('pothole/traffif_sign_model.h5')
   
    cap = cv2.VideoCapture(0)
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while(1):
        ret, img = cap.read()
        img_bin = preprocess_img(img, erode_dilate=False)
        cv2.imshow("Bin_image",img_bin)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area = min_area) #get x,y,w,h
        img_bbx = img.copy()
        for rect in rects:
            # rect[2] is width and rect[3] for height
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)
            
            size = max(rect[2], rect[3])
            x1 = int(max(0, (xc - size / 2)))
            y1 = int(max(0, (yc - size / 2)))
            x2 = int(min(cols, int(xc + size / 2)))
            y2 = int(min(rows, int(yc + size / 2)))
            
            if rect[2] > 100 and rect[3] > 100:
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            
            crop_img = np.asarray(img[y1:y2, x1:x2])
            crop_img = cv2.resize(crop_img, (32,32))
            crop_img = preprocessing(crop_img)
            cv2.imshow("afterprocessing", crop_img)
            crop_img = crop_img.reshape(1, 32, 32, 1) #(1,32,32) after reshape it become (1,32,32,1)
            
            #make prediction
            predictions = model_sign_video.predict(crop_img)
            classIndex = np.argmax(predictions, axis = 1)
            probabilityValue = np.amax(predictions)
            print("probabilityValue",probabilityValue)
            if probabilityValue > threshold:
                if classIndex <= 42 and classIndex != 13:
                    #write class name on the output screen
                    cv2.putText(img_bbx, str(classIndex) + " " + str(get_classe(classIndex)), (rect[0], rect[1] - 10), 
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    # write probability value on the output screen
                    cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                                (0, 0, 255), 2, cv2.LINE_AA)
                
        cv2.imshow("detect result", img_bbx)
        if cv2.waitKey(1) & 0xFF == ord('q'):           # q for quit 
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template("user_index.html")








@app.route('/sign_video',methods=['GET', 'POST'])
def sign_video():
    threshold = 0.75
    font = cv2.FONT_HERSHEY_SIMPLEX
    model_sign_video = keras.models.load_model('pothole/traffif_sign_model.h5')
    
    if request.method == 'POST':

        file= request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        video_path = "pothole/static/uploads/"+file.filename  # Update with your video file path
        print(video_path)
        
        cap = cv2.VideoCapture(video_path)
        cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            
            ret, img = cap.read()
            if not ret:
                
                break
            
            img_bin = preprocess_img(img, erode_dilate=False)
            min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
            rects = contour_detect(img_bin, min_area=min_area)  # get x,y,w,h
            img_bbx = img.copy()
            
            for rect in rects:
                xc = int(rect[0] + rect[2] / 2)
                yc = int(rect[1] + rect[3] / 2)
                
                size = max(rect[2], rect[3])
                x1 = int(max(0, (xc - size / 2)))
                y1 = int(max(0, (yc - size / 2)))
                x2 = int(min(cols, int(xc + size / 2)))
                y2 = int(min(rows, int(yc + size / 2)))
                
                if rect[2] > 100 and rect[3] > 100:
                    cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                
                crop_img = np.asarray(img[y1:y2, x1:x2])
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = preprocessing(crop_img)
                crop_img = crop_img.reshape(1, 32, 32, 1)  # (1,32,32) after reshape it becomes (1,32,32,1)
                
                # Make prediction
                predictions = model_sign_video.predict(crop_img)
                classIndex = np.argmax(predictions, axis=1)[0]
                probabilityValue = np.amax(predictions)
                if probabilityValue > threshold:
                        #write class name on the output screen
                    cv2.putText(img_bbx, str(classIndex) + " " + str(get_classe(classIndex)), (rect[0], rect[1] - 10), 
                                font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    # write probability value on the output screen
                    cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                                (0, 0, 255), 2, cv2.LINE_AA)
                    
            cv2.imshow("detect result", img_bbx)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q for quit
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template("sign_video.html")










@app.route('/livedetect',methods=['GET', 'POST'])
def livedetect(save_img=False):
    new=0
    temp = 0
    average = 0
        



    # source='video.mp4'
    source = 0
    print(source)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    source, weights, view_img, save_txt, imgsz, trace = source, 'trained_weights/yolov7_pothole_multi_res/weights/best.pt', True, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave# and not source.endswith('.txt')  # save inference images
    webcam = True
    # source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(str(source), img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            pothole=s
            print("s=",s)
            try:
                s = s[s.index(" ")+1:]
                print("sssss", s)
                y = int(s[0:s.index(" ")])
                print("try-y",y)
            except:
                y = 0
                print("except-y",y)
            print("y",y)

            if pothole:

                new=new+y

                if(temp == 0):
                    if y > 0 :
                        pygame.mixer.Sound(os.path.join(app.config['UPLOAD_FOLDER'], 'pothole_detected.wav')).play()
                temp += 1
                if(temp == 10):
                    temp = 0
                    
            print("new",new)

            average=new/20
            a=int(average)
            print(int(average))
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")



        print(f'Done. ({time.time() - t0:.3f}s)')
        if time.time() - t0 > 40:  # Close the window after 40 seconds
            cv2.destroyAllWindows()
            break

        # return render_template("/detection.html",output=average) 
        
    return render_template("/detection.html",output=a)  


@app.route('/detect',methods=['GET', 'POST'])
def detect(save_img=False):
    if request.method == 'POST':
        new=0
        temp = 0
        file= request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        # print(file.filename)
        # obj =Detection(data=file.filename)
        # db.session.add(obj)
        # db.session.commit()


        # source='video.mp4'
        source = file.filename
        print(source)
        parser = argparse.ArgumentParser()
        # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        print(opt)
        # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        source, weights, view_img, save_txt, imgsz, trace = source, 'trained_weights/yolov7_pothole_multi_res/weights/best.pt', True, opt.save_txt, opt.img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                pothole=s
                y = int(s[0:s.index(" ")])
                print(y)

                if pothole:
                    new=new+y

                    # if(temp == 0):
                    #     pygame.mixer.Sound(os.path.join(app.config['UPLOAD_FOLDER'], 'pothole_detected.wav')).play()
                    # temp += 1
                    # if(temp == 10):
                    #     temp = 0
                     
                    print(new)

                    # pygame.mixer.Sound(os.path.join(app.config['UPLOAD_FOLDER'], 'pothole_detected.wav')).play()

                average=new/20
                a=int(average)
                print(int(average))
                
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")



        print(f'Done. ({time.time() - t0:.3f}s)')

        return render_template("/detection.html",output=a) 
        
    return render_template("/detection.html")   


# if __name__ == '__main__':
    
#     check_requirements(exclude=('pycocotools', 'thop'))

#     with torch.no_grad():
#         detect(source='video.mp4')
        
        
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        #     detect()

   











@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/user_index',methods=['GET', 'POST'])
def user_index():
    return render_template("user_index.html")


@app.route('/admin_index',methods=['GET', 'POST'])
def admin_index():
    return render_template("admin_index.html")


@app.route('/officer_index',methods=['GET', 'POST'])
def officer_index():
    return render_template("officer_index.html")




@app.route('/layout')
def layout():
    return render_template("layout.html")

@app.route('/admin_layout')
def admin_layout():
    return render_template("admin_layout.html")


@app.route('/user_layout')
def user_layout():
    return render_template("user_layout.html")


@app.route('/officer_layout')
def officer_layout():
    return render_template("officer_layout.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")





@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        address= request.form['address']
        contact= request.form['contact'] 
        my_data = Register(name=name,email=email,password=password,contact=contact,address=address,usertype="user")
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/login')
    return render_template("register.html")


@app.route('/login',methods=['GET','POST'])
def login():
    
    if request.method=="POST":
        

        email=request.form['email']
        password=request.form['password']
        admin =Register.query.filter_by(email=email, password=password,usertype= 'admin').first()
        user =Register.query.filter_by(email=email, password=password,usertype= 'user').first()
        officer =Register.query.filter_by(email=email, password=password,usertype= 'officer').first()  
               
        if admin:
            login_user(admin)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect('/admin_index') 
        
        elif user:

            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect('/user_index') 

        elif officer:
            login_user(officer)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect('/officer_index')
        
        else:
            d="Invalid Username or Password!"
            return render_template("login.html",d=d)
    return render_template("login.html")


@app.route('/add_officer',methods=['GET','POST'])
def add_officer():
    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        address= request.form['address']
        contact= request.form['contact']  
        my_data = Register(name=name,email=email,password=password,contact=contact,address=address,usertype="officer")
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/admin_index')
    return render_template("add_officer.html")



@app.route('/add_complaints',methods=['GET','POST'])
def add_complaints():
    a=Register.query.filter_by(id=current_user.id).first()
    email = a.email
    if request.method == 'POST':
        

        subject= request.form['subject']
        message= request.form['message']  
        my_data = Complaints(subject=subject,message=message,uid=current_user.id,email=email)
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/user_index')
    return render_template("add_complaints.html",a=a)


@app.route('/add_speedlimit',methods=['GET','POST'])
def add_speedlimit():
    if request.method == 'POST':
        district = request.form['district']
        area = request.form['area']
        limit= request.form['limit']  
        my_data = Speedlimit(district=district,area=area,limit=limit)
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/admin_index')
    return render_template("add_speedlimit.html")


@app.route('/add_rules',methods=['GET','POST'])
def add_rules():
    if request.method == 'POST':
        rules = request.form['rules']
        image = request.files['image'] 
        pic_file = save_picture(image)
        view = pic_file
        print(view) 
        my_data = Rules(rules=rules,image=view)
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/admin_index')
    return render_template("add_rules.html")


@app.route('/add_news',methods=['GET','POST'])
def add_news():
    if request.method == 'POST':
        news = request.form['news']
        image = request.files['image'] 
        pic_file = save_picture(image)
        view = pic_file
        print(view) 
        my_data = News(news=news,image=view)
        db.session.add(my_data) 
        db.session.commit()
        return redirect('/admin_index')
    return render_template("add_news.html")


@app.route('/add_projects',methods=['GET','POST'])
def add_projects():
    if request.method == 'POST':
        date = request.form['date']
        image = request.files['image'] 
        details = request.form['details']
        pic_file = save_picture(image)
        view = pic_file
        print(view) 
        my_data = Projects(date=date,image=view,details=details)
        db.session.add(my_data)
        db.session.commit()
        return redirect('/officer_index')
    return render_template("add_projects.html")


@app.route('/add_response/<int:id>',methods=['GET','POST'])
def add_response(id):
    c=Complaints.query.filter_by(id=id).first()
    if request.method == 'POST':
        c.response =  request.form['response']
        c.status="success"
        db.session.commit()
        a_sendmail(c.email,c.response)
        return redirect('/officer_index')
    return render_template("add_response.html",c=c)


def a_sendmail(username,response):
    msg = Message('Response',
                  recipients=[username])
    msg.body = f''' Response for your Complaint is , {response}  '''
    mail.send(msg)

@app.route('/view_users')
def view_users():
    obj = Register.query.filter_by(usertype="user").all()
    return render_template("view_users.html",obj=obj)


@app.route('/view_officer')
def view_officer():
    obj = Register.query.filter_by(usertype="officer").all()
    return render_template("view_officer.html",obj=obj)

@app.route('/view_complaints')
def view_complaints():
    obj = Complaints.query.filter_by(status="NULL").all()
    return render_template("view_complaints.html",obj=obj)


@app.route('/view_speedlimit')
def view_speedlimit():
    obj = Speedlimit.query.all()
    return render_template("view_speedlimit.html",obj=obj)

@app.route('/ad_view_limit')
def ad_view_limit():
    obj = Speedlimit.query.all()
    return render_template("ad_view_limit.html",obj=obj)

@app.route('/view_rules')
def view_rules():
    obj = Rules.query.all()
    return render_template("view_rules.html",obj=obj)

@app.route('/ad_vw_rules')
def ad_vw_rules():
    obj = Rules.query.all()
    return render_template("ad_vw_rules.html",obj=obj)


@app.route('/view_news')
def view_news():
    obj = News.query.all()
    return render_template("view_news.html",obj=obj)

@app.route('/ad_vw_news')
def ad_vw_news():
    obj = News.query.all()
    return render_template("ad_vw_news.html",obj=obj)


@app.route('/view_projects')
def view_projects():
    obj = Projects.query.all()
    return render_template("view_projects.html",obj=obj)


@app.route('/p_view_projects')
def p_view_projects():
    obj = Projects.query.all()
    return render_template("p_view_projects.html",obj=obj)


@app.route('/u_view_complaints')
def u_view_complaints():
    obj = Complaints.query.filter_by(uid=current_user.id).all()
    return render_template("u_view_complaints.html",obj=obj)

@app.route('/view_response/<int:id>')
def view_response(id):
    obj = Complaints.query.filter_by(id=id).all()
    return render_template("view_response.html",obj=obj)

@app.route('/edit_officer/<int:id>',methods=["GET","POST"])
def edit_officer(id):
    c= Register.query.get_or_404(id)
    if request.method == 'POST':
        c.name =  request.form['name']
        c.email =  request.form['email']
        c.contact =  request.form['contact']
        c.password =  request.form['password']

        db.session.commit()
        return redirect('/view_officer')
    else:
        return render_template('edit_officer.html',c=c)   
    
@app.route('/delete_officer/<int:id>', methods = ['GET','POST'])
def delete_officer(id):

    delet = Register.query.get_or_404(id)
    try:
        db.session.delete(delet)
        db.session.commit()
        return redirect('/view_officer')
    except:
        return 'There was a problem deleting that task'

@login_required
@app.route("/logout")
def logout():
    logout_user()
    return redirect('/')



def save_picture(form_picture):
    random_hex = random_with_N_digits(14)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = str(random_hex) + f_ext
    picture_path = os.path.join(app.root_path, 'static/uploads', picture_fn)
    output_size = (500, 500)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

