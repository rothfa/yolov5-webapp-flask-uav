from flask import Flask, render_template, Response, request, session, redirect
import argparse
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import sys
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import json 
import pandas as pd
from datetime import datetime
import time
from threading import Thread
#import plotly
#import plotly.express as px

global predictions

@torch.no_grad()
def model_load(weights="best.pt",  # model.pt path(s)
               device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    print("Loading Model, this takes few seconds!")
    return model

app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    model = model_load()
    sourse = "rtmp://192.168.178.20:1935/live"
    #sourse = "testimage.jpg"
    device = select_device('0')
    imgsz = [640, 640]  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    view_img = True  # show results
    save_txt = True  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # ugmented inference
    visualize = False  # visualize features
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    source = str(sourse)
    #webcam = False
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    if pt and device.type != "cpu":
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    #generate filename for prediction
    now = datetime.now()
    pred_time = now.strftime("%H%M%S")
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
  
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + (
            #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:
                             #f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}') #{conf:.2f}
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                        #                  BGR=True)
                        #Write results into json-file
                        results_json = [{"classid": cls.item(),"label": label,"Confidence": conf.item(), "recommendation": get_recom(cls)}]
                        #create prediction file
                        pred_path = str(get_date()[0]) + "/" + str(get_date()[1]) + "/" + str(get_date()[2]) + "/"
                        if not os.path.exists("runs/detect/" + pred_path):
                        	os.makedirs("runs/detect/" + pred_path)
                        if os.path.isfile("runs/detect/" + pred_path + str(pred_time) +"_predictons.json") == False:
                        	with open("runs/detect/" + pred_path + str(pred_time) +"_predictons.json", 'w') as f:
                        		json.dump(results_json,f)
                        else: 
                        	update_file(pred_path,pred_time,cls.item(),label,conf.item())
                        	

                        
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        #cv2.imwrite('frame.jpg', im0)
        frame =  cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # String results
        print(s)
        # wait key to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


#get date for generating file name
def get_date():
	now = datetime.now()
	date_time = now.strftime("%m_%d_%Y-%H%M%S")
	month = now.strftime("%m")
	day = now.strftime("%d")
	year = now.strftime("%Y")
	return year,month,day
	

#read recommendations
def read_recommendations():
	rec_data = ""
	with open("runs/recommendations/recom.json", 'r') as rec:
		rec_data = json.load(rec)
	return rec_data

#get recommendation for first prediction
def get_recom(cls):
	recom = ""
	rec_data = read_recommendations()
	for i in range(0,len(rec_data)):
		if (rec_data[i]["class"] == cls):
			recom = rec_data[i]["recommendation"]
		if recom == "": #if there is no recommendation for a prediction
			recom = "Ask Supervisor for Recommendation"
	return recom

#update file after writing
def update_file(pred_path,pred_time,cls,label,conf):
	listobj = []
	recom = ""
	rec_data = read_recommendations()
	for i in range(0,len(rec_data)):
		if (rec_data[i]["class"] == cls):
			recom = rec_data[i]["recommendation"]
			print("Recommendation: ", recom)
		if recom == "": #if there is no recommendation for a prediction
			recom = "Ask Supervisor for Recommendation"		                     		
	with open("runs/detect/" + pred_path + str(pred_time) +"_predictons.json", 'r') as f:
		listobj = json.load(f)
		listobj.append({"classid": cls,"label": label,"Confidence": conf,"recommendation": recom})
	with open("runs/detect/" + pred_path + str(pred_time) +"_predictons.json", 'w') as fp:
		json.dump(listobj,fp,indent=4,separators=(',',': '))
	time.sleep(0)

@app.route('/camera_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommendations', methods=("POST","GET"))
def show_results():
	df_results = pd.read_json('runs/detect/09_13_2022-124144_predictons.json')
	return render_template('recommendations.html', tables=[df_results.to_html(classes='data',header = "true")])

#@app.route('/plot')
#def show_plot():
#	df_results = pd.read_json('runs/detect/09_13_2022-124144_predictons.json')
#	fig = px.bar(df_results, x ='label',barmode ='group')
#	graphJSON = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
#	
#	return render_template('plot.html',graphJSON=graphJSON)
		
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='rtmp://192.168.178.20:1935/live', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--frame-rate', default=0, type=int, help='sample rate')
    opt = parser.parse_args()
    app.run(host='0.0.0.0', threaded=True, port=5001)

