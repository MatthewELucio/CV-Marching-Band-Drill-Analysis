# demo paths:
# /home/mlucio/Documents/CS/yolov5/demo/tj.mp4
# /home/mlucio/Documents/CS/yolov5/demo/vertical.mp4
# /home/mlucio/Documents/CS/yolov5/demo/numbers.mp4
# /home/mlucio/Documents/CS/yolov5/demo/yardlines.mp4

from tkinter import *
from tkinter import ttk, filedialog
from cv2 import RETR_LIST, threshold
import imageio
import threading
import time
import cv2
import numpy as np
import os
import time
import sys
import imutils
import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class Page:
    def __init__(self, root):
        self.icon = ImageTk.PhotoImage(Image.open("gui/media/note.png"))
        root.iconphoto(True, self.icon)
        root.title("Marching Band Drill Analysis")
        self.original_file = ""
        self.video_file = ""
        self.canvas = tk.Canvas(root, width = 750, height = 500)
        self.canvas.pack()
        self.tk_img = ImageTk.PhotoImage(Image.open("yolo/media/back.jpg"))
        self.canvas.create_image(0, 0, image = self.tk_img)
        self.mainframe = ttk.Frame(root)
        self.l = tk.Label(root, text = "Marching Band Drill Analysis - Matthew Lucio", bg = "#00a4ff", fg = "white", font = ("Arial", 15, "bold"), width = 120, height = 3, anchor = CENTER)
        self.l_copy = tk.Label(root, text = "", bg = "#00a4ff", fg = "white", width = 120, height = 6, anchor = CENTER)
        self.l.place(x = -333, y = 0)
        self.l_copy.place(x = 0, y = 50)
        self.mainframe.place(x = 10, y = 10)
        self.mainframe.grid_remove()
        self.videoFrame = Frame(root)
        self.videoFrame.place(x = -5, y = 105)
        self.videoFrame.grid_remove()
        self.lmain = tk.Label(self.videoFrame, borderwidth = 0, highlightthickness = 0)
        self.lmain.pack()
        self.mainframe.pack()
        self.read_video_button = tk.Button(
            root, text = "Read Video", command = self.read_video, width = 9, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.read_video_button_window = self.canvas.create_window(10, 55, anchor = "nw", window = self.read_video_button)
        self.yolo3_button = tk.Button(
            root, text = "Yolov3", command = lambda: self.make_progress_bar(version = "yolo3"), width = 7, bd = 0, bg = "#00a4ff", fg = "white",underline = 5)
        self.yolo3_button_window = self.canvas.create_window(105, 55, anchor = "nw", window = self.yolo3_button)
        self.yolo5_button_marchers = tk.Button(
            root, text = "Yolov5 Marchers", command = lambda: self.make_progress_bar(version = "yolo5_marchers"), width = 14, bd = 0, bg = "#00a4ff", fg = "white", underline = 7)
        self.yolo5_button_window = self.canvas.create_window(185, 55, anchor = "nw", window = self.yolo5_button_marchers)
        self.yolo5_button_lines = tk.Button(
            root, text = "Yolov5 Lines", command = lambda: self.make_progress_bar(version = "yolo5_lines"), width = 10, bd = 0, bg = "#00a4ff", fg = "white", underline = 5)
        self.yolo5_lines_button_window = self.canvas.create_window(315, 55, anchor = "nw", window = self.yolo5_button_lines)
        self.yolo5_button_numbers = tk.Button(
            root, text = "Numbers", command = lambda: self.make_progress_bar(version = "yolo5_numbers"), width = 5, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.yolo5_button_numbers_window = self.canvas.create_window(420, 55, anchor = "nw", window = self.yolo5_button_numbers)
        self.line_button = tk.Button(
            root, text = "Line", command = lambda: self.make_progress_bar(version = "line"), width = 5, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.line_button_window = self.canvas.create_window(490, 55, anchor = "nw", window = self.line_button)
        self.show_video_button = tk.Button(
            root, text = "Show Video", command = self.stream_video, width = 11, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.show_video_button_window = self.canvas.create_window(560, 55, anchor = "nw", window = self.show_video_button)
        self.quit_button = tk.Button(
            root, text = "Quit", command = root.destroy, width = 7, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.quit_button_window = self.canvas.create_window(670, 55, anchor = "nw", window = self.quit_button)
        self.yolo3_button["state"] = tk.DISABLED
        self.yolo5_button_marchers["state"] = tk.DISABLED
        self.yolo5_button_lines["state"] = tk.DISABLED
        self.yolo5_button_numbers["state"] = tk.DISABLED
        self.line_button["state"] = tk.DISABLED
        self.show_video_button["state"] = tk.DISABLED
        root.bind("<<PhishDoneEvent>>", self.video_done)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("purple.Horizontal.TProgressbar", troughcolor = "blue", background = "purple", lightcolor = "blue", darkcolor = "purple", border = "purple")
        self.pb = ttk.Progressbar(root, style = "purple.Horizontal.TProgressbar", orient = HORIZONTAL, length = 750, mode = "determinate")
        self.pb.pack()
        self.pb.pack_forget()
        root.bind("r", self.read_video)
        root.bind("3", lambda event: self.make_progress_bar(version = "yolo3"))
        root.bind("m", lambda event: self.make_progress_bar(version = "yolo5_marchers"))
        root.bind("5", lambda event: self.make_progress_bar(version = "yolo5_lines"))
        root.bind("n", lambda event: self.make_progress_bar(version = "yolo5_numbers"))
        root.bind("l", lambda event: self.make_progress_bar(version = "line"))
        root.bind("s", self.stream_video)
        root.bind("q", quit)
        self.root = root

    def read_video(self, event = None):
        self.original_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("Video files",
                                                            "*.mp4*"),
                                                        ("all files",
                                                            "*.*")))
        self.video_file = self.original_file
        if(os.path.exists(self.original_file)):
            self.yolo3_button["state"] = tk.NORMAL
            self.yolo5_button_marchers["state"] = tk.NORMAL
            self.yolo5_button_lines["state"] = tk.NORMAL
            self.yolo5_button_numbers["state"] = tk.NORMAL
            self.line_button["state"] = tk.NORMAL
            self.show_video_button["state"] = tk.NORMAL
    
    def make_progress_bar(self, version, *args):
        self.pb.pack()
        self.read_video_button["state"] = tk.DISABLED
        self.yolo3_button["state"] = tk.DISABLED
        self.yolo5_button_marchers["state"] = tk.DISABLED
        self.yolo5_button_lines["state"] = tk.DISABLED
        self.yolo5_button_numbers["state"] = tk.DISABLED
        self.line_button["state"] = tk.DISABLED
        self.show_video_button["state"] = tk.DISABLED
        t1 = threading.Thread(target = self.update, args = [root, version])
        t1.start()

    def update(self, root, version):
        if version == "yolo3":
            self.yolo3_video()
        elif version == "yolo5_marchers":
            self.yolo5_video("yolov5/new_best.pt", "marchers")
        elif version == "yolo5_lines":
            self.yolo5_video("yolov5/lines.pt", "lines")
        elif version == "yolo5_numbers":
            self.yolo5_video("yolov5/new_number.pt", "numbers")
        elif version == "line":
            self.line_video()
        self.root.event_generate("<<PhishDoneEvent>>")

    def yolo3_video(self):
        self.pb["value"] = 0
        CONFIDENCE = 0.4
        SCORE_THRESHOLD = 0.5
        IOU_THRESHOLD = 0.5
        config_path = "yolo/yolov3.cfg"
        weights_path = "yolo/yolov3.weights"
        font_scale = 1
        thickness = 1
        labels = open("yolo/coco.names").read().strip().split("\n")
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        cap = cv2.VideoCapture(self.original_file)
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = float(cap.get(prop))
        _, image = cap.read()
        h, w = image.shape[:2]
        yolo3_file = self.original_file.split(".")[0] + "_yolo3.avi"
        self.video_file = yolo3_file
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(yolo3_file, fourcc, 20.0, (w, h))
        while True:
            bool, image = cap.read()
            if not bool:
                break
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.perf_counter()
            layer_outputs = net.forward(ln)
            time_took = time.perf_counter() - start
            boxes, confidences, class_ids = [], [], []
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > CONFIDENCE:
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
            font_scale = 1
            thickness = 1
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    if w < 1000 and h < 1000:
                        cv2.rectangle(image, (x, y), (x + w, y + h),
                                      color=(0, 0, 0), thickness=thickness)
                        cv2.circle(image, (x + int(w/2), y + int(h/2)),
                                   radius=5, color=(0, 0, 0), thickness=-1)
                    text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                    (text_width, text_height) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x
                    text_offset_y = y - 5
                    box_coords = ((text_offset_x, text_offset_y),
                                  (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    if w < 1000 and h < 1000:
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(
                            0, 0, 0), thickness=cv2.FILLED)
                        cv2.circle(overlay, (int(x + int(w/2)), int(y + int(h/2))),
                                   radius=5, color=(0, 0, 0), thickness=-1)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    if w < 1000 and h < 1000:
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)).resize((760, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
            out.write(image)
            i += 1.0
            self.pb["value"] += 100/(total - 1)
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(5)

    def yolo5_video(self, w, t):
        self.pb["value"] = 0
        yolo5_file = self.original_file.split(".")[0] + "_yolo5_" + t + ".avi"
        self.video_file = yolo5_file
        self.run(self.original_file, weights = w, transformation = t)

    @torch.no_grad()
    def run(self, source, weights, transformation, imgsz=[640, 640], conf_thres=0.5, iou_thres=0.45, max_det=1000, device='', view_img=True, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, line_thickness=1, hide_labels=True, hide_conf=True, half=False, dnn=False):
        source = str(source)
        filename, ext = source.split(".")
        output = filename + "_yolo5_" + transformation + ".avi"
        white = filename + "_yolo5_" + transformation + "_white" + ".avi"
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        vid = cv2.VideoCapture(self.original_file)
        total = float(vid.get(prop))
        if is_url and is_file:
            source = check_file(source)  # download
        save_dir = output # increment run
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if transformation == "numbers":
            hide_labels = False
            conf_thres = 0.6
        if pt:
            model.model.half() if half else model.model.float()
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer, white_writer = [None] * bs, [None] * bs, [None] * bs
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):  # per image
                self.pb["value"] += 100/(total - 1)
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                w, h = im0.shape[1], im0.shape[0]
                im1 = np.zeros([h, w, 3], dtype = np.uint8)
                im1.fill(255)
                im2 = im0.copy()
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if transformation == "numbers":
                                annotator.box_label(xyxy, label, color = colors(c, True))
                            else:
                                annotator.box_label(xyxy, label, color = (0, 0, 0))
                            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x, y, width, height = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            c = (int(p1[0] + p1[1]/2), h - min(p1[0], p1[1]))
                            if transformation == "marchers":
                                cv2.circle(im1, ((int((x + width)/2)), int((y + height)/2)), radius = 2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
                            elif transformation == "lines":
                                crop_img = im0[y:height, x:width]
                                blur = cv2.GaussianBlur(crop_img, (5,5), 0)
                                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                                minLineLength = 10
                                maxLineGap = 55
                                cannyThreshold1 = 50
                                cannyThreshold2 = 255
                                edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
                                linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 130, minLineLength=minLineLength, maxLineGap=maxLineGap)
                                if linesP is not None:
                                    for j in range(0, len(linesP)):
                                        l = linesP[j][0]
                                        if abs(l[0] + x - p1[0]) + abs(l[1] + y - p1[1]) < 50 and abs(l[2] + x - p2[0]) + abs(l[3] + y - p2[1]) < 50: # diag down to right check
                                            cv2.line(im0, (l[0] + x, l[1] + y), (l[2] + x, l[3] + y), (0 , 0, 0), 3, cv2.LINE_AA)
                                            cv2.line(im1, (l[0] + x, l[1] + y), (l[2] + x, l[3] + y), (0 , 0, 0), 3, cv2.LINE_AA)
                                        elif abs(l[0]) + abs(l[1] - (height - y)) < 50 and abs(l[2] - (width - x)) + l[3] < 50: #diag from top right to bottom left
                                            if abs(width - x) > 10 and abs(l[0] - l[2]) < 10:
                                                a = 2
                                            else:
                                                cv2.line(im0, (l[0] + x, l[1] + y), (l[2] + x, l[3] + y), (0, 0, 0), 3, cv2.LINE_AA)
                                                cv2.line(im1, (l[0] + x, l[1] + y), (l[2] + x, l[3] + y), (0, 0, 0), 3, cv2.LINE_AA)
                                low_colour = np.array([0,0,0], dtype="uint8")
                                high_colour = np.array([255,255,255], dtype="uint8")
                                mask_img = cv2.inRange(crop_img, low_colour, high_colour)
                                kernel = np.ones((2,2), np.uint8)
                                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=2)
                                contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                                cnt = contours[0]
                                [vx, vy, tx, ty] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.05, 0.05)
                                left = int((-tx*vy/vx) + ty)
                                right = int(((gray.shape[1]-tx)*vy/vx) + ty)
                                cv2.rectangle(im1, p1, p2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                im0 = annotator.result()
                if view_img:
                    frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)).resize((760, 400)))
                    self.lmain.config(image=frame_image)
                    self.lmain.image = frame_image
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(output, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_dir:  # new video
                            vid_path[i] = output
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            #if isinstance(white_writer[i], cv2.VideoWriter):
                               # white_writer[i].release()
                            if vid_cap:  # 
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_dir+= '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            #white_writer[i] = cv2.VideoWriter(white, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                        #white_writer[i].write(im1)
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    def line_video(self):
        self.pb["value"] = 0
        line_file = self.original_file.split(".")[0] + "_line.avi"
        self.video_file = line_file
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoCapture(self.original_file)
        _, image = vid.read()
        h, w = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(line_file, fourcc, 20.0, (w, h))
        vid = cv2.VideoCapture(self.original_file)
        low_colour = np.array([120,60,70], dtype="uint8")
        high_colour = np.array([138,128,118], dtype="uint8")
        img_height = 778
        img_width = 1381
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = float(vid.get(prop))
        while True: 
            self.pb["value"] += 100/(total - 1)
            bool, img_original = vid.read()
            if not bool:
                break
            lower = np.array([210, 210, 210], dtype = "uint8")
            upper = np.array([255, 255, 255], dtype = "uint8")
            mask = cv2.inRange(img_original, lower, upper)
            img = cv2.bitwise_and(img_original, img_original, mask = mask)
            blur = cv2.GaussianBlur(img, (9,9), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            minLineLength = 100
            maxLineGap = 1000
            cannyThreshold1 = 20
            cannyThreshold2 = 60
            img = img_original.copy()
            edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
            grad_array = []
            lines_len = 0
            drawn_line_list = []
            if lines is not None:
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        if (x1 == x2) or (y1 == y2):
                            pass
                        elif abs((y2 - y1)) > abs((x2 - x1)):
                            new_line_flag = True
                            for start_pos in drawn_line_list:
                                if ((x1 > start_pos) and x1 < (start_pos + 100)) or (x1 > (start_pos - 100) and (x1 < start_pos)):
                                    new_line_flag = False
                            if new_line_flag:    
                                cv2.line(img_original,(x1,y1),(x2,y2),(0,255,255),5)
                                mean = int((x1 + x2) / 2)
                                grad = (y2 - y1) / (x2 - x1)
                                if grad > 0:
                                    grad_array.append(grad)
                                    lines_len += 1
                            drawn_line_list.append(x1)
                        else:
                            pass
            grad_array.sort()    
            found = True
            if(len(grad_array) == 0 or len(grad_array) == 1):
                found = False
            elif (len(grad_array) % 2) == 0:
                i = int(len(grad_array) / 2)
                grad_average = (grad_array[i] + grad_array[i-1]) / 2
            elif (len(grad_array) % 2) != 0:
                i = int(len(grad_array) / 2)
                grad_average = (grad_array[i + 1] + grad_array[i] + grad_array[i-1]) / 3
            if found:
                mask_img = cv2.inRange(img_original, low_colour, high_colour)
                kernel = np.ones((2,2), np.uint8)
                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel, iterations=2)
                contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                try: hierarchy = hierarchy[0]
                except: hierarchy = []
                height, width = mask_img.shape
                min_x, min_y = width, height
                max_x = max_y = 0
                x_2 = 1381
                y_2 = 0
                for contour, hier in zip(contours, hierarchy):
                    (x, y, w, h) = cv2.boundingRect(contour)
                    min_x, max_x = min(x, min_x), max(x + w, max_x)
                    min_y, max_y = min(y, min_y), max(y + h, max_y)
                    if (min_x < x_2) and (min_x > x_2):
                        x_2 = min_x
                        y_2 = min_y
                x_1 = int(x_2 - y_2/grad_average)
                x_3 = int((img_height-y_2)/grad_average + x_2)
                x_1 = int((x_1 + x_1) / 2)
                x_3 = int((x_3 + x_3) / 2)
            out.write(img_original)
            frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)).resize((760, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
        vid.release()
        cv2.destroyAllWindows()

    def video_done(self, event = None):
        self.read_video_button["state"] = tk.NORMAL
        self.yolo3_button["state"] = tk.NORMAL
        self.yolo5_button_marchers["state"] = tk.NORMAL
        self.yolo5_button_lines["state"] = tk.NORMAL
        self.yolo5_button_numbers["state"] = tk.NORMAL
        self.line_button["state"] = tk.NORMAL
        self.show_video_button["state"] = tk.NORMAL
        self.pb.pack_forget()

    def stream_video(self, event = None):
        thread = threading.Thread(target = self.show_video, args=())
        thread.daemon = 1
        thread.start()

    def show_video(self):
        self.show_video_button["state"] = tk.DISABLED
        video = imageio.get_reader(self.video_file)
        for image in video.iter_data():
            if self.show_video_button["state"] == tk.NORMAL:
                self.show_video_button["state"] = tk.DISABLED
            frame_image = ImageTk.PhotoImage(Image.fromarray(image).resize((760, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
        self.show_video_button["state"] = tk.NORMAL

if __name__ == "__main__":
    root = Tk()
    Page(root)
    root.mainloop()