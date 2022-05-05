# video path most often used:
# /home/mlucio/Documents/CS/gui/media/band.mp4
# /home/mlucio/Documents/CS/gui/media/osu_short.mp4
# /home/mlucio/Documents/CS/gui/media/osu_test.mp4
# /home/mlucio/Documents/CS/gui/media/ohio.mp4
# /home/mlucio/Documents/CS/gui/media/zoom_short.mp4
# /home/mlucio/Documents/CS/gui/media/zoom_Trim.mp4
# /home/mlucio/Documents/CS/gui/media/track.mp4
# /home/mlucio/Documents/CS/gui/media/ellipse.mp4
# /home/mlucio/Documents/CS/gui/media/crop.mp4
# /home/mlucio/Documents/CS/gui/media/crop_Trim.mp4

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
import matplotlib.pyplot as plt
import math
#from kapteyn import kmpfit
#from matplotlib.backends._backend_agg import FigureCanvasAgg
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

def fit_ellipse(x, y):
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    #print(S1, S2, S3)
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    den = b**2 - a*c
    if den > 0:
        return None, None, None, None, None
    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)
    #print(e)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    #print(type(phi))
    if not isinstance(phi, np.float):
        return None, None, None, None, None
    phi = phi % np.pi
    return x0, y0, ap, bp, phi

def intersactionPoints(a,b,h,k,x1,y1,x2,y2):
    #xi1, yi1, xi2, yi2 <- intersection points
    xi1, yi1, xi2, yi2, aa, bb, cc, m = 0, 0, 0, 0, 0, 0, 0, 0
    if x1 != x2:
        m = (y2 - y1)/(x2 - x1)
        c = y1 - m * x1
        aa = b * b + a * a * m * m
        bb = 2 * a * a * c * m - 2 * a * a * k * m - 2 * h * b * b
        cc = b * b * h * h + a * a * c * c - 2 * a * a * k * c + a * a * k * k - a * a * b * b
    else:
        # vertical line case
        aa = a * a
        bb = -2.0 * k * a * a
        cc = -a * a * b * b + b * b * (x1 - h) * (x1 - h)
    d = bb * bb - 4 * aa * cc
    # intersection points : (xi1,yi1) and (xi2,yi2)
    if d > 0:
        if (x1 != x2):
            xi1 = (-bb + (d**0.5)) / (2 * aa)
            xi2 = (-bb - (d**0.5)) / (2 * aa)
            yi1 = y1 + m * (xi1 - x1)
            yi2 = y1 + m * (xi2 - x1)
        else:
            yi1 = (-bb + (d**0.5)) / (2 * aa)
            yi2 = (-bb - (d**0.5)) / (2 * aa)
            xi1 = x1
            xi2 = x1
    return xi1, yi1, xi2, yi2

class Page:
    def __init__(self, root):
        self.icon = ImageTk.PhotoImage(Image.open("gui/media/note.png"))
        root.iconphoto(True, self.icon)
        root.title("Marching Band Drill Analysis")
        self.original_file = ""
        self.video_file = ""
        self.canvas = tk.Canvas(root, width = 950, height = 500)
        self.canvas.pack()
        self.tk_img = ImageTk.PhotoImage(Image.open("yolo/media/back.jpg"))
        self.canvas.create_image(0, 0, image = self.tk_img)
        self.mainframe = ttk.Frame(root)
        self.l = tk.Label(root, text = "Marching Band Drill Analysis - Matthew Lucio", bg = "#00a4ff", fg = "white", font = ("Arial", 15, "bold"), width = 120, height = 3, anchor = CENTER)
        self.l_copy = tk.Label(root, text = "", bg = "#00a4ff", fg = "white", width = 150, height = 6, anchor = CENTER)
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
        #self.title = tk.Text(root, height = 10, width = 50)
        #self.l = tk.Label(root, text = "Marching Band Drill Analysis Senior Research Project - Matthew Lucio")
        #self.title.insert(tk.END, self.l)
        #self.l.pack()
        #self.test_button = self.canvas.create_image(250, 200, image=Image, anchor='c')
        self.read_video_button = tk.Button(
            root, text = "Read Video", command = self.read_video, width = 9, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.read_video_button_window = self.canvas.create_window(10, 55, anchor = "nw", window = self.read_video_button)
        #self.read_video_button_window.attributes()
        self.yolo5_button_marchers = tk.Button(
            root, text = "Combine", command = lambda: self.make_progress_bar(version = "combine"), width = 10, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.yolo5_button_window = self.canvas.create_window(105, 55, anchor = "nw", window = self.yolo5_button_marchers)
        self.yolo5_button_lines = tk.Button(
            root, text = "Ellipse", command = lambda: self.make_progress_bar(version = "ellipse"), width = 10, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.yolo5_lines_button_window = self.canvas.create_window(215, 55, anchor = "nw", window = self.yolo5_button_lines)
        self.yolo5_button_numbers = tk.Button(
            root, text = "Horizontal Positions Calculation", command = lambda: self.make_progress_bar(version = "yardlines"), width = 30, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.yolo5_button_numbers_window = self.canvas.create_window(325, 55, anchor = "nw", window = self.yolo5_button_numbers)
        self.line_button = tk.Button(
            root, text = "Vertical Marcher Line", command = lambda: self.make_progress_bar(version = "line"), width = 22, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.line_button_window = self.canvas.create_window(570, 55, anchor = "nw", window = self.line_button)
        self.show_video_button = tk.Button(
            root, text = "Show Video", command = self.stream_video, width = 11, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.show_video_button_window = self.canvas.create_window(760, 55, anchor = "nw", window = self.show_video_button)
        self.quit_button = tk.Button(
            root, text = "Quit", command = root.destroy, width = 7, bd = 0, bg = "#00a4ff", fg = "white", underline = 0)
        self.quit_button_window = self.canvas.create_window(870, 55, anchor = "nw", window = self.quit_button)
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
        root.bind("c", lambda event: self.make_progress_bar(version = "combine"))
        root.bind("e", lambda event: self.make_progress_bar(version = "ellipse"))
        root.bind("h", lambda event: self.make_progress_bar(version = "yardlines"))
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
            self.yolo5_button_marchers["state"] = tk.NORMAL
            self.yolo5_button_lines["state"] = tk.NORMAL
            self.yolo5_button_numbers["state"] = tk.NORMAL
            self.line_button["state"] = tk.NORMAL
            self.show_video_button["state"] = tk.NORMAL
    
    def make_progress_bar(self, version, *args):
        self.pb.pack()
        self.read_video_button["state"] = tk.DISABLED
        self.yolo5_button_marchers["state"] = tk.DISABLED
        self.yolo5_button_lines["state"] = tk.DISABLED
        self.yolo5_button_numbers["state"] = tk.DISABLED
        self.line_button["state"] = tk.DISABLED
        self.show_video_button["state"] = tk.DISABLED
        t1 = threading.Thread(target = self.update, args = [root, version])
        t1.start()

    def update(self, root, version):
        if version == "combine":
            self.combine("yolov5/new_number.pt", "yolov5/new_best.pt", "numbers", "marchers")
            #self.yolo5_video("yolov5/new_best.pt", "combine")
        elif version == "ellipse":
            self.yolo5_video("yolov5/ellipse.pt", "ellipse", 0)
        elif version == "yardlines":
            x_50, y_50, x_40l, y_40l, x_40r, y_40r = self.yolo5_video("yolov5/new_number.pt", "yardnumbers", 0)
            #print((x_50, y_50, x_40l, y_40l, x_40r, y_40r))
            self.yolo5_video("yolov5/new_best.pt", "yardlines", (x_50, y_50, x_40l, y_40l, x_40r, y_40r))
        elif version == "mask":
            self.mask_video()
        elif version == "line":
            self.line_video()
        self.root.event_generate("<<PhishDoneEvent>>")

    def combine(self, w1, w2, t1, t2):
        self.pb["value"] = 0
        #print(self.video_file)
        self.yolo5_video(w1, t1, 0)
        self.line_video()
        #
        self.yolo5_video(w2, t2, 0)
        #print(self.video_file)

    def yolo5_video(self, w, t, n):
        #print(t)
        
        yolo5_file = self.video_file.split(".")[0] + "_yolo5_" + t + ".avi"
        
        #print(self.video_file)
        (x_50, y_50, x_40l, y_40l, x_40r, y_40r) = self.run(self.video_file, weights = w, transformation = t, number = n)
        if t != "yardnumbers":
            self.video_file = yolo5_file
        if x_50 != None:
            return (x_50, y_50, x_40l, y_40l, x_40r, y_40r)
        else:
            return None

    @torch.no_grad()
    def run(self, source, weights, transformation, number, imgsz=[640, 640], conf_thres = 0.5, iou_thres = 0.45, max_det = 1000, device = "", view_img = True, save_txt = False, save_crop = False, nosave = False, classes = None, agnostic_nms = False, augment = False, visualize = False, update = False, line_thickness = 1, hide_labels = True, hide_conf = True, half = False, dnn = False):
        source = str(source)
        filename, ext = source.split(".")
        output = filename + "_yolo5_" + transformation + ".avi"
        white = filename + "_yolo5_" + transformation + "_white" + ".avi"
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        vid = cv2.VideoCapture(self.video_file)
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
        if transformation == "yardnumbers": 
            hide_labels = False
            conf_thres = 0.6
            #numbers = {"30l": 0, "40l": 0, "50": 0, "40r": 0, "30r": 0}
            x_50 = 0
            y_50 = 0
            x_40l = 0
            y_40l = 0
            x_40r = 0
            y_40r = 0
            num_50 = 0
            num_40l = 0
            num_40r = 0
        if transformation == "yardlines":
            hide_labels = False
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
            #max_chi = 100
            prev_50 = 5
            prev_40l = 5
            prev_40r = 5
            prev_50_c1 = (0, 0)
            prev_50_c2 = (0, 0)
            prev_40l_c1 = (0, 0)
            prev_40l_c2 = (0, 0)
            prev_40r_c1 = (0, 0)
            prev_40r_c2 = (0, 0)
            m_50 = 1
            b_50 = 1
            m_40l = 1
            b_40l = 1
            m_40r = 1
            b_40r = 1
            for i, det in enumerate(pred):  # per image
                self.pb["value"] += (100/(total - 1))/3
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
                #print(str(names))
                w, h = im0.shape[1], im0.shape[0]
                im1 = np.zeros([h, w, 3], dtype = np.uint8)
                im1.fill(255)
                #im2 = im0.copy()
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if transformation == "yardlines":
                        #print(number)
                        d_50 = False
                        d_40l = False
                        d_40r = False
                        blur = cv2.GaussianBlur(im0, (9,9), 0)
                        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                        minLineLength = 50
                        maxLineGap = 1000
                        cannyThreshold1 = 20
                        cannyThreshold2 = 60
                        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 40, minLineLength=minLineLength, maxLineGap=maxLineGap)
                        drawn_line_list = []
                        if lines is not None:
                            for line in lines:
                                for x1,y1,x2,y2 in line:
                                    if abs((y2 - y1)) > abs((x2 - x1)):
                                        new_line_flag = True
                                        for start_pos in drawn_line_list:
                                            if ((x1 > start_pos) and x1 < (start_pos + 100)) or (x1 > (start_pos - 100) and (x1 < start_pos)):
                                                new_line_flag = False
                                        if new_line_flag:    
                                            d_50 = False
                                            d_40l = False
                                            d_40r = False
                                            mind = 10000
                                            if x2 != x1:
                                                a = (y2 - y1)/(x2 - x1)
                                            else:
                                                a = 10000
                                            c = y1 - (a * x1)
                                            mindex = 0
                                            for ind in range(3):
                                                x3 = number[2 * ind]
                                                y3 = number[2 * ind + 1]
                                                e = 1/a * number[2 * ind] + number[2 * ind + 1]
                                                m1 = np.array([[a, -1], [(-1/a), -1]])
                                                m2 = np.array([-c, -e])
                                                m3 = np.linalg.solve(m1, m2)
                                                d = math.dist((x3, y3), (m3[0], m3[1]))
                                                if d < mind:
                                                    mindex = ind
                                                    mind = d
                                            if mind < 25:
                                                if mindex == 0 and m_50 == 1:
                                                    # 50
                                                    d_50 = True
                                                    cv2.line(im0, (x1, y1), (x2, y2),(255, 0, 0), 3)
                                                    m_50 = a
                                                    b_50 = c
                                                    prev_50 = 5
                                                    prev_50_c1 = (x1, y1)
                                                    prev_50_c2 = (x2, y2)
                                                elif mindex == 1 and m_40l == 1:
                                                    # 40l
                                                    d_40l = True
                                                    cv2.line(im0, (x1, y1), (x2, y2),(0, 255, 0), 3)
                                                    m_40l = a
                                                    b_40l = c
                                                    prev_40l = 5
                                                    prev_40l_c1 = (x1, y1)
                                                    prev_40l_c2 = (x2, y2)
                                                elif mindex == 2 and m_40r == 1:
                                                    # 40r
                                                    d_40r = True
                                                    cv2.line(im0, (x1, y1), (x2, y2),(0, 0, 255), 3)
                                                    m_40r = a
                                                    b_40r = c
                                                    prev_40r = 5
                                                    prev_40r_c1 = (x1, y1)
                                                    prev_40r_c2 = (x2, y2)
                                            else:
                                                cv2.line(im0, (x1, y1), (x2, y2),(255, 255, 255), 3)
                                        drawn_line_list.append(x1)
                                    else:
                                        pass
                        if not d_50:
                            #print("50")
                            cv2.line(im0, prev_50_c1, prev_50_c2,(255, 0, 0), 3)
                        if not d_40l:
                            #print("40l")
                            cv2.line(im0, prev_40l_c1, prev_40l_c2,(0, 255, 0), 3)
                        if not d_40r:
                            #print("40r")
                            cv2.line(im0, prev_40r_c1, prev_40r_c2,(0, 0, 255), 3)
                        #cv2.circle(im0, (int(abs(number[0])), int(abs(number[1]))), radius = 3, color = (0, 0, 0), thickness = 5, lineType=cv2.LINE_AA)
                        #cv2.circle(im0, (int(abs(number[2])), int(abs(number[3]))), radius = 3, color = (0, 0, 0), thickness = 5, lineType=cv2.LINE_AA)
                        #cv2.circle(im0, (int(abs(number[4])), int(abs(number[5]))), radius = 3, color = (0, 0, 0), thickness = 5, lineType=cv2.LINE_AA)
                    if transformation == "ellipse":
                        ellipse_x = []
                        ellipse_y = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if transformation == "numbers":
                                annotator.box_label(xyxy, label, color = colors(c, True))
                            elif transformation == "yardnumbers":
                                annotator.box_label(xyxy, label, color = colors(c, True))
                            elif transformation == "yardlines":
                                x, y, width, height = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                #print(x, y)
                                #print(label, type(label))
                                if x > 300 and x < 690:
                                    #print(label, type(label))
                                    ud = b_50 + m_50 * x - y
                                    #print(ud)
                                    if m_50 != 1 and m_40r != 1 and m_40l != 1:
                                        d_50 = (abs(m_50*x + -1 * y + b_50))/(math.sqrt(m_50**2 + b_50**2))
                                        d_40l = (abs(m_40l*x + -1 * y + b_40l))/(math.sqrt(m_40l**2 + b_40l**2))
                                        d_40r = (abs(m_40r*x + -1 * y + b_40r))/(math.sqrt(m_40r**2 + b_40r**2))
                                        #print(d_50, d_40l, d_40r)
                                        d_40 = d_40l if ud > 0 else d_40r
                                        #print(d_40l)
                                        prop = d_40/(d_40 + d_50)
                                        #print(prop, im0.shape[0])
                                        if d_40l < d_40r:
                                            #print("working")
                                            h = round(40 + prop * 10, 2)
                                        else:
                                            h = round(60 - prop * 10, 2)
                                        annotator.box_label(xyxy, str(h), color = (0, 0, 0), small = True)
                                    elif m_50 != 1 and m_40r != 1 and ud < 0:
                                        #print("right")
                                        d_50 = (abs(m_50*x + -1 * y + b_50))/(math.sqrt(m_50**2 + b_50**2))
                                        d_40r = (abs(m_40r*x + -1 * y + b_40r))/(math.sqrt(m_40r**2 + b_40r**2))
                                        prop = d_40r/(d_40r + d_50)
                                        #print(prop, im0.shape[0])
                                        h = round(60 - prop * 10, 2)
                                        annotator.box_label(xyxy, str(h), color = (0, 0, 0), small = True)
                                    elif m_50 != 1 and m_40l != 1 and ud > 0:
                                        #print("left")
                                        d_50 = (abs(m_50*x + -1 * y + b_50))/(math.sqrt(m_50**2 + b_50**2))
                                        d_40l = (abs(m_40l*x + -1 * y + b_40l))/(math.sqrt(m_40l**2 + b_40l**2))
                                        prop = d_40l/(d_40l + d_50)
                                        h = round(40 + prop * 10, 2)
                                        annotator.box_label(xyxy, str(h), color = (0, 0, 0), small = True)
                                    else:    
                                        label = None
                                        annotator.box_label(xyxy, label, color = (0, 0, 0))
                                else:
                                    label = None
                                    annotator.box_label(xyxy, label, color = (0, 0, 0))
                            else:
                                annotator.box_label(xyxy, label, color = (0, 0, 0))
                            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x, y, width, height = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            center = (int(p1[0] + p1[1]/2), h - min(p1[0], p1[1]))
                            if transformation == "marchers":
                                cv2.circle(im1, ((int((x + width)/2)), int((y + height)/2)), radius = 2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
                            elif transformation == "yardnumbers":
                                #print(names[c], center[0], type(names[c]))
                                #cv2.circle(im0, ((int((x + width)/2)), int((y + height)/2)), radius = 2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
                                if names[c] == "50":
                                    x_50 += (x + width)/2
                                    y_50 += (y + height)/2
                                    num_50 += 1
                                elif names[c] == "40" and x - im0.shape[0] < -10:
                                    #print("40l")
                                    x_40l += (x + width)/2
                                    y_40l += (y + height)/2
                                    num_40l += 1
                                elif names[c] == "40" and x - im0.shape[0] > 10:
                                    #print("40r")
                                    x_40r += (x + width)/2
                                    y_40r += (y + height)/2
                                    num_40r += 1
                            elif transformation == "ellipse":
                                ellipse_x.append(int((x + width)/2))
                                ellipse_y.append(int((y + height)/2))
                                #cv2.circle(im0, ((int((x + width)/2)), int((y + height)/2)), radius = 2, color = (0, 0, 0), thickness = 3, lineType=cv2.LINE_AA)
                                cv2.circle(im1, ((int((x + width)/2)), int(height)), radius = 2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
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
                                po1 = (x + right, y + int(gray.shape[1] - 1))
                                po2 = (x + left, y)
                                cv2.rectangle(im1, p1, p2, color = (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    if transformation == "ellipse":
                        ellipse_x = np.array(ellipse_x)
                        ellipse_y = np.array(ellipse_y)
                        if len(ellipse_x) > 10:
                            coeffs = fit_ellipse(ellipse_x, ellipse_y)
                            x0, y0, ap, bp, phi = cart_to_pol(coeffs)
                            if x0 != None:
                                total = 0
                                for j in range(len(ellipse_x)):
                                    #print(ap, bp)
                                    x, y = ellipse_x[j], ellipse_y[j]
                                    x1 = (x - x0) * math.cos(phi) + (y - y0) * math.sin(phi)
                                    y1 = - (x - x0) * math.sin(phi) + (y - y0) * math.cos(phi)
                                    xi1, yi1, xi2, yi2 = intersactionPoints(ap,bp,x0,y0,x1,y1,0,0)
                                    #print(xi1, yi1, xi2, yi2)
                                    if xi1 != 0:
                                        d1 = math.dist((x, y), (xi1, yi1))
                                    else: 
                                        d1 = 10000
                                    if xi2 != 0:
                                        d2 = math.dist((x, y), (xi2, yi2))
                                    else: 
                                        d2 = 10000
                                    total += (min(d1, d2))**2
                                #print(num/len(ellipse_x))
                                #print(total/len(ellipse_x))
                                if total/len(ellipse_x) < 60000000:
                                    cv2.ellipse(im0, ((x0, y0), (2 * ap, 2 * bp), -1 * phi), (0, 0, 0), 2)
                                    cv2.ellipse(im1, ((x0, y0), (2 * ap, 2 * bp), -1 * phi), (0, 0, 0), 2)
                
                im0 = annotator.result()
                if view_img:
                    #self.imgtk = ImageTk(self)
                    # switch between im0 or im1 to show video vs just dots on white background
                    frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)).resize((960, 400)))
                    self.lmain.config(image=frame_image)
                    self.lmain.image = frame_image
                    #cv2.imshow(str(p), im1)
                    #cv2.waitKey(1)  # 1 millisecond
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(output, im1)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_dir:  # new video
                            vid_path[i] = output
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if isinstance(white_writer[i], cv2.VideoWriter):
                                white_writer[i].release()
                            if vid_cap:  # 
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_dir+= '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            white_writer[i] = cv2.VideoWriter(white, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                        white_writer[i].write(im1)
        if transformation == "yardnumbers" and num_50 != 0 and num_40l != 0 and num_40r != 0:
            #print(fifty)
            #print(num)
            return (x_50/num_50, y_50/num_50, x_40l/num_40l, y_40l/num_40l, x_40r/num_40r, y_40r/num_40r)
        #print(min_error)
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        return (None, None, None, None, None, None)

    def mask_video(self):
        self.pb["value"] = 0
        mask_file = self.original_file.split(".")[0] + "_mask.avi"
        self.video_file = mask_file
        mask = "mrcnn/mask-rcnn-coco"
        CONFIDENCE = 0.5
        threshold = 0.3
        labelsPath = os.path.sep.join([mask, "object_detection_classes_coco.txt"])
        LABELS = open(labelsPath).read().strip().split("\n")
        RED_COLOR = np.array([255, 0, 0]) 
        WHITE_COLOR = np.array([255, 255, 255]) 
        weightsPath = os.path.sep.join([mask, "frozen_inference_graph.pb"])
        configPath = os.path.sep.join([mask,"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
        vs = cv2.VideoCapture(self.original_file)
        writer = None
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            #print("[INFO] {} total frames in video".format(total))
        except:
            #print("[INFO] could not determine # of frames in video")
            total = -1
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            (boxes, masks) = net.forward(["detection_out_final","detection_masks"])
            end = time.time()
            for i in range(0, boxes.shape[2]):
                classID = int(boxes[0, 0, i, 1])
                confidence = boxes[0, 0, i, 2]
                if confidence > CONFIDENCE:
                    (H, W) = frame.shape[:2]
                    box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxW = endX - startX
                    boxH = endY - startY
                    mask = masks[i, classID]
                    mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > threshold)
                    roi = frame[startY:endY, startX:endX][mask]
                    blended = ((0.4 * RED_COLOR) + (0.6 * roi)).astype("uint8")
                    frame[startY:endY, startX:endX][mask] = blended
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255,255,255), 2)
                    text = "{}: {:.4f}".format("Person", confidence)
                    cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(mask_file, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
                if total > 0:
                    elap = (end - start)
            writer.write(frame)
            self.pb["value"] += (100/(total - 1))/3

    def line_video(self):
        #self.pb["value"] = 0
        line_file = self.video_file.split(".")[0] + "_line.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoCapture(self.video_file)
        _, image = vid.read()
        h, w = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(line_file, fourcc, 20.0, (w, h))
        vid = cv2.VideoCapture(self.video_file)
        low_colour = np.array([120,60,70], dtype="uint8")
        high_colour = np.array([138,128,118], dtype="uint8")
        img_height = 778
        img_width = 1381
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = float(vid.get(prop))
        while True: 
            self.pb["value"] += (100/(total - 1))/3
            bool, img_original = vid.read()
            print(type(img_original))
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
                                cv2.line(img_original,(x1,y1),(x2,y2),(255,255,255),3)
                                mean = int((x1 + x2) / 2)
                                grad = (y2 - y1) / (x2 - x1)
                                if grad > 0:
                                    grad_array.append(grad)
                                    lines_len += 1
                            drawn_line_list.append(x1)
                        else:
                            pass
            grad_array.sort()    
            #print(len(grad_array))
            found = True
            if(len(grad_array) == 0 or len(grad_array) == 1):
                found = False
                #print("no yard lines found")
                #break
            elif (len(grad_array) % 2) == 0:
                i = int(len(grad_array) / 2)
                grad_average = (grad_array[i] + grad_array[i-1]) / 2
            elif (len(grad_array) % 2) != 0:
                i = int(len(grad_array) / 2)
                #print(i)
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
                mean_los = int((x_1 + x_3) / 2)
                combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)
            out.write(img_original)
            frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)).resize((960, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
                #self.pb["value"] += 100/(total - 1)
        self.video_file = line_file
        vid.release()
        cv2.destroyAllWindows()

    def video_done(self, event = None):
        self.read_video_button["state"] = tk.NORMAL
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
        #"""
        self.show_video_button["state"] = tk.DISABLED
        video = imageio.get_reader(self.video_file)
        for image in video.iter_data():
            #if stop_event.is_set():
                #break
            if self.show_video_button["state"] == tk.NORMAL:
                self.show_video_button["state"] = tk.DISABLED
            frame_image = ImageTk.PhotoImage(Image.fromarray(image).resize((960, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
        self.show_video_button["state"] = tk.NORMAL
        #"""
        """
        video = imageio.get_reader(self.video_file)
        while self.stop()
            frame_image = ImageTk.PhotoImage(Image.fromarray(image).resize((760, 400)))
            self.lmain.config(image=frame_image)
            self.lmain.image = frame_image
        """
        
        

if __name__ == "__main__":
    root = Tk()
    Page(root)
    root.geometry("950x500+200+100")
    root.mainloop()