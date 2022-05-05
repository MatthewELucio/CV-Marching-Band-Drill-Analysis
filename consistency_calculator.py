from tkinter import *
from tkinter import ttk
import imageio
import cv2
import numpy as np
import os
import time
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import os
import sys

# file location: 
# /home/mlucio/Documents/CS/gui/test.txt
# /home/mlucio/Documents/CS/gui/new.txt

coords = {}
height = int(10 * 53.3)
width = 10 * 100

def getCoords(s):
    horizontal, vertical = s.split("ln")
    horizontal, vertical = horizontal.strip(), vertical.strip()

    num = horizontal.split(" ")[0]
    length = horizontal.split(" ")[1]

    temp = horizontal.split("yd")[0].strip()
    yard = temp.split(" ")[-1]

    if "On" in horizontal:
        side = horizontal.split(":")[0].split(" ")[-1]
        if side == "1":
            x = float(yard) #50?
        else:
            x = 100 - float(yard)
        #print(yard)
    else:
        side = horizontal.split(":")[0].split(" ")[-1]
        #print(side1)
        shift = horizontal.split(":")[1].strip().split(" ")[0]
        #print(shift1)
        inout = ""
        if "inside" in horizontal:
            inout = "inside"
        else:
            inout = "outside"
        if side == "1":
            if inout == "inside":
                x = float(yard) + 5/8 * float(shift)
            else:
                x = float(yard) - 5/8 * float(shift)
        else:
            if inout == "inside":
                x = 100 - float(yard) - 5/8 * float(shift)
            else:
                x = 100 - float(yard) + 5/8 * float(shift)
    #print(vertical)            
    if "Front" in vertical:
        if "Hash" in vertical:
            #y = float(vertical.split(" ")[0])
            if "behind" in vertical:
                y = 17.78 + 5/8 * float(vertical.split(" ")[0])
            elif "On" in vertical:
                y = 17.78
            else:
                y = 17.78 - 5/8 * float(vertical.split(" ")[0])
        else:
            if "On" in vertical:
                y = 0
            else:
                y = 5/8 * float(vertical.split(" ")[0])
    else:
        if "Hash" in vertical:
            if "behind" in vertical:
                y = 35.56 + 5/8 * float(vertical.split(" ")[0])
            elif "On" in vertical:
                y = 35.56
            else:
                y = 35.56 - 5/8 * float(vertical.split(" ")[0])
        else:
            if "On" in vertical:
                y = 53.33
            else:
                y = 53.33 - float(vertical.split(" ")[0])
    coords[num] = [(x, y), length]
    return x, y

def distance(c1, c2):
    return math.sqrt((c2[1] - c1[1])**2 + (c2[0] - c1[0])**2)

def printString(x, y):
    new_x = ""
    new_y = ""
    if round(x, 3) == 50:
        new_x = "On 50 yd ln"
    else:
        for i in range(0, 100, 5):
            #print(i)
            if abs(x - i) < 0.05 and x < 50:
                new_x = "Side 1: On " + str(i) + " yd ln"
                break
            elif abs(x - i) < 2.4 and x - i > 0 and x < 50:
                dif = round(8/5 * abs(x - i), 3)
                new_x = "Side 1: " + str(dif) + " steps inside " + str(i) + " yd ln"
                break
            elif abs(x - i) < 2.4 and x - i < 0 and x < 50:
                dif = round(8/5 * abs(x - i), 3)
                new_x = "Side 1: " + str(dif) + " steps outside " + str(i) + " yd ln"
                break
            elif abs(x - i) > 2.4 and abs(x - i) <= 2.5 and x < 50:
                new_x = "Side 1: 4.0 steps inside " + str(i) + " yd ln"
                break
            elif abs(x - i) < 0.05 and x > 50:
                new_x = "Side 2: On " + str(100 - i) + " yd ln"
            elif abs(x - i) < 2.4 and x - i > 0 and x > 50:
                dif = round(8/5 * abs(x - i), 3)
                new_x = "Side 2: " + str(dif) + " steps outside " + str(100 - i) + " yd ln"
                break
            elif abs(x - i) < 2.4 and x - i < 0 and x > 50:
                dif = round(8/5 * abs(x - i), 3)
                new_x = "Side 2: " + str(dif) + " steps inside " + str(100 - i) + " yd ln"
                break
            elif abs(x - i) > 2.4 and abs(x - i) <= 2.5 and x > 50:
                new_x = "Side 2: 4.0 steps inside " + str(100 - i) + " yd ln"
                break
            else:
                hi = 1
        #print("done")
    if abs(y - 17.78) < 0.1:
        new_y = "On Front Hash (HS)"
    elif abs(y - 35.56) < 0.1:
        new_y = "On Back Hash (HS)"
    elif y < 0.1:
        new_y = "On Front side line"
    elif y > 53.2:
        new_y = "On Back side line"
    else:
        if y < 8.89:
            dif = round((8/5 * y), 3)
            new_y = str(dif) + " steps behind Front side line" 
        elif y < 17.78:
            dif = round(8/5 * (17.78 - y), 3)
            new_y = str(dif) + " steps in front of Front Hash (HS)" 
        elif y < 26.67:
            dif = round(8/5 * (y - 17.78), 3)
            new_y = str(dif) + " steps behind Front Hash (HS)"
        elif y < 35.56:
            dif = round(8/5 * (35.56 - y), 3)
            new_y = str(dif) + " steps in front of Back Hash (HS)" 
        elif y < 44.45:
            dif = round(8/5 * (y - 35.56), 3)
            new_y = str(dif) + " steps behind Back Hash (HS)" 
        else:
            dif = round(8/5 * (53.33 - y), 3)
            new_y = str(dif) + " steps in front of Back side line" 
    return(str(new_x) + " " + str(new_y))

class Page:
    def __init__(self, root):
        self.icon = ImageTk.PhotoImage(Image.open("gui/media/drum.png"))
        root.iconphoto(True, self.icon)
        root.title("Marching Band Consistency Calculator")
        self.canvas = tk.Canvas(root, width = 1200, height = 600)
        self.canvas.pack()
        self.tk_img = ImageTk.PhotoImage(Image.open("gui/media/bg.jpg"))
        self.canvas.create_image(0, 0, image = self.tk_img)
        self.mainframe = ttk.Frame(root)
        self.mainframe.place(x = 10, y = 10)
        self.mainframe.grid_remove()
        self.mainframe.pack()
        self.l = tk.Label(root, text = "Marching Band Consistency Calculator - Matthew Lucio", bg = "#90ee90", fg = "green", font = ("Arial", 15, "bold"), width = 120, height = 3, anchor = CENTER)
        self.l_copy = tk.Label(root, text = "", bg = "#90ee90", fg = "green", width = 250, height = 7, anchor = CENTER)
        self.l.place(x = -100, y = 0)
        self.l_copy.place(x = 0, y = 50)
        self.file = ""
        self.read_text_button = tk.Button(
            root, text = "Read Text File", command = self.read_text, width = 12, bd = 0, bg = "#006400", fg = "white", underline = 0)
        self.read_text_button_window = self.canvas.create_window(8, 60, anchor = "nw", window = self.read_text_button)
        self.find_positions_button = tk.Button(
            root, text = "Find Positions", command = self.find_positions, width = 12, bd = 0, bg = "#006400", fg = "white", underline = 0)
        self.find_positions_button_window = self.canvas.create_window(120, 60, anchor = "nw", window = self.find_positions_button)
        self.position_input = Text(root, height = 1, width = 12, bg = "light yellow")
        self.position_input.place(x = 385, y = 80)
        self.find_position_button = tk.Button(
            root, text = "Find Position at Set (Type Below)", command = lambda : self.find_position(self.position_input.get("1.0", "end-1c")), width = 31, bd = 0, bg = "#006400", fg = "white", underline = 5)
        self.find_position_button_window = self.canvas.create_window(235, 60, anchor = "nw", window = self.find_position_button)
        self.set_1_input = Text(root, height = 1, width = 7, bg = "light yellow")
        self.set_1_input.place(x = 635, y = 80)
        self.set_2_input = Text(root, height = 1, width = 7, bg = "light yellow")
        self.set_2_input.place(x = 725, y = 80)
        self.find_distance_button = tk.Button(
            root, text = "Distance Between Sets (Ex: 1) and (Ex: 2)", command = lambda : self.find_distance(self.set_1_input.get("1.0", "end-1c"), self.set_2_input.get("1.0", "end-1c")), width = 39, bd = 0, bg = "#006400", fg = "white", underline = 0)
        self.find_distance_button_window = self.canvas.create_window(480, 60, anchor = "nw", window = self.find_distance_button)
        self.consistency_input = Text(root, height = 1, width = 10, bg = "light yellow")
        self.consistency_input.place(x = 1040, y = 80)
        self.consistency_button = tk.Button(
            root, text = "Find Subset with Sets using Decimal (Ex: 0.5)", command = lambda : self.find_consistency(self.set_1_input.get("1.0", "end-1c"), self.set_2_input.get("1.0", "end-1c"), self.consistency_input.get("1.0", "end-1c")), width = 43, bd = 0, bg = "#006400", fg = "white", underline = 5)
        self.consistency_button_window = self.canvas.create_window(790, 60, anchor = "nw", window = self.consistency_button)
        self.quit_button = tk.Button(
            root, text = "Quit", command = root.destroy, width = 7, bd = 0, bg = "#006400", fg = "white", underline = 0)
        self.quit_button_window = self.canvas.create_window(1120, 60, anchor = "nw", window = self.quit_button)
        self.find_positions_button["state"] = tk.DISABLED
        self.find_position_button["state"] = tk.DISABLED
        self.find_distance_button["state"] = tk.DISABLED
        self.consistency_button["state"] = tk.DISABLED
        root.bind("r", self.read_text)
        root.bind("f", self.find_positions)
        root.bind("p", lambda event: self.find_position(self.position_input.get("1.0", "end-1c")))
        root.bind("d", lambda event: self.find_distance(self.set_1_input.get("1.0", "end-1c"), self.set_2_input.get("1.0", "end-1c")))
        root.bind("s", lambda event: self.find_consistency(self.set_1_input.get("1.0", "end-1c"), self.set_2_input.get("1.0", "end-1c"), self.consistency_input.get("1.0", "end-1c")))
        root.bind("q", quit)
        self.root = root

    def read_text(self, event = None):
        self.file = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))
        self.find_positions_button["state"] = tk.NORMAL
        

    def find_positions(self, event = None):
        #print(self.file)
        with open(self.file, newline = "") as rf:
            blank_image = np.zeros((height, width, 3), np.uint8)
            for i in range(len(blank_image)):
                for j in range(len(blank_image[i])):
                    blank_image[i][j][1] = 255
            #print(blank_image)
            cv2.line(blank_image, (0, height), (width, height), color = (255, 255, 255), thickness = 5)
            cv2.line(blank_image, (0, 1), (width, 1), color = (255, 255, 255), thickness = 5)
            cv2.line(blank_image, (0, 0), (0, height), color = (255, 255, 255), thickness = 5)
            cv2.line(blank_image, (width, 0), (width, height), color = (255, 255, 255), thickness = 5)
            for i in range(20):
                if i == 10:
                    cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (0, 0, 255), thickness = 3)   
                else: 
                    cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (255, 255, 255), thickness = 3)
            for i in range(100):
                if i % 5 != 0:
                    cv2.line(blank_image, (10 * i, int(height/3)), (10 * i, 5 + int(height/3)), color = (255, 255, 255), thickness = 1)
                    cv2.line(blank_image, (10 * i, int(2 * height/3)), (10 * i, 5 + int(2 * height/3)), color = (255, 255, 255), thickness = 1)
            lines = rf.readlines()
            for i in range(len(lines) - 1):
                prev_x_val, prev_y_val = getCoords(lines[i].strip())
                #printString(prev_x_val, prev_y_val)
                next_x_val, next_y_val = getCoords(lines[i + 1].strip())
                if prev_x_val != next_x_val or prev_y_val != next_y_val:
                    cv2.arrowedLine(blank_image, (int(10 * prev_x_val), height - int(10 * prev_y_val)), (int(10 * next_x_val), height - int(10 * next_y_val)), color = (0, 0, 255), thickness = 2, tipLength = 0.25)
                cv2.circle(blank_image, center = (int(10 * prev_x_val), height - int(10 * prev_y_val)), radius = 5, color = (255, 0, 0), thickness = -1)
            last_x_val, last_y_val = getCoords(lines[len(lines) - 1].strip())
            cv2.circle(blank_image, center = (int(10 * last_x_val), height - int(10 * last_y_val)), radius = 5, color = (255, 0, 0), thickness = -1)
            self.imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)).resize((803, 400)))
            self.picFrame = Frame(root)
            self.picFrame.place(x = 200, y = 160)
            self.label = tk.Label(self.picFrame, borderwidth = 0, highlightthickness = 0)
            self.label.config(image=self.imgtk)
            self.label.image = self.imgtk
            self.label.pack()
            self.find_position_button["state"] = tk.NORMAL
            self.find_distance_button["state"] = tk.NORMAL
            self.consistency_button["state"] = tk.NORMAL
            
    def find_position(self, setnumber, event = None):
        x_val, y_val = coords[setnumber][0][0], coords[setnumber][0][1]
        #print("position in the marching band coordinate system: ")
        string_coords = printString(x_val, y_val)
        blank_image = np.zeros((height, width,3), np.uint8)
        for i in range(len(blank_image)):
            for j in range(len(blank_image[i])):
                blank_image[i][j][1] = 255
        #print(blank_image)
        cv2.line(blank_image, (0, height), (width, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 1), (width, 1), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 0), (0, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (width, 0), (width, height), color = (255, 255, 255), thickness = 5)
        for i in range(20):
            if i == 10:
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (0, 0, 255), thickness = 3)   
            else: 
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (255, 255, 255), thickness = 3)
        for i in range(100):
            if i % 5 != 0:
                cv2.line(blank_image, (10 * i, int(height/3)), (10 * i, 5 + int(height/3)), color = (255, 255, 255), thickness = 1)
                cv2.line(blank_image, (10 * i, int(2 * height/3)), (10 * i, 5 + int(2 * height/3)), color = (255, 255, 255), thickness = 1)
        cv2.circle(blank_image, center = (int(10 * x_val), height - int(10 * y_val)), radius = 5, color = (255, 0, 0), thickness = -1) 
        self.imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)).resize((803, 400)))
        self.picFrame = Frame(root)
        self.picFrame.place(x = 200, y = 160)
        self.label = tk.Label(self.picFrame, borderwidth = 0, highlightthickness = 0)
        self.label.config(image=self.imgtk)
        self.label.image = self.imgtk
        self.label.pack()
        self.txt = tk.Label(root, text = string_coords, bg = "#90ee90").place(x = 5, y = 100)


    def find_distance(self, choice1, choice2):
        prev_c = coords[choice1][0]
        next_c = coords[choice2][0]
        d = distance(prev_c, next_c)
        if prev_c[0] != next_c[0]:
            theta = math.atan((prev_c[1] - next_c[1])/(prev_c[0] - next_c[0]))
            #print(theta)
        else:
            theta = 0
        dis = str(round(8/5 * d, 3)) + " steps"
        #print("angle between the points:", round((180 * theta)/math.pi, 3), "degrees (from the marcher's perspective)")
        #print("into x and y components: ", round(d * math.cos(theta), 3), ", ", round(d * math.sin(theta), 3), " (in yards) and ", round(8/5 * d * math.cos(theta), 3), ", ", round(8/5 * d * math.sin(theta), 3), " (in steps)", sep = "")
        blank_image = np.zeros((height, width,3), np.uint8)
        for i in range(len(blank_image)):
            for j in range(len(blank_image[i])):
                blank_image[i][j][1] = 255
        #print(blank_image)
        cv2.line(blank_image, (0, height), (width, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 1), (width, 1), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 0), (0, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (width, 0), (width, height), color = (255, 255, 255), thickness = 5)
        for i in range(20):
            if i == 10:
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (0, 0, 255), thickness = 3)   
            else: 
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (255, 255, 255), thickness = 3)
        for i in range(100):
            if i % 5 != 0:
                cv2.line(blank_image, (10 * i, int(height/3)), (10 * i, 5 + int(height/3)), color = (255, 255, 255), thickness = 1)
                cv2.line(blank_image, (10 * i, int(2 * height/3)), (10 * i, 5 + int(2 * height/3)), color = (255, 255, 255), thickness = 1)
        cv2.circle(blank_image, center = (int(10 * prev_c[0]), height - int(10 * prev_c[1])), radius = 5, color = (255, 0, 0), thickness = -1) 
        cv2.circle(blank_image, center = (int(10 * next_c[0]), height - int(10 * next_c[1])), radius = 5, color = (255, 255, 0), thickness = -1)    
        if prev_c[0] != next_c[0] or prev_c[1] != next_c[1]:
            cv2.arrowedLine(blank_image, (int(10 * prev_c[0]), height - int(10 * prev_c[1])), (int(10 * next_c[0]), height - int(10 * next_c[1])), color = (0, 0, 255), thickness = 2, tipLength = 0.25) 
        self.imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)).resize((803, 400)))
        self.picFrame = Frame(root)
        self.picFrame.place(x = 200, y = 160)
        self.label = tk.Label(self.picFrame, borderwidth = 0, highlightthickness = 0)
        self.label.config(image=self.imgtk)
        self.label.image = self.imgtk
        self.label.pack()
        self.txt = tk.Label(root, text = dis, bg = "#90ee90").place(x = 550, y = 100)

    def find_consistency(self, choice1, choice2, sub):
        #print(type(sub), choice1, choice2)
        prev_c = coords[choice1][0]
        next_c = coords[choice2][0]
        d = distance(prev_c, next_c)
        if prev_c[0] != next_c[0]:
            theta = math.atan((next_c[1] - prev_c[1])/(next_c[0] - prev_c[0]))
            #print(theta)
        else:
            theta = 0
        #print(sub, d, theta, d * math.cos(theta), d * math.sin(theta))
        if prev_c[0] > next_c[0]:
            new_x = prev_c[0] - float(sub) * d * math.cos(theta)
            new_y = prev_c[1] - float(sub) * d * math.sin(theta)
        else:
            new_x = prev_c[0] + float(sub) * d * math.cos(theta)
            new_y = prev_c[1] + float(sub) * d * math.sin(theta)
        #print("nth subset: ", round(new_x, 3), ", ", round(new_y, 3), " (in yards)", sep = "")
        #print("and in Marching Band coordinate system:")
        string_coords = printString(new_x, new_y)
        blank_image = np.zeros((height, width,3), np.uint8)
        for i in range(len(blank_image)):
            for j in range(len(blank_image[i])):
                blank_image[i][j][1] = 255
        #print(blank_image)
        cv2.line(blank_image, (0, height), (width, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 1), (width, 1), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (0, 0), (0, height), color = (255, 255, 255), thickness = 5)
        cv2.line(blank_image, (width, 0), (width, height), color = (255, 255, 255), thickness = 5)
        for i in range(20):
            if i == 10:
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (0, 0, 255), thickness = 3)   
            else: 
                cv2.line(blank_image, (50 * i, 0), (50 * i, width), color = (255, 255, 255), thickness = 3)
        for i in range(100):
            if i % 5 != 0:
                cv2.line(blank_image, (10 * i, int(height/3)), (10 * i, 5 + int(height/3)), color = (255, 255, 255), thickness = 1)
                cv2.line(blank_image, (10 * i, int(2 * height/3)), (10 * i, 5 + int(2 * height/3)), color = (255, 255, 255), thickness = 1)
        cv2.circle(blank_image, center = (int(10 * prev_c[0]), height - int(10 * prev_c[1])), radius = 5, color = (255, 0, 0), thickness = -1) 
        cv2.circle(blank_image, center = (int(10 * next_c[0]), height - int(10 * next_c[1])), radius = 5, color = (255, 255, 0), thickness = -1)  
        cv2.circle(blank_image, center = (int(10 * new_x), height - int(10 * new_y)), radius = 5, color = (0, 255, 255), thickness = -1)   
        if prev_c[0] != new_x or prev_c[1] != new_y:
            cv2.arrowedLine(blank_image, (int(10 * prev_c[0]), height - int(10 * prev_c[1])), (int(10 * new_x), height - int(10 * new_y)), color = (0, 0, 255), thickness = 2, tipLength = 0.25) 
        if new_x != next_c[0] or new_y != next_c[1]:
            cv2.arrowedLine(blank_image, (int(10 * new_x), height - int(10 * new_y)), (int(10 * next_c[0]), height - int(10 * next_c[1])), color = (0, 0, 255), thickness = 2, tipLength = 0.25)
        self.imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)).resize((803, 400)))
        self.picFrame = Frame(root)
        self.picFrame.place(x = 200, y = 160)
        self.label = tk.Label(self.picFrame, borderwidth = 0, highlightthickness = 0)
        self.label.config(image=self.imgtk)
        self.label.image = self.imgtk
        self.label.pack()
        self.txt = tk.Label(root, text = string_coords, bg = "#90ee90").place(x = 690, y = 100)
        

if __name__ == "__main__":
    root = Tk()
    Page(root)
    root.geometry("1200x600+100+50")
    root.mainloop()