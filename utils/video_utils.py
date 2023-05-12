import threading
import collections
import itertools
import time
import cv2
import logging
import numpy as np
from threading import Thread
from datetime import datetime
import imageio
from torch import Tensor, prod
import os 
import argparse


class VideoCaptureThreading:
    def __init__(self, src=0, resolution = None,skip=None, wait = None,fps=None,fourcc=None):
        self.src = src
        
        self.resolution = resolution
        self.fourcc = fourcc
        self.skip = skip
        self.init_cap()
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.fps = FPS()
        self.frames_grabbed = 0
        self.wait = wait
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)


    def set(self, var1, var2):
        self.cap.set(var1, var2)


    def init_cap(self):
        self.cap = cv2.VideoCapture(self.src)
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.resolution is not None:
            width,height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.wait:
                threading.Event().wait(self.wait)
            if self.skip:
                for i in range(self.skip):
                    self.cap.grab()
            grabbed, frame = self.cap.read()
            fps = self.fps()
            if grabbed:
                self.frames_grabbed+=1
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                if self.frames_grabbed % 20 == 1:
                    pass
                    #logging.debug(f"Camera {self.src} thread fps: {fps:.2f}")
            else:
                # try:
                #     new_path = os.system('sudo ls /dev/video*')
                #     self.src=new_path
                #     self.cap.release()
                #     self.init_cap()
                # except:
                logging.warning("Camera thread Frame not grabbed!")
                
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame
    def generator(self):
        while True:
            with self.read_lock:
                frame = self.frame.copy()
                grabbed = self.grabbed
            if grabbed:
                yield frame
            else:
                break
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

class FPS:
    def __init__(self,decay=0.1):
        self.last_timestamp = time.time()
        self.decay = decay
        self.ema_fps = 0.0

    def __call__(self):
        ts = time.time()
        passed = ts - self.last_timestamp
        self.last_timestamp = ts
        self.ema_fps = 1./passed*self.decay+self.ema_fps*(1.-self.decay)
        return self.ema_fps

class Screenshot():

    def __init__(self,path):
        
        self.flag = False
        self.frame = None
        self.path = path
        self.thread = Thread(target=self.do_screenshot)
        self.success_status = False 
        self.started = False
        self.wait = 0.11

    def get_frame(self,frame):
        self.flag = True
        self.frame = frame

    def make_screenshot(self):
        if self.path is None:
            return False
        success = False
        try:
            imageio.imwrite(''.join([self.path, 'frame_', datetime.now().strftime('%H_%M_%S_%f')[:-3], '.jpg']), self.frame)
            success = True
        except:
            #print("Cannot Do Screenshots\n")
            pass
        return success
    
    def start(self):
        self.started = True
        self.thread.start()

    def do_screenshot(self):
        while True:
            if not self.started:
                return
            if self.flag:
                self.success_status = self.make_screenshot()
                self.flag = False 
            time.sleep(self.wait) # The value passed in sleep should be set based on screenshot-free FPS, 
          
            
    def stop_thread(self):
        self.started = False
        print("Stoped screenshot threading")
        self.thread.join()


def optimize_position(current_pos_x,current_pos_y, target_sz, search_area_scale, resolution):
    size = np.sqrt(prod(target_sz*search_area_scale ))
    width, height = resolution

    center_x = current_pos_x
    center_y = current_pos_y
    

    if current_pos_x<=width/3 and current_pos_y<=height/2:
        if size/2-current_pos_x >0:
            center_x = size/2
        if size/2-current_pos_y >0:
            center_y = size/2

    elif current_pos_x>=width/3 and current_pos_x<=2*width/3  and current_pos_y<=height/2:
        # if size/2-current_pos_x >0:
        #     center_x = size/2
        if size/2-current_pos_y >0:
            
            center_y = size/2

    elif current_pos_x>=2*width/3 and  current_pos_y<=height/2:
        if size/2-width+current_pos_x >0:
            center_x = width-size/2
        if size/2-current_pos_y >0:
            center_y = size/2
    
    elif current_pos_x<=width/3 and current_pos_y>height/2:
        if size/2-current_pos_x >0:
            center_x = size/2
        if size/2-height+current_pos_y >0:
            center_y = height-size/2

    elif current_pos_x>=width/3 and current_pos_x<=2*width/3  and current_pos_y>height/2:
        # if size/2-current_pos_x >0:
        #     center_x = size/2
        if size/2-height+current_pos_y >0:
            
            center_y = height-size/2

    elif current_pos_x>=2*width/3 and  current_pos_y>height/2:
        if size/2-width+current_pos_x >0:
            center_x = width-size/2
        if size/2-height+current_pos_y >0:
            center_y = height-size/2
    
    if size >= height:
        center_y = height/2 
    
    if width-target_sz[1]*search_area_scale < target_sz[1]:
        center_x = width/2 

    return Tensor([center_y, center_x])



def Frames_gen(path,wait=False):
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        if wait:
            cv2.waitKey(1)
        if success:
            yield frame
        else:
            break

def draw_xywh(frame,xywh, bbox, caption=None,color=(0, 255, 0)):
    state = [int(i) for i in xywh]
    cv2.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), color, 5)
    if caption is not None:
        cv2.putText(frame, str(caption), tuple(state[:2]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
                    thickness=1)
        if bbox is not None:
            cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[0] + bbox[2]),(int(bbox[1] + bbox[3]))),(0,0,0),3)

def normalize_image(depth,astype="uint8"):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*1))-1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape)
    return out.astype(astype)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

         
# if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG)
    #cap = VideoCaptureThreading()
    #cap.start()
    #for f in cap.generator():
        #cv2.imshow("f",f)
     #   cv2.waitKey(1)
    # sc = Screenshot("abc")
    # sc.start()