import time

import cv2
import pyautogui
import numpy as np
import win32api, win32con

def cap(region=None):
    img = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def mouse_down(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)

def mouse_move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def mouse_up(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

def mouse_click(x, y):
    mouse_down(x, y)
    mouse_up(x, y)

def match_img(img, target, type=cv2.TM_CCOEFF):
    h, w = target.shape[:2]
    res = cv2.matchTemplate(img, target, type)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return (*max_loc, max_loc[0] + w, max_loc[1] + h, max_loc[0] + w // 2, max_loc[1] + h // 2)

def list_add(li, num):
    if isinstance(num, int) or isinstance(num, float):
        return [x+num for x in li]
    elif isinstance(num, list) or isinstance(num, tuple):
        return [x+y for x,y in zip(li,num)]

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))