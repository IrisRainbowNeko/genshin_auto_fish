import pyautogui
import cv2
import numpy as np
import time

def match_img(img, target):
    h, w = target.shape[:2]
    res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return (*max_loc, max_loc[0] + w, max_loc[1] + h, max_loc[0] + w//2, max_loc[1] + h//2)

img = cv2.imread('imgs/a.png')[94:94 + 103, 712:712 + 496, :]
t_l = cv2.imread('imgs/target_left.png')
t_r = cv2.imread('imgs/target_right.png')
t_n = cv2.imread('imgs/target_now.png')

start=time.time()
img2 = pyautogui.screenshot(region=[712, 94, 496, 103])
bbox_l=match_img(img, t_l)
cv2.rectangle(img, bbox_l[0:2], bbox_l[2:4], (255,0,0), 2)  # 画出矩形位置
bbox_r=match_img(img, t_r)
cv2.rectangle(img, bbox_r[0:2], bbox_r[2:4], (0,255,0), 2)  # 画出矩形位置
bbox_n=match_img(img, t_n)
cv2.rectangle(img, bbox_n[0:2], bbox_n[2:4], (0,0,255), 2)  # 画出矩形位置
end=time.time()
print(end-start)
cv2.imshow('a',img)
cv2.waitKey()

'''
#959,166 r=21 ring center
img = pyautogui.screenshot(region=[712, 94, 496, 103])  # x,y,w,h
img.save('bar.png')'''

