import time
from utils import *

import keyboard
import winsound
import cv2

'''i=0
while True:
    keyboard.wait('t')
    img = pyautogui.screenshot()
    img.save(f'img_tmp/{i}.png')
    i+=1'''

im_exit = cv2.imread('./imgs/exit.png')

print('ok')
keyboard.wait('t')

exit_pos = match_img(cap_raw(), im_exit)
gvars.genshin_window_rect_img = (exit_pos[0] - 32, exit_pos[1] - 19, DEFAULT_MONITOR_WIDTH, DEFAULT_MONITOR_HEIGHT)

for i in range(56,56+20):
    img = cap()
    img.save(f'fish_dataset/{i}.png')
    time.sleep(0.5)
winsound.Beep(500, 500)