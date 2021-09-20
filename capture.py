import time

import pyautogui
import keyboard
import winsound

'''i=0
while True:
    keyboard.wait('t')
    img = pyautogui.screenshot()
    img.save(f'img_tmp/{i}.png')
    i+=1'''

print('ok')
keyboard.wait('t')
for i in range(56,56+20):
    img = pyautogui.screenshot()
    img.save(f'fish_dataset/{i}.png')
    time.sleep(0.5)
winsound.Beep(500, 500)