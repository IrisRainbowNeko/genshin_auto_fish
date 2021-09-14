import pyautogui
import keyboard

i=0
while True:
    keyboard.wait('t')
    img = pyautogui.screenshot()
    img.save(f'img_tmp/{i}.png')
    i+=1

