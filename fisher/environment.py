import numpy as np
from utils import *
import cv2
from pymouse import *
import pyautogui
import win32api, win32con
import time
from copy import deepcopy
from collections import Counter
import traceback
import os

class FishFind:
    def __init__(self, predictor, show_det=True):
        self.predictor = predictor
        self.food_imgs = [
            cv2.imread('./imgs/food_gn.png'),
            cv2.imread('./imgs/food_cm.png'),
            cv2.imread('./imgs/food_bug.png'),
            cv2.imread('./imgs/food_fy.png'),
        ]
        self.ff_dict={'hua jiang':0, 'ji yu':1, 'die yu':2, 'jia long':3, 'pao yu':3}
        self.dist_dict={'hua jiang':130, 'ji yu':80, 'die yu':80, 'jia long':80, 'pao yu':80}
        self.food_rgn=[580,400,740,220]
        self.last_fish_type='hua jiang'
        self.show_det=show_det
        os.makedirs('img_tmp/', exist_ok=True)

    def get_fish_types(self, n=12, rate=0.6):
        counter = Counter()
        fx = lambda x: int(np.sign(np.cos(np.pi * (x / (n // 2)) + 1e-4)))
        for i in range(n):
            obj_list = self.predictor.image_det(cap())
            if obj_list is None:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 70 * fx(i), 0, 0, 0)
                time.sleep(0.1)
                continue
            cls_list = set([x[0] for x in obj_list])
            counter.update(cls_list)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 70 * fx(i), 0, 0, 0)
            time.sleep(0.2)
            # pyautogui.moveRel(50, 0, duration=0.5)
            # for u in range(1,51):
            #    mosue.move(sx + u, sy)
            #    time.sleep(0.003)
        fish_list = [k for k, v in dict(counter).items() if v / n >= rate]
        return fish_list

    def throw_rod(self, fish_type):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(1)

        def move_func(dist):
            if dist>100:
                return 50 * np.sign(dist)
            else:
                return (abs(dist)/2.5+10) * np.sign(dist)

        for i in range(50):
            try:
                obj_list, outputs, img_info = self.predictor.image_det(cap(), with_info=True)
                if self.show_det:
                    cv2.imwrite(f'img_tmp/det{i}.png', self.predictor.visual(outputs[0],img_info))

                rod_info = sorted(list(filter(lambda x: x[0] == 'rod', obj_list)), key=lambda x: x[1], reverse=True)
                if len(rod_info)<=0:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, np.random.randint(-50,50), np.random.randint(-50,50), 0, 0)
                    time.sleep(0.1)
                    continue
                rod_info=rod_info[0]
                rod_cx = (rod_info[2][0] + rod_info[2][2]) / 2
                rod_cy = (rod_info[2][1] + rod_info[2][3]) / 2

                fish_info = min(list(filter(lambda x: x[0] == fish_type, obj_list)),
                                key=lambda x: distance((x[2][0]+x[2][2])/2, (x[2][1]+x[2][3])/2, rod_cx, rod_cy))

                if (fish_info[2][0] + fish_info[2][2]) > (rod_info[2][0] + rod_info[2][2]):
                    #dist = -self.dist_dict[fish_type] * np.sign(fish_info[2][2] - (rod_info[2][0] + rod_info[2][2]) / 2)
                    x_dist = fish_info[2][0] - self.dist_dict[fish_type] - rod_cx
                else:
                    x_dist = fish_info[2][2] + self.dist_dict[fish_type] - rod_cx

                print(x_dist, (fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3])
                if abs(x_dist)<30 and abs((fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3])<30:
                    break

                dx = move_func(x_dist)
                #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, fish_info[2][2] - rod_info[2][2] + 50, (fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3], 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, move_func((fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3]), 0, 0)
            except Exception as e:
                traceback.print_exc()
            #time.sleep(0.3)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def select_food(self, fish_type):
        pyautogui.press('f')
        time.sleep(1)
        pyautogui.click(1650, 790, button=pyautogui.SECONDARY)
        time.sleep(0.5)
        bbox_food = match_img(cap(self.food_rgn), self.food_imgs[self.ff_dict[fish_type]], type=cv2.TM_CCOEFF_NORMED)
        pyautogui.click(bbox_food[4]+self.food_rgn[0], bbox_food[5]+self.food_rgn[1])
        time.sleep(0.1)
        pyautogui.click(1183, 756)

    def do_fish(self, fish_init=True):
        if fish_init:
            self.fish_list = self.get_fish_types()
        if self.fish_list[0]!=self.last_fish_type:
            self.select_food(self.fish_list[0])
            self.last_fish_type = self.fish_list[0]
        self.throw_rod(self.fish_list[0])

class Fishing:
    def __init__(self, delay=0.1, max_step=100, show_det=True, predictor=None):
        self.mosue = PyMouse()
        self.t_l = cv2.imread('./imgs/target_left.png')
        self.t_r = cv2.imread('./imgs/target_right.png')
        self.t_n = cv2.imread('./imgs/target_now.png')
        self.im_bar = cv2.imread('./imgs/bar2.png')
        self.bite = cv2.imread('./imgs/bite.png', cv2.IMREAD_GRAYSCALE)
        self.std_color=np.array([192,255,255])
        self.r_ring=21
        self.delay=delay
        self.max_step=max_step
        self.count=0
        self.show_det=show_det

        self.add_vec=[0,2,0,2,0,2]

    def reset(self):
        self.y_start = self.find_bar()[0]
        self.img=cap([712 - 10, self.y_start, 496 + 20, 103])

        self.fish_start=False
        self.zero_count=0
        self.step_count=0
        self.reward=0
        self.last_score=self.get_score()

        return self.get_state()

    def drag(self):
        self.mosue.click(1630,995)

    def do_action(self, action):
        if action==1:
            self.drag()

    def scale(self, x):
        return (x-5-10)/484

    def find_bar(self, img=None):
        img = cap(region=[700, 0, 520, 300]) if img is None else img[:300, 700:700+520, :]
        bbox_bar = match_img(img, self.im_bar)
        if self.show_det:
            img=deepcopy(img)
            cv2.rectangle(img, bbox_bar[:2], bbox_bar[2:4], (0, 0, 255), 1)  # 画出矩形位置
            cv2.imwrite(f'../img_tmp/bar.jpg', img)
        return bbox_bar[1]-9, bbox_bar

    def is_bite(self):
        img = cap(region=[1595, 955, 74, 74])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_output = cv2.Canny(gray, 50, 150)
        return psnr(self.bite, edge_output)>10

    def get_state(self, all_box=False):
        bar_img=self.img[2:34,:,:]
        bbox_l = match_img(bar_img, self.t_l)
        bbox_r = match_img(bar_img, self.t_r)
        bbox_n = match_img(bar_img, self.t_n)

        bbox_l = tuple(list_add(bbox_l, self.add_vec))
        bbox_r = tuple(list_add(bbox_r, self.add_vec))
        bbox_n = tuple(list_add(bbox_n, self.add_vec))

        if self.show_det:
            img=deepcopy(self.img)
            cv2.rectangle(img, bbox_l[:2], bbox_l[2:4], (255, 0, 0), 1)  # 画出矩形位置
            cv2.rectangle(img, bbox_r[:2], bbox_r[2:4], (0, 255, 0), 1)  # 画出矩形位置
            cv2.rectangle(img, bbox_n[:2], bbox_n[2:4], (0, 0, 255), 1)  # 画出矩形位置
            fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontScale = 1
            thickness = 1
            cv2.putText(img, str(self.last_score), (257+30, 72), fontScale=fontScale,fontFace=fontFace, thickness=thickness, color=(0,255,255))
            cv2.putText(img, str(self.reward), (257+30, 87), fontScale=fontScale,fontFace=fontFace, thickness=thickness, color=(255,255,0))
            cv2.imwrite(f'./img_tmp/{self.count}.jpg',img)
        self.count+=1

        #voc dataset
        '''cv2.imwrite(f'./bar_dataset/{self.count}.jpg', self.img)
        with open(f'./bar_dataset/{self.count}.xml', 'w', encoding='utf-8') as f:
            f.write(self.voc_tmp.format(self.count, *bbox_l[:4], *bbox_r[:4], *bbox_n[:4]))'''
        if all_box:
            return bbox_l, bbox_r, bbox_n
        else:
            return self.scale(bbox_l[4]),self.scale(bbox_r[4]),self.scale(bbox_n[4])

    def get_score(self):
        cx,cy=247+10,72
        for x in range(4,360,2):
            px=int(cx+self.r_ring*np.sin(np.deg2rad(x)))
            py=int(cy-self.r_ring*np.cos(np.deg2rad(x)))
            if np.mean(np.abs(self.img[py,px,:]-self.std_color))>5:
                return x//2-2
        return 360//2-2

    def step(self, action):
        self.do_action(action)

        time.sleep(self.delay-0.05)
        self.img=cap([712 - 10, self.y_start, 496 + 20, 103])
        self.step_count+=1

        score=self.get_score()
        if score>0:
            self.fish_start=True
            self.zero_count=0
        else:
            self.zero_count+=1
        self.reward=score-self.last_score
        self.last_score=score

        return self.get_state(), self.reward, (self.step_count>self.max_step or (self.zero_count>=15 and self.fish_start) or score>176)

    def render(self):
        pass

class Fishing_sim:
    def __init__(self, bar_range=(0.18, 0.4), move_range=(30,60*2), resize_freq_range=(15,60*5),
                 move_speed_range=(-0.3,0.3), tick_count=60, step_tick=15, stop_tick=60*15,
                 drag_force=0.4, down_speed=0.015, stable_speed=-0.32, drawer=None):
        self.bar_range=bar_range
        self.move_range=move_range
        self.resize_freq_range=resize_freq_range
        self.move_speed_range=(move_speed_range[0]/tick_count, move_speed_range[1]/tick_count)
        self.tick_count=tick_count

        self.step_tick=step_tick
        self.stop_tick=stop_tick
        self.drag_force=drag_force/tick_count
        self.down_speed=down_speed/tick_count
        self.stable_speed=stable_speed/tick_count

        self.drawer=drawer

        self.reset()

    def reset(self):
        self.len = np.random.uniform(*self.bar_range)
        self.low = np.random.uniform(0,1-self.len)
        self.pointer = np.random.uniform(0,1)
        self.v=0

        self.resize_tick = 0
        self.move_tick = 0
        self.move_speed = 0

        self.score = 100
        self.ticks = 0

        return (self.low,self.low+self.len,self.pointer)

    def drag(self):
        self.v=self.drag_force

    def move_bar(self):
        if self.move_tick<=0:
            self.move_tick=np.random.uniform(*self.move_range)
            self.move_speed=np.random.uniform(*self.move_speed_range)
        self.low=np.clip(self.low+self.move_speed, a_min=0, a_max=1-self.len)
        self.move_tick-=1

    def resize_bar(self):
        if self.resize_tick<=0:
            self.resize_tick=np.random.uniform(*self.resize_freq_range)
            self.len=min(np.random.uniform(*self.bar_range),1-self.low)
        self.resize_tick-=1

    def tick(self):
        self.ticks+=1
        if self.pointer>self.low and self.pointer<self.low+self.len:
            self.score+=1
        else:
            self.score-=1

        if self.ticks>self.stop_tick or self.score<=-100000:
            return True

        self.pointer+=self.v
        self.pointer=np.clip(self.pointer, a_min=0, a_max=1)
        self.v=max(self.v-self.down_speed, self.stable_speed)

        self.move_bar()
        self.resize_bar()
        return False

    def do_action(self, action):
        if action==1:
            self.drag()

    def get_state(self):
        return self.low,self.low+self.len,self.pointer

    def step(self, action):
        self.do_action(action)

        done=False
        score_before=self.score
        for x in range(self.step_tick):
            if self.tick():
                done=True
        return self.get_state(), (self.score-score_before)/self.step_tick, done

    def render(self):
        if self.drawer:
            self.drawer.draw(self.low, self.low+self.len,self.pointer,self.ticks)

