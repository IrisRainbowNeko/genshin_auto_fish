from environment import Fishing_sim
import time
from tkinter import *
import threading
import numpy as np

env=Fishing_sim()

# mouse callback function
def drag(event):
    env.drag()

w,h=500,100
root = Tk()
root.geometry('500x100')
cv = Canvas(root, bg='black', width=w, height=h)
root.bind("<Button-1>", drag)

low=cv.create_rectangle(int(env.low*w)-3, 0, int(env.low*w)+3, h, fill='blue')
high=cv.create_rectangle(int((env.low+env.len)*w)-3, 0, int((env.low+env.len)*w)+3, h, fill='green')
pointer=cv.create_rectangle(int(env.pointer*w)-3, 0, int(env.pointer*w)+3, h, fill='red')
cv.pack()

def update():
    env.tick()
    cv.coords(low, int(env.low*w)-3, 0, int(env.low*w)+3, h)
    cv.coords(high, int((env.low+env.len)*w)-3, 0, int((env.low+env.len)*w)+3, h)
    cv.coords(pointer, int(env.pointer*w)-3, 0, int(env.pointer*w)+3, h)
    root.after(int(np.round(1000/env.tick_count)), update)
root.after(int(np.round(1000/env.tick_count)), update)
root.mainloop()