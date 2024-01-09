'''
    Title: "get_data.py"
    Author: Ben Brixton
    Last modified: 18/09/23
    Description:    - Get user to draw letters
                    - Store mouse position at intervals
                    - Store data in files to be used in 
                      text recognission machine learning
                      program
'''

import cv2
import numpy as np
import os
import time

def makeWindow():
    cv2.namedWindow('Window')
    cv2.setMouseCallback('Window', draw)
    while(1):
        cv2.imshow('Window', img)
        k=cv2.waitKey(1)&0xFF
        if(k==27):
            break
    cv2.destroyWindow('Window')

def draw(event, x, y, flags, param):
    global drawing, output
    
    if (event == cv2.EVENT_LBUTTONDOWN):
        drawing = True
    elif (event == cv2.EVENT_LBUTTONUP):
        drawing = False
    elif (event == cv2.EVENT_MOUSEMOVE):
        if(drawing == True):
            cv2.circle(img,(x,y),1,(255,255,255),-1)
            output.append([x/500, (500-y)/500, time.time()])
        else:
            if (output):
                output.append(['NaN', 'NaN', time.time()])

def saveLetter(folder, letter, iter):
    while(output[-1][0] == 'NaN'):
        output.pop()
    dirname = os.path.dirname(__file__)
    os.makedirs(os.path.join(dirname, f'{folder}'), exist_ok=True)
    with open(os.path.join(dirname, f'{folder}/{letter}{iter}.csv'), 'w') as f:
        for pair in (output):
            f.write(f"{pair[0]},{pair[1]},{pair[2]}\n")

def getLetter(folder, letter, number):
    global img, output
    for i in range(number):
        output.clear()
        print(f"Draw \"{letter}\"")
        img = np.zeros((500,500,3), np.uint8)
        makeWindow()
        saveLetter(folder, letter, i)

if(__name__ == "__main__"):
    output = []
    img = np.zeros((500,500,3), np.uint8)
    drawing = False

    getLetter('duration_aratio_data', 'a', 20)
    getLetter('duration_aratio_data', 'b', 20)
    getLetter('duration_aratio_data', 'c', 20)
    getLetter('duration_aratio_data', 'd', 20)
    getLetter('duration_aratio_data', 'e', 20)
    getLetter('duration_aratio_data', 'f', 20)
    getLetter('duration_aratio_data', 'g', 20)
    getLetter('duration_aratio_data', 'h', 20)
    getLetter('duration_aratio_data', 'i', 20)
    getLetter('duration_aratio_data', 'j', 20)
    getLetter('duration_aratio_data', 'k', 20)
    getLetter('duration_aratio_data', 'l', 20)
    getLetter('duration_aratio_data', 'm', 20)
    getLetter('duration_aratio_data', 'n', 20)
    getLetter('duration_aratio_data', 'o', 20)
    getLetter('duration_aratio_data', 'p', 20)
    getLetter('duration_aratio_data', 'q', 20)
    getLetter('duration_aratio_data', 'r', 20)
    getLetter('duration_aratio_data', 's', 20)
    getLetter('duration_aratio_data', 't', 20)
    getLetter('duration_aratio_data', 'u', 20)
    getLetter('duration_aratio_data', 'v', 20)
    getLetter('duration_aratio_data', 'w', 20)
    getLetter('duration_aratio_data', 'x', 20)
    getLetter('duration_aratio_data', 'y', 20)
    getLetter('duration_aratio_data', 'z', 20)
