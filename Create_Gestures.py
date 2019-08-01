#Creating our own dataset
import cv2
import numpy as np
import os

#Global parameters of our image,i.e, width = 50 and height = 50
image_x, image_y = 50, 50

#Creating a function to create a folder
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


#Creating a function to store images in our dataest from gestures
def store_images(gesture_id):
    total_pics = 1200    #For each gesture we will  be storing 1200 images
    cap = cv2.VideoCapture(0)      #Opening the webcam
    x, y, w, h = 300, 50, 350, 350    #Specifying particular area on the screen to detect gestures to nullify any noises present out there
    
    create_folder("gestures/" + str(gesture_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    while True:
        ret, frame = cap.read()  
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))     #so that the colour lies between the  skin colour
        res = cv2.bitwise_and(frame, frame, mask = mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)
        
        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations = 2)                #reducing the noise in the image
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)        #further smoothening the image
        
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                save_img = cv2.resize(save_img, (image_x, image_y))
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("gestures/" + str(gesture_id) + "/" + str(pic_no) + ".jpg", save_img)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 255, 255))
            cv2.imshow("Capturing gesture", frame)
            cv2.imshow("thresh", thresh)
            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                if flag_start_capturing == False:
                    flag_start_capturing = True
                else:
                    flag_start_capturing = False
                    frames = 0
                
            if flag_start_capturing == True:
                frames += 1
            if pic_no == total_pics:
                break


#create_folder(gestures)
gesture_id = input("Enter gesture no.: ")
store_images(gesture_id)