import cv2
from keras.models import load_model
import numpy as np
from collections import deque

# Load pretrained model on MNIST dataset
model = load_model('mnistdawk.h5')

def main():
    cap = cv2.VideoCapture(0)
    # Color range in HSV for color detection
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    points = deque(maxlen=512)
    drawboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((224, 224, 3), dtype=np.uint8)
    pred_class = 0
    
    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        # Color detection function
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Noise redution process
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # Contor detection
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(drawboard, points[i - 1], points[i], (255, 255, 255), 20)
                    cv2.line(img, points[i - 1], points[i], (0, 0, 255), 14)
        elif len(cnts) == 0:
            if len(points) != []:
                drawboard_gray = cv2.cvtColor(drawboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(drawboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                drawboard_cnts,hh = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                if len(drawboard_cnts) >= 1:
                    cnt = max(drawboard_cnts, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = drawboard_gray[y:y + h, x:x + w]
                        pred_proba, pred_class = num_predict(model, digit)
                        print("The number you have drawn is ",pred_class,)

            points = deque(maxlen=512)
            drawboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow("Frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def num_predict(model, image):
    processed = process_image(image) 
    pred_class = model.predict_classes(processed)
    pred_proba = model.predict_proba(processed)
    return pred_proba, pred_class

def process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    cv2.imshow('digit',img)
    img = np.array(img, dtype=np.uint8)
    img= img/255
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

num_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()