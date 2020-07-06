import argparse
import cv2 as cv
import numpy as np

from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False, default="../data/train.mp4")
    parser.add_argument('--delay', type=int, required=False, default=30)
    parser.add_argument('--speed', action='store_true') # visualize speed
    args = parser.parse_args()
    globals().update(vars(args))
    
    if speed:
        speed_array = openSpeedArr(path, suffix="")
        predicted_speed_array = openSpeedArr(path, suffix="_predicted")

    cap = cv.VideoCapture(path)
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(first_frame.shape).astype(np.uint8)
    mask[..., 1] = 255 # saturation to maximum
    i = 0
    log_array = []
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # draw speed on the orig. frame 
        if speed:
            cv.putText(frame, "Speed: " + speed_array.get(i) + " mph", (15, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv.putText(frame, "Predicted: " + predicted_speed_array.get(i) + " mph", (15, 60), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv.imshow("input", frame)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 2, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normal...)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        
        # # crop ego-car (should be before optical flow calc.)
        # mask[350:, :,[0, 2]] = 0

        # HSV-->BGR
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)
        # Updates previous frame
        prev_gray = gray
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv.destroyAllWindows()
