import os
import sys
import shutil
import argparse

import cv2 as cv
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False, default="../data/train.mp4")
    args = parser.parse_args()
    globals().update(vars(args))

    base_path = ".".join(path.split(".")[:-1])
    if os.path.exists(base_path) and os.path.isdir(base_path):
        shutil.rmtree(base_path)
    os.mkdir(base_path)

    cap = cv.VideoCapture(path)
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(first_frame.shape[:-1]+(2,)).astype(np.float32)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 2, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 1] = magnitude
        # crop ego-car (should be before optical flow calc.)
        mask[350:,...] = 0
        # dumps mask
        np.save(os.path.join(base_path, "frame_{}.npy".format(i)), mask)
        # Updates previous frame
        prev_gray = gray
        i += 1

    cap.release()
    cv.destroyAllWindows()
