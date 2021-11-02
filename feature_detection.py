import numpy as np
import cv2
import argparse
import dlib
import imutils


def shape_to_numpy_array(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates


def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, 0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image = cv2.imread("ted_cruz.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

np.savetxt("madhu1.txt", shape, fmt='%.0f')

points = [];
with open("madhu1.txt") as file:
    for line in file:
        x, y = line.split()
        points.append((int(x), int(y)))

for p in points:
    draw_point(image, p, (255, 0, 235))

img_name = "snapshot6.png"
cv2.imshow(img_name, image)
cv2.waitKey(0)
