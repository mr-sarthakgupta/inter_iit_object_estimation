import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours as cntrs
import imutils
from flask import jsonify


def tup(point):
    return (int(point[0]), int(point[1]))


def get_dims_from_one_img(img):
    # img = cv2.imread(img_path)
    scale = 0.25
    h, w = img.shape[:2]
    h = int(scale*h)
    w = int(scale*w)
    img = cv2.resize(img, (w, h))
    copy = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = cv2.inRange(s, 30, 255)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    (contours, _) = cntrs.sort_contours(contours)
    ref_object = contours[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel_horizontal = 0.5*(euclidean(tl, tr) + euclidean(bl, br))
    dist_in_pixel_vertical = 0.5*(euclidean(tl, bl) + euclidean(tr, br))
    dist_in_cm = 2
    pixel_per_cm_horizontal = dist_in_pixel_horizontal/dist_in_cm
    pixel_per_cm_vertical = dist_in_pixel_vertical/dist_in_cm
    # neeche waali line me dhyaan se 0 ki jagah 1 krna h
    contour = contours[0]  # just take the first one
    num_points = 999999
    step_size = 0.01
    percent = step_size
    while num_points >= 6:
        epsilon = percent * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_points = len(approx)
        percent += step_size
    percent -= step_size * 2
    epsilon = percent * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(img, [approx], -1, (0, 0, 200), 2)
    for point in approx:
        point = point[0]  # drop extra layer of brackets
        center = (int(point[0]), int(point[1]))
        cv2.circle(img, center, 4, (150, 200, 0), -1)
    proposals = []
    size = len(approx)
    dims = []
    for a in range(size):
        two = approx[a - 2][0]
        one = approx[a - 1][0]
        curr = approx[a][0]
        dx = two[0] - one[0]
        dy = two[1] - one[1]
        hidden = [curr[0] + dx, curr[1] + dy]
        proposals.append([hidden, curr, a, two])
        c = np.copy(copy)
        cv2.circle(c, tup(two), 4, (255, 0, 0), -1)
        cv2.circle(c, tup(one), 4, (0, 255, 0), -1)
        cv2.circle(c, tup(curr), 4, (0, 0, 255), -1)
        cv2.circle(c, tup(hidden), 4, (255, 255, 0), -1)
        cv2.line(c, tup(two), tup(one), (0, 0, 200), 1)
        cv2.line(c, tup(curr), tup(hidden), (0, 0, 200), 1)
        dims.append((euclidean(two, one) + euclidean(curr, hidden))/2)
    vol_weight = 1
    for d in dims:
        vol_weight *= d
    avg_dims = []
    for k in range(3):
        avg_dims.append((dims[k] + dims[k + 3])/2)
    vol_weight = vol_weight**0.5
    return jsonify({
        'volumetricWeight': vol_weight,
        'length': avg_dims[0],
        'breath': avg_dims[1],
        'height': avg_dims[2],
        'sku': 1
    })
