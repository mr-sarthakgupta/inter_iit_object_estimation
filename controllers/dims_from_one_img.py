import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours as cntrs
import imutils
from flask import jsonify
import requests
import math

def tup(point):
    return (int(point[0]), int(point[1]))


## for cam with the whitener on the brown tripod
def get_dims_from_one_img_1(img):
    cv2.imwrite("with-bg_1.jpg", img)
    img = open("with-bg_1.jpg")
    response = requests.post(
    'https://api.remove.bg/v1.0/removebg',
    files={'image_file': img},
    data={'size': 'auto'},
    headers={'X-Api-Key': 'cpXjLbYJAmPMStKwAcfjrfZ5'},)
    if response.status_code == requests.codes.ok:
        with open('no-bg_1.png', 'wb') as out:
            out.write(response.content)
            # img = response.content
    else:
        print("Error:", response.status_code, response.text)
    scale = 0.25
    # img.save("success.png")
    img = cv2.imread("no-bg_1.png")
    height_in_pixels = 0.5*(euclidean((640, 460), (480, 460)) + euclidean((80, 210), (230, 550)))*scale
    width_in_pixels = (euclidean((640, 460), (0, 460)))*scale
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
    horizontal_field = 64 #cm
    vretical_field = 45 #cm
    cm_per_pixel_horizontal = horizontal_field/width_in_pixels
    cm_per_pixel_vertical = vretical_field/height_in_pixels
    pixel_per_cm = 0.5*(cm_per_pixel_horizontal + cm_per_pixel_vertical)
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
    dim_angle = []
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
        dims.append(euclidean(two, one))
        # print(f"euclidean(two, one): {euclidean(two, one)}")
        dim_angle.append((abs(two[1] - one[1])/abs(two[0] - one[0])))
        # print(f"angle: {math.degrees(math.atan(abs(two[1] - one[1])/abs(two[0] - one[0])))}")
        c = np.copy(copy);
        cv2.circle(c, tup(two), 4, (255, 0, 0), -1);
        cv2.circle(c, tup(one), 4, (0,255,0), -1);
        cv2.line(c, tup(two), tup(one), (0,0,200), 1);
        cv2.imshow("Mark", c);
        cv2.waitKey(0);
    vol_weight = 1
    dist_in_cm = []
    # print(len(dims))
    # print(cm_per_pixel_horizontal)
    # print(cm_per_pixel_vertical)
    for t in range(6):
        dist_in_cm.append(dims[t]*((cm_per_pixel_horizontal*math.cos(abs(math.atan(abs(dim_angle[t]) - math.radians(15))))) + (cm_per_pixel_vertical*math.sin(math.atan(abs(dim_angle[t]) - math.radians(15))))))
    for t in dist_in_cm:
        vol_weight *= t
    vol_weight = math.sqrt(vol_weight)
    avg_dims = []
    for t in range(3):
        avg_dims.append(0.5*(dist_in_cm[t] + dist_in_cm[t + 3]))
    # print(dims)
    # print(dist_in_cm)
    # print(f"vol_weight: {vol_weight}")
    return jsonify({
        'volumetricWeight': vol_weight,
        'length': avg_dims[0],
        'breadth': avg_dims[1],
        'height': avg_dims[2],
        'sku': 1,
        'num_dims_seen': len(dist_in_cm)
    })


## for cam without the whitener on the silver tripod
def get_dims_from_one_img_2(img):
    cv2.imwrite("with-bg_2.jpg", img)
    img = open("with-bg_2.jpg")
    response = requests.post(
    'https://api.remove.bg/v1.0/removebg',
    files={'image_file': img},
    data={'size': 'auto'},
    headers={'X-Api-Key': 'cpXjLbYJAmPMStKwAcfjrfZ5'},)
    if response.status_code == requests.codes.ok:
        with open('no-bg_2.png', 'wb') as out:
            out.write(response.content)
            # img = response.content
    else:
        print("Error:", response.status_code, response.text)
    scale = 0.25
    # img.save("success.png")
    img = cv2.imread("no-bg_2.png")
    height_in_pixels = 0.5*(euclidean((80, 480), (50, 260)) + euclidean((640, 370), (450, 210)))*scale
    width_in_pixels = (euclidean((70, 480), (630, 370)))*scale
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
    horizontal_field = 64 #cm
    vretical_field = 45 #cm
    cm_per_pixel_horizontal = horizontal_field/width_in_pixels
    cm_per_pixel_vertical = vretical_field/height_in_pixels
    pixel_per_cm = 0.5*(cm_per_pixel_horizontal + cm_per_pixel_vertical)
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
    dim_angle = []
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
        dims.append(euclidean(two, one))
        # print(f"euclidean(two, one): {euclidean(two, one)}")
        dim_angle.append((abs(two[1] - one[1])/abs(two[0] - one[0])))
        # print(f"angle: {math.degrees(math.atan(abs(two[1] - one[1])/abs(two[0] - one[0])))}")
        c = np.copy(copy);
        cv2.circle(c, tup(two), 4, (255, 0, 0), -1);
        cv2.circle(c, tup(one), 4, (0,255,0), -1);
        cv2.line(c, tup(two), tup(one), (0,0,200), 1);
        # cv2.imshow("Mark", c);
        # cv2.waitKey(0);
    vol_weight = 1
    dist_in_cm = []
    # print(len(dims))
    # print(cm_per_pixel_horizontal)
    # print(cm_per_pixel_vertical)
    for t in range(6):
        dist_in_cm.append(dims[t]*((cm_per_pixel_horizontal*math.cos(abs(math.atan(abs(dim_angle[t]) - math.radians(15))))) + (cm_per_pixel_vertical*math.sin(math.atan(abs(dim_angle[t]) - math.radians(15))))))
    for t in dist_in_cm:
        vol_weight *= t
    vol_weight = math.sqrt(vol_weight)
    avg_dims = []
    for t in range(3):
        avg_dims.append(0.5*(dist_in_cm[t] + dist_in_cm[t + 3]))
    # print(dims)
    # print(dist_in_cm)
    # print(f"vol_weight: {vol_weight}")
    return jsonify({
        'volumetricWeight': vol_weight,
        'length': avg_dims[0],
        'breadth': avg_dims[1],
        'height': avg_dims[2],
        'sku': 1,
        'num_dims_seen': len(dist_in_cm)
    })


def integrate_results(json_1, json_2):
    if abs((json_1['volumetricWeight'] - json_2['volumetricWeight'])/(json_1['volumetricWeight'] + json_2['volumetricWeight'])) > 2.8:
        if json_1['num_dims_seen'] == 6 and json_2['num_dims_seen'] == 6:
            return jsonify({
                'volumetricWeight': 0.5*(json_1['volumetricWeight'] + json_2['volumetricWeight']),
                'length': 0.5*(json_1['length'] + json_2['length']),
                'breadth': 0.5*(json_1['breadth'] + json_2['breadth']),
                'height': 0.5*(json_1['height'] + json_2['height']),
                'sku': 1
            })
        if json_1['num_dims_seen'] == 6 and json_2['num_dims_seen'] != 6:
            return jsonify({
                'volumetricWeight': json_1['volumetricWeight'],
                'length': json_1['length'],
                'breadth': json_1['breadth'],
                'height': json_1['height'],
                'sku': 1
            })
        if json_1['num_dims_seen'] != 6 and json_2['num_dims_seen'] == 6:
            return jsonify({
                'volumetricWeight': json_2['volumetricWeight'],
                'length': json_2['length'],
                'breadth': json_2['breadth'],
                'height': json_2['height'],
                'sku': 1
            })
        else:
            return jsonify({
                'volumetricWeight': 0.5*(json_1['volumetricWeight'] + json_2['volumetricWeight']),
                'length': 0.5*(json_1['length'] + json_2['length']),
                'breadth': 0.5*(json_1['breadth'] + json_2['breadth']),
                'height': 0.5*(json_1['height'] + json_2['height']),
                'sku': 1
            })
    else:
        return jsonify({
                'volumetricWeight': 0.5*(json_1['volumetricWeight'] + json_2['volumetricWeight']),
                'length': 0.5*(json_1['length'] + json_2['length']),
                'breadth': 0.5*(json_1['breadth'] + json_2['breadth']),
                'height': 0.5*(json_1['height'] + json_2['height']),
                'sku': 1
            })   
