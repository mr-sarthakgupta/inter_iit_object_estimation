import base64
from flask import Flask, request, json
from flask_cors import CORS
import torch
import torchvision
from torchvision import models
from torchvision.transforms import ToTensor
import cv2
import numpy as np

platform_width = 27
platform_height = 14

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

def calculate_scale_factor(platform_mask):
    # cv2.imshow("platform", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Find the contours of the platform
    # print(platform_mask)
    contours, _ = cv2.findContours(platform_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    platform_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(platform_contour)

    aspect_ratio = w / h

    if aspect_ratio > 1:
        platform_width_pixels = w
    else:
        platform_width_pixels = h

    scale_factor = platform_width / platform_width_pixels
    # print(platform_width_pixels)
    return scale_factor

def segment_object(img):

    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()


    tensor_img = ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor_img)
    
    masks = outputs[0]['masks'].squeeze().cpu().numpy()
    platform_mask = np.zeros_like(masks[0]) 
    object_mask = np.zeros_like(masks[0])

    p, o = 0, 0
    y_size = masks.shape[1]

    threshold = 0.5

    for i in range(masks.shape[0]):
        obj_mask = (255*(masks[i] > threshold)).astype(np.uint8)

        y_idx = (-1)*(y_size//4)
        print(obj_mask[y_idx][masks.shape[1]//2])
        if obj_mask[y_idx][masks.shape[1]//2] == 255:
            platform_mask += obj_mask
            p += 1
        else:
            object_mask += obj_mask
            o += 1

    platform_mask = platform_mask//p
    object_mask = object_mask//o


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # cv2.imshow("Kernel Mask", obj_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    platform_mask = cv2.morphologyEx(platform_mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Final Platform Mask", platform_mask)
    # cv2.waitKey()
    # cv2.imshow("Final Object Mask", object_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return platform_mask.astype(np.uint8), object_mask.astype(np.uint8)


def measure_object_volume(scale_factor, object_mask):
    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    L_o = np.linalg.norm(box[0] - box[1])*scale_factor
    W_o = np.linalg.norm(box[1] - box[2])*scale_factor
    H_o = np.linalg.norm(box[0] - box[3])*scale_factor
    dimensions_pixels = [L_o, W_o, H_o]
    print(dimensions_pixels)

    dimensions_cm = dimensions_pixels
    
    volume = dimensions_cm[0]*dimensions_cm[1]*dimensions_cm[2]
    
    return {"volumetricWeight": volume, "length":L_o, "breadth":W_o, "height":H_o, "sku": "123456"}

def decod(response):
    decoded = base64.b64decode(response["images"][0].encode()[1:])
    decode = open('decoded.png', 'wb')
    decode.write(decoded)
    decode.close()

app = Flask(__name__)
CORS(app)

@app.route("/scanner", methods=['POST'])
def scanner():
    with open('sample.json', 'w') as outfile:
        json.dump(json.load(request.files['file']), outfile)
    uploaded_file = open('sample.json')
    uploaded_file = json.load(uploaded_file)
    x1 = base64.b64decode(uploaded_file['images'][0][1:])
    img_1 = open('object.png', 'wb')
    img_1.write(x1)
    object_img = cv2.imread("object.png")
    platform_mask, object_mask = segment_object(object_img)
    return measure_object_volume(calculate_scale_factor(platform_mask), object_mask)

if __name__ == "__main__":
    app.run(debug=True)

