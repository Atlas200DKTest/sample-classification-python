import numpy as np
import copy
#from atlasutil.presenteragent.presenter_types import *
import cv2 as cv

'''
def SSDPostProcess(inference_result, image_resolution, confidence_threshold):
    result = inference_result[0]
    shape = result.shape
    detection_result_list = []
    for i in range(0, shape[0]):
        item = result[i, 0, 0, ]
        if item[2] < confidence_threshold:
            continue
        detection_item = ObjectDetectionResult()
        detection_item.attr = int(item[1])
        detection_item.confidence = item[2]
        detection_item.lt.x = int(item[3] * image_resolution[0])
        detection_item.lt.y = int(item[4] * image_resolution[1])
        detection_item.rb.x = int(item[5] * image_resolution[0])
        detection_item.rb.y = int(item[6] * image_resolution[1])
        detection_item.result_text = str(detection_item.attr) + " " + str(detection_item.confidence*100) + "%"
        detection_result_list.append(detection_item)

    return detection_result_list 
'''

def SSDPostProcess(resultList, resolution, confidence_threshold):
    '''
    ssd postprocess, output the box cooridinates with required confidence
    the output of ssd model is [image_id, label, confidence, xmin, ymin, xmax, ymax], with shape [n,1,1,7]

    Args:
        resultList: the output of inference
        resolution: the resolution of image, (width, height)

    Returns:
        bbox: [condidence, lt_x, lt_y, rb_x, rb_y], with shape (n,5)
    '''
    result_tensor = resultList[0]
    bbox = []
    for arr in result_tensor:
        if arr[0][0][2] >= confidence_threshold:
            lt_x = arr[0][0][3] * resolution[0]
            lt_y = arr[0][0][4] * resolution[1]
            rb_x = arr[0][0][5] * resolution[0]
            rb_y = arr[0][0][6] * resolution[1]
            bbox.append([arr[0][0][2], lt_x, lt_y, rb_x, rb_y])
    if bbox == []:
        print("[SSDPostProcess]: no object detected")
    return bbox

label = ["background", "person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

anchors_yolo = [[(116,90),(156,198),(373,326)],[(30,61),(62,45),(59,119)],[(10,13),(16,30),(33,23)]]

def sigmoid(x):
    s = 1 / (1 + np.exp(-1*x))
    return s

def getMaxClassScore(class_scores):
    class_score = 0
    class_index = 0
    for i in range(len(class_scores)):
        if class_scores[i] > class_score:
            class_index = i+1
            class_score = class_scores[i]
    return class_score,class_index

def getBBox(feat, anchors, image_shape, confidence_threshold):
    box = []
    for i in range(len(anchors)):
        for cx in range(feat.shape[0]):
            for cy in range(feat.shape[1]):
                tx = feat[cx][cy][0 + 85 * i]
                ty = feat[cx][cy][1 + 85 * i]
                tw = feat[cx][cy][2 + 85 * i]
                th = feat[cx][cy][3 + 85 * i]
                cf = feat[cx][cy][4 + 85 * i]
                cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]

                bx = (sigmoid(tx) + cx)/feat.shape[0]
                by = (sigmoid(ty) + cy)/feat.shape[1]
                bw = anchors[i][0]*np.exp(tw)/image_shape[0]
                bh = anchors[i][1]*np.exp(th)/image_shape[1]

                b_confidence = sigmoid(cf)
                b_class_prob = sigmoid(cp)
                b_scores = b_confidence*b_class_prob
                b_class_score,b_class_index = getMaxClassScore(b_scores)

                if b_class_score > confidence_threshold:
                    box.append([bx,by,bw,bh,b_class_score,b_class_index])
    return box


def donms(boxes,nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2]
    b_h = boxes[:, 3]
    scores = boxes[:,4]
    areas = (b_w+1)*(b_h+1)
    order = scores.argsort()[::-1]
    keep = [] 
    while order.size > 0:
        i = order[0]
        keep.append(i) 
        xx1 = np.maximum(b_x[i], b_x[order[1:]])
        yy1 = np.maximum(b_y[i], b_y[order[1:]])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        IoU = inter / union
        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1] 

    final_boxes = [boxes[i] for i in keep]
    return final_boxes

def getBoxes(resultList, anchors, img_shape, confidence_threshold, nms_threshold):
    boxes = []
    for i in range(resultList):
        feature_map = resultList[i][0].transpose((2, 1, 0))
        box = getBBox(feature_map, anchors[i], img_shape, confidence_threshold)
        boxes.extend(box)
    Boxes = donms(np.array(boxes),nms_threshold)
    return Boxes

def Yolov3_post_process(resultList, img, confidence_threshold, nms_threshold):
    '''
    processes YOLOv3 inference result, and returns boxes detected

    Args:
        resultList: a list, inference result
        img: numpy array, image data
        confidence_threshold: float number, confidence threshold
        nms_threshold: float number, NMS threshold
    '''
    resultArray = resultList[0]
    img_shape = img.shape
    boxes = getBoxes(resultArray, anchors_yolo, img_shape, confidence_threshold, nms_threshold)
    return boxes

def GenerateTopNClassifyResult(resultList, n):
    '''
    processes classification result, returns top n categories

    Args:
        resultList: list, classification result
        n: integer, the quantity of categories with top confidece level user wants to obtain
    
    Returns:
        topNArray: numpy array, top n confidence
        confidenceIndex: numpy array, the corresponding index of top n confidence
    '''
    resultArray = resultList[0]
    confidenceList = resultArray[0, 0, 0, :]
    confidenceArray = np.array(confidenceList)
    confidenceIndex = np.argsort(-confidenceArray)
    topNArray = np.take(confidenceArray, confidenceIndex[0:n])
    return topNArray, confidenceIndex[0:n]

def FasterRCNNPostProcess(resultList, confidence_threshold):
    '''
    processes Faster RCNN inference result, returns a list of box coordinates

    Args:
        resultList: list, inference result
        confidence_threshold: float number, confidence threshold
    
    Returns:
        result_bbox: list, box coordinates
    '''
    tensor_num = resultList[0].reshape(-1)
    tensor_bbox = resultList.reshape(64, 304, 8)
    result_bbox = []
    for num in tensor_num:
        for bbox_idx in range(num):
            class_idx = attr * 2
            lt_x = tensor_bbox[class_idx][bbox_idx][0]
            lt_y = tensor_bbox[class_idx][bbox_idx][1]
            rb_x = tensor_bbox[class_idx][bbox_idx][2]
            rb_y = tensor_bbox[class_idx][bbox_idx][3]
            score = tensor_bbox[class_idx][bbox_idx][4]
            if score >= confidence_threshold:
                result_bbox.append([lt_x, lt_y, rb_x, rb_y, attr, score])
    return result_bbox