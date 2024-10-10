import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from collections import Counter

def sharpenes_img(image, kernel_size=(5, 5)):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def basic_blur(image, kernel_size=(10,10)):
    blur_image = cv2.blur(image, kernel_size)
    return blur_image

def gaussian_blue(image,kernel_size=(9,9)):
    # sigma두개를 다 0으로 줘서 가운데 값을 기준으로 블러 처리
    gaussian_image = cv2.GaussianBlur(image, kernel_size,sigmaX=0,sigmaY=0)
    return gaussian_image

def brightness_img(image, value):
    result = cv2.add(image,np.ones_like(image, dtype=np.uint8) * value)
    return result

def abjusted_image(image, alpha):
    mean_brightness = np.mean(image)  # 0~255 기준 평균 밝기
    beta = -mean_brightness * alpha
    adjusted_image = cv2.convertScaleAbs(image, alpha=1+alpha, beta=beta)
    return adjusted_image

def histogram_equalization_color(image):
    b, g, r = cv2.split(image)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    equalized_image = cv2.merge([b_eq, g_eq, r_eq])
    return equalized_image

def bilateral_filter(image, d=9, sigmaColor=50, sigmaSpace=50): # test 2
    '''
    image (numpy array): 필터를 적용할 이미지
    d (int): 필터링에 사용될 이웃 픽셀 거리
    sigmaColor (float): 색상 공간 필터 강도
    sigmaSpace (float): 거리 공간 필터 강도
    '''
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def edge_detection(image): 
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    combined_image = cv2.addWeighted(image, 0.8, edges_colored, 0.4, 0)
    return combined_image

def gamma_correction(image, gamma=1.5): # test 4
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def morphological_transformations(image): # test 5
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)
    dilation = cv2.dilate(image, kernel, iterations = 1)
    return erosion, dilation

def filter_small_bboxes(image, bboxes, labels, min_area_ratio=0.01):
    filtered_bboxes = []
    filtered_labels = []
    removed_bboxes = []
    removed_labels = []
    image_area = image.shape[0] * image.shape[1]

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_ratio = bbox_area / image_area

        if bbox_ratio >= min_area_ratio:
            filtered_bboxes.append(bbox)
            filtered_labels.append(label)
        else:
            removed_bboxes.append(bbox)
            removed_labels.append(label)

    # print(f"FILTER SMALL BBOX DONE! Total: {len(bboxes)}, Filtered: {len(filtered_bboxes)}, Removed: {len(removed_bboxes)}")
    # for idx, bbox in enumerate(removed_bboxes):
    #     print(f"Removed BBox {idx+1}: Area Ratio = {((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / image_area:.5f}")
    return np.array(filtered_bboxes), np.array(filtered_labels)


def process_img(image):
    processing_img = image.copy()
    image_255 = (processing_img * 255).astype(np.uint8)
    # result = brightness_img(image_255, 50)
    erosion,dilation = morphological_transformations(image_255)
    result = erosion
    result = gamma_correction(result)
    return result / 255.0
