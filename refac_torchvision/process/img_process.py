import cv2
import numpy as np

def sharpen_image(image, kernel_size=(5, 5)):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def basic_blur(image, kernel_size=(10, 10)):
    return cv2.blur(image, kernel_size)

def gaussian_blur(image, kernel_size=(9, 9)):
    return cv2.GaussianBlur(image, kernel_size, sigmaX=0, sigmaY=0)

def adjust_brightness(image, value):
    return cv2.add(image, np.ones_like(image, dtype=np.uint8) * value)

def adjusted_image(image, alpha):
    mean_brightness = np.mean(image)
    beta = -mean_brightness * alpha
    return cv2.convertScaleAbs(image, alpha=1 + alpha, beta=beta)

def histogram_equalization_color(image):
    b, g, r = cv2.split(image)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])

def bilateral_filter(image, d=9, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return cv2.addWeighted(image, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.4, 0)

def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def morphological_transformations(image):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    return erosion, dilation

def filter_small_bboxes(image, bboxes, labels, min_area_ratio=0.01):
    filtered_bboxes, filtered_labels = [], []
    image_area = image.shape[0] * image.shape[1]

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area / image_area >= min_area_ratio:
            filtered_bboxes.append(bbox)
            filtered_labels.append(label)
    return np.array(filtered_bboxes), np.array(filtered_labels)

def process_img(image):
    processing_img = image.copy()
    image_255 = (processing_img * 255).astype(np.uint8)
    erosion,dilation = morphological_transformations(image_255)
    result = erosion
    result = gamma_correction(result)
    return result / 255.0
