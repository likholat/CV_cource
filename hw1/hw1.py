import cv2 as cv
import numpy as np


def detect_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)

    if len(rects) == 0:
        return []
    return rects[0]


def draw_rect(img, rect, color=(0, 255, 0)):
    draw_rect_img = img.copy()
    x, y, h, w = rect
    cv.rectangle(draw_rect_img, (x, y), (x+w, y+h), color, 2)
    return draw_rect_img


def crop_img_10pc(img, rect):
    x, y, h, w = rect
    img_shape = img.shape

    percentage_h = int(h / 10)
    percentage_w = int(w / 10)

    x1 = x - percentage_w
    x1 = 0 if x1 < 0 else x1

    x2 = x + w + percentage_w
    x2 = img_shape[1] - 1 if x2 >= img_shape[1] else x2

    y1 = y - percentage_h
    y1 = 0 if y1 < 0 else y1

    y2 = y + h + percentage_h
    y2 = img_shape[0] - 1 if y2 >= img_shape[0] else y2

    return img[y1:y2, x1:x2]


def contours(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
    equal_hist = cv.equalizeHist(blur)
    canny = cv.Canny(equal_hist, 100, 150, 3)
    return canny


def main():
    img = cv.imread('data/lenna.jpg')

    face_rect = detect_face(img)
    rect_img = draw_rect(img, face_rect)
    cv.imwrite('res/face_detection.png', rect_img)

    img = crop_img_10pc(img, face_rect)
    cv.imwrite('res/crop_image.png', img)
    shape = img.shape

    contour = contours(img)
    cv.imwrite('res/contours.png', contour)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv.dilate(contour, kernel)
    cv.imwrite('res/dilation.png', dilation)

    gaussian_blur = cv.GaussianBlur(dilation, (5, 5), cv.BORDER_DEFAULT)
    cv.imwrite('res/gaussian_blur.png', gaussian_blur)
    norm_image = cv.normalize(gaussian_blur, None, 0, 1.0, cv.NORM_MINMAX)

    bilateral = cv.bilateralFilter(img, 15, 75, 75)
    cv.imwrite('res/bilateral.png', bilateral)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_img = cv.filter2D(img, -1, kernel)
    cv.imwrite('res/sharp_img.png', sharp_img)

    res = np.zeros(shape)
    for i in range(shape[2]):
        res[:, :, i] = norm_image * sharp_img[:, :, i] + \
            (1-norm_image)*bilateral[:, :, i]

    cv.imwrite('res/res.png', res)


if __name__ == "__main__":
    main()
