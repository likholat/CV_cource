import cv2 as cv
import numpy as np


def corners(image):
    corners = cv.goodFeaturesToTrack(image, 200, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(image, (x, y), 2, 255, 2)

    return image


def average(image, integral, distances, k):
    h, w = image.shape
    result = image.copy()

    for i in range(h):
        for j in range(w):
            radius = int(k * distances[i, j] / 2)
            if radius != 0:
                x1 = i - radius
                x1 = 0 if x1 < 0 else x1

                x2 = i + radius
                x2 = h - 1 if (h - 1) < x2 else x2

                y1 = j - radius
                y1 = 0 if y1 < 0 else y1

                y2 = j + radius
                y2 = w - 1 if (w - 1) < y2 else y2

                a = integral[x2, y2]
                b = integral[x2, y1]
                c = integral[x1, y2]
                d = integral[x1, y1]

                avg = (a - b - c + d) / ((x2 - x1) * (y2 - y1))
                result[i, j] = avg if avg > 0 else 1
    return result


def main():
    img = cv.imread('data/nature.jpg')

    yuv_img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    Y, U, V = cv.split(yuv_img)

    equal_hist = cv.equalizeHist(Y)
    canny = cv.Canny(equal_hist, 100, 150, 3)
    cv.imwrite('res/contours.png', canny)

    img_corners = corners(canny)
    cv.imwrite('res/corners.png', img_corners)

    img_invert = cv.bitwise_not(img_corners)
    dist = cv.distanceTransform(img_invert, cv.DIST_L2, 3)

    integral_img = cv.integral(Y)
    avg_img = average(Y, integral_img, dist, 2)

    merged = cv.merge([avg_img, U, V])
    res = cv.cvtColor(merged, cv.COLOR_YUV2BGR)
    cv.imwrite('res/res.png', res)


if __name__ == "__main__":
    main()
