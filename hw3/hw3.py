import cv2
import numpy as np


def conv(input):
    W = input.shape[0]
    H = input.shape[1]
    M, R, S, C = 5, 3, 3, 3

    filters = np.random.uniform(size=(M, C, R, S))
    b = np.random.uniform(size=(M))
    res = np.zeros((W, H, M), dtype=np.float32)

    for x in range(W):
        for y in range(H):
            for m in range(M):
                res[x, y, m] = b[m]
                for i in range(R):
                    for j in range(S):
                        for c in range(C):
                            x_crop = x + i if x + i < W else W - 1
                            y_crop = y + j if y + j < H else H - 1
                            res[x, y, m] += input[x_crop, y_crop, c] * \
                                filters[m, c, i, j]
    return res


def norm(input):
    W, H, M = input.shape

    x = np.random.uniform(size=(M))
    y = np.random.uniform(size=(M))
    res = np.zeros((W, H, M), dtype=np.float32)

    for m in range(M):
        mean = np.mean(input[:, :, m])
        std = np.std(input[:, :, m])
        res[:, :, m] = x[m] * (input[:, :, m] - mean) / std + y[m]

    return res


def relu(input):
    return np.maximum(0, input)


def max_pool(input):
    W, H, M = input.shape

    n = 2
    W = W // n
    H = H // n
    res = np.zeros((W, H, M), dtype=np.float32)

    for m in range(M):
        for i in range(W):
            for j in range(H):
                res[i, j, m] = np.max(input[i*n: (i+1)*n, j*n: (j+1)*n, m])

    return res


def softmax(input):
    W, H, M = input.shape
    res = np.zeros((W, H, M), dtype=np.float32)

    for i in range(W):
        for j in range(H):
            exp = np.exp(input[i, j, :])
            res[i, j, :] = exp / exp.sum()

    return res


def main():
    input = cv2.imread('data/lenna.jpg')
    print(input.shape)

    conv_res = conv(input)
    print(conv_res.shape)
    norm_res = norm(conv_res)
    relu_res = relu(norm_res)
    pool_res = max_pool(relu_res)
    print(pool_res.shape)
    res = softmax(pool_res)
    print(res.shape)

    W, H, M = res.shape
    for m in range(M):
        n = np.zeros((W, H), dtype=np.float32)
        cv2.normalize(res[:, :, m], n, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'res/ch_{m}.png', n)


if __name__ == "__main__":
    main()
