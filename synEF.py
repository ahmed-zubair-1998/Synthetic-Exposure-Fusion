import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture


def synEF(img, rows, cols):
    yuv = cv.cvtColor((img), cv.COLOR_BGR2YUV)
    blur = cv.bilateralFilter(np.add(yuv[...,0], 0.000001),1,5,5)
    yuv[...,0] = np.divide(np.square(yuv[...,0]), blur)
    lum = yuv[...,0]
    X = lum.reshape((lum.size, 1))

    m= 3
    print("M", m)


    gmm = GaussianMixture(n_components=m, covariance_type='diag')
    gmm.fit(X)

    labels = gmm.predict(X)
    labels = np.reshape(labels, (rows, cols))

    g = []
    a = []
    sums = [0]*m
    count = [0]*m

    for i in range(lum.shape[0]):
        for j in range(lum.shape[1]):
            v = np.log(max(lum[i][j], 0.001))
            sums[labels[i][j]] += v
            count[labels[i][j]] += 1

    for i in range(m):
        v = sums[i]/(count[i] + 0.00001)
        g.append(np.exp(v))
        a.append(0.18/np.exp(v))

    exp = []

    for i in range(m):
        exp.append(lum * a[i])

    for i in range(m):
        lm = np.amax(exp[i])
        r = np.add(np.true_divide(exp[i], lm), 1)
        l = np.true_divide(exp[i], np.add(exp[i], 1))
        f = np.multiply(l, r)
        exp[i] = np.true_divide(f, np.add(yuv[...,0], 0.003))

    pme = []
    t_img = np.zeros((rows, cols,3))

    for i in range(m):
        for j in range(3):
            t_img[...,j] = np.multiply(exp[i], img[...,j])
        pme.append(t_img * 255)

        

    mergeMertens = cv.createMergeMertens()
    lo = mergeMertens.process(pme)
    
    cv.imshow('res', lo)
    cv.imshow('prev', img/255)
    cv.imwrite('imgs/sample-output.jpg', lo*255)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = cv.imread('imgs/sample.jpg', 1).astype('float32')
    synEF(img, img.shape[0], img.shape[1])


if __name__ == '__main__':
    main()


    
