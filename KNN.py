import numpy as np
from mnist import MNIST
from scipy.spatial.distance import cdist
from scipy.stats import mode

PCA_Dimension = 50
KNN_K = 10


def pca(raw_img, sub_img):
    global PCA_Dimension
    img = np.array(raw_img).T
    zero_img = np.apply_along_axis(lambda x: x - np.mean(x), axis=1, arr=img)
    cov_img = np.cov(zero_img)
    w, v = np.linalg.eig(cov_img)
    sort_list = np.argsort(w)
    pick_list = [0] * PCA_Dimension
    for i, j in enumerate(sort_list):
        if j >= 784 - PCA_Dimension:
            pick_list[783 - j] = i
    p = np.take(v, pick_list, axis=1).reshape(784, PCA_Dimension).T
    y = np.matmul(p, img).T
    suby = np.matmul(p, np.array(sub_img).T).T
    return y, suby


if __name__ == '__main__':
    mn = MNIST('MNIST-data')
    a_img, a_labels = mn.load_training()
    b_img, b_labels = mn.load_testing()
    a_img, b_img = pca(a_img, b_img)
    # dis = np.apply_along_axis(lambda x: [np.linalg.norm(x - i) for i in a_img], axis=1, arr=b_img)
    # print(dis.shape)
    dists = cdist(a_img, b_img)
    idx = np.argpartition(dists, PCA_Dimension, axis=0)[:PCA_Dimension]
    nearest_dist = np.take(a_labels, idx)
    out = mode(nearest_dist, axis=0)[0][0]
    accuracy = np.sum(np.array(b_labels) == np.array(out)) / len(b_labels)
    print('accuracy: %.3f' % accuracy)
