from PIL import Image
import numpy as np
from numpy.linalg import linalg
import q6
import q7

def runQ6():
    image = Image.open("fox_2.jpg")
    R, G, B = q6.getImageMatrixes(image)
    ks = [5, 50, 100, 250, 300, 350, 400, 550]

    for k in ks:
        nR = q6.decomposeSVD(k, R)
        nG = q6.decomposeSVD(k, G)
        nB = q6.decomposeSVD(k, B)
        newIm = q6.createNewImage(image, nR, nG, nB, k)


def runQ7():
    s_list = [5, 10, 50, 200, 500, 1024]
    k_list = [1, 5, 10, 15, 50, 200]
    train_d, train_l = q7.train_data()
    test_d, test_l = q7.test_data()

    predictions = dict()
    U, e, v = np.linalg.svd(train_d, full_matrices=False)

    for s in s_list:
        print(s)
        Us = U[:, :s]
        train_proj = np.matmul(Us.T, train_d)
        test_proj = np.matmul(Us.T, test_d)
        inner_products = np.matmul(train_proj.T, test_proj)
        normsTrain = np.square(np.linalg.norm(train_proj, axis=0))

        for i in range(test_proj.shape[1]):
            distances = normsTrain - 2 * inner_products[:, i]
            prediction = q7.find_k_neighbors(distances, train_l, k_list)
            for k in prediction:
                predictions[s, k, i] = prediction[k]

    errors = q7.find_errors(predictions, k_list, s_list, test_l)
    print(errors)


if __name__ == '__main__':
    runQ7()






