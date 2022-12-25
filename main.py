from PIL import Image
import numpy as np
from numpy.linalg import linalg

def getImageMatrixes(image):
    pix = np.array(image)
    R = np.zeros((pix.shape[0], pix.shape[1]))
    G = np.zeros((pix.shape[0], pix.shape[1]))
    B = np.zeros((pix.shape[0], pix.shape[1]))

    R = pix[:, :, 0]
    G = pix[:, :, 1]
    B = pix[:, :, 2]

    return R, G, B

def decomposeSVD(k, A):
    u, s, vt = linalg.svd(A,  full_matrices=False)
    newEpsilon = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(0, k):
        newEpsilon[i][i] = s[i]
    newMat = np.matmul(u, (np.matmul(newEpsilon, vt)))
    return newMat.astype(int)

def createNewImage(image, R, G, B, k):
    pix = np.array(image)
    newImg = np.zeros((pix.shape[0],pix.shape[1], 3))
    newImg[:, :, 0] = R
    newImg[:, :, 1] = G
    newImg[:, :, 2] = B

    new_Img = Image.fromarray(newImg.astype('uint8'))
    new_Img.save(str(k) + "Fox_2.jpg")
    return newImg

def error(sig, k):
    return sum(sig[k+1:] ** 2) / sum(sig**2)

if __name__ == '__main__':
  image = Image.open("fox_2.jpg")

  R, G, B = getImageMatrixes(image)
  ks = [5, 50, 250, 300, 350, 400, 600]

  for k in ks:
        nR = decomposeSVD(k, R)
        nG = decomposeSVD(k, G)
        nB = decomposeSVD(k, B)
        newIm = createNewImage(image, nR, nG, nB, k)

        print(k)
        print( + error(np.linalg.svd(R)[1], k))



