import numpy as np
import cv2
from scipy.fftpack import dct, idct

BLOCK_SIZE = 8
QT = np.matrix('16 11 10 16 24 40 51 61\
              ; 12 12 14 19 26 58 60 55\
              ; 14 13 16 24 40 57 69 56\
              ; 14 17 22 29 51 87 80 62\
              ; 18 22 37 56 68 109 103 77\
              ; 24 35 55 64 81 104 113 92\
              ; 49 64 78 87 103 121 120 101\
              ; 72 92 95 98 112 100 103 99')
DC_SHIFT = 1024

def dct2d(m):
    transform = dct(dct(m.T, norm='ortho').T, norm='ortho')
    transform[0, 0] -= DC_SHIFT
    return transform


def idct2d(m0):
    m = np.copy(m0)
    m[0, 0] += DC_SHIFT
    return idct(idct(m, norm='ortho').T, norm='ortho').T


class BaseStego:
    """A base class for stego implementations"""

    def __init__(self):
        """Initializes instance variables"""
        self.dctrep = np.zeros((3, 1, 1, 8, 8))
        self.quantized = False

    def setcover(self, path):
        """Stores a cover image in DCT format"""
        img = cv2.imread(path)
        # Add rows so # of rows is divisible by 8
        if img.shape[0] % BLOCK_SIZE != 0:
            padamount = BLOCK_SIZE - img.shape[0] % BLOCK_SIZE
            lastrow = img[-1]
            padding = np.empty((padamount, lastrow.shape[0], lastrow.shape[1]), dtype=np.uint8)
            for i in range(padamount):
                padding[i] = np.copy(lastrow)
            img = np.append(img, padding, axis=0)

        # Add columns so # of columns is divisible by 8
        if img.shape[1] % BLOCK_SIZE != 0:
            padamount = BLOCK_SIZE - img.shape[1] % BLOCK_SIZE
            lastcolumn = img[:, -1]
            padding = np.empty((lastcolumn.shape[0], padamount, lastrow.shape[1]), dtype=np.uint8)
            for i in range(padamount):
                padding[:, i] = np.copy(lastcolumn)
            img = np.append(img, padding, axis=1)

        # DCT representation axes order is color plane, xcoord of block, ycoord of block, block xcoord, block ycoord
        dimens = (img.shape[2], img.shape[1]//BLOCK_SIZE, img.shape[0]//BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

        # Transform each block
        dctrep = np.empty(dimens)
        for colorplane in range(img.shape[2]):
            for xcoord in range(0, img.shape[1], BLOCK_SIZE):
                for ycoord in range(0, img.shape[0], BLOCK_SIZE):
                    spatialblock = img[ycoord:ycoord+BLOCK_SIZE, xcoord:xcoord+BLOCK_SIZE, colorplane]
                    dctblock = dct2d(spatialblock)
                    dctrep[colorplane, xcoord//BLOCK_SIZE, ycoord//BLOCK_SIZE] = dctblock
        self.dctrep = dctrep

    def save(self, path):
        """Saves the current image to a file"""
        if not hasattr(self, 'dctrep'):
            print('save(): Please set cover image first')
            return

        if self.quantized:
            self.dequantize()

        dimens = (self.dctrep.shape[2]*BLOCK_SIZE, self.dctrep.shape[1]*BLOCK_SIZE, self.dctrep.shape[0])
        spatialrep = np.empty(dimens, dtype=np.uint8)
        for xcoord in range(self.dctrep.shape[1]):
            for ycoord in range(self.dctrep.shape[2]):
                for colorplane in range(self.dctrep.shape[0]):
                    dctblock = self.dctrep[colorplane, xcoord, ycoord]
                    spatialblock = idct2d(dctblock)
                    spatialrep[ycoord*BLOCK_SIZE:(ycoord+1)*BLOCK_SIZE, xcoord*BLOCK_SIZE:(xcoord+1)*BLOCK_SIZE, colorplane] = spatialblock
        cv2.imwrite(path, spatialrep)

    def quantize(self):
        """Quantizes the current DCT representation"""
        if not hasattr(self, 'dctrep'):
            print('quantize(): Please set cover image first')
            return

        for colorplane in range(self.dctrep.shape[0]):
            for xcoord in range(self.dctrep.shape[1]):
                for ycoord in range(self.dctrep.shape[2]):
                    block = self.dctrep[colorplane, xcoord, ycoord]
                    for u in range(block.shape[0]):
                        for v in range(block.shape[1]):
                            # QT is in row-major order
                            block[u, v] = round(block[u, v] / QT[v, u])

        self.quantized = True

    def dequantize(self):
        """Multiples by QT to approximate origianl"""
        if not hasattr(self, 'dctrep'):
            print('dequantize(): Please set cover image first')
            return
        if not self.quantized:
            print('dequantize(): DCT is not currently quantized')
            return

        for colorplane in range(self.dctrep.shape[0]):
            for xcoord in range(self.dctrep.shape[1]):
                for ycoord in range(self.dctrep.shape[2]):
                    block = self.dctrep[colorplane, xcoord, ycoord]
                    for u in range(block.shape[0]):
                        for v in range(block.shape[1]):
                            # QT is in row-major order
                            block[u, v] *= QT[v, u]

test = BaseStego()
test.setcover('lena.jpg')
test.quantize()
test.save('test.jpg')

# testm = np.matrix('52 55 61 66 70 61 64 73\
#                  ; 63 59 55 90 109 85 69 72\
#                  ; 62 59 68 113 144 104 66 73\
#                  ; 63 58 71 122 154 106 70 69\
#                  ; 67 61 68 104 126 88 68 70\
#                  ; 79 65 60 70 77 68 58 75\
#                  ; 85 71 64 59 55 61 65 83\
#                  ; 87 79 69 68 65 76 78 94')
#
# print(testm)
# print(dct2d(testm))
# print(idct2d(dct2d(testm)))