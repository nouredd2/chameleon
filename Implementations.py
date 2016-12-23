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
    """2D version of DCT (DCT type 2)"""
    transform = dct(dct(m.T, norm='ortho').T, norm='ortho')
    transform[0, 0] -= DC_SHIFT
    return transform


def idct2d(m0):
    """2D version of IDCT (DCT type 3)"""
    m = np.copy(m0)
    m[0, 0] += DC_SHIFT
    return idct(idct(m, norm='ortho').T, norm='ortho').T


def ycbcr(rgb):
    """Formula to convert from RGB to Y'CbCr"""
    red, green, blue = rgb
    y = int(min(max(0, round(0.299*red + 0.587*green + 0.114*blue)), 255))
    cb = int(min(max(0, round((-0.299*red - 0.587*green + 0.886*blue)/1.772 + 128)), 255))
    cr = int(min(max(0, round((0.701*red - 0.587*green - 0.114*blue)/1.402 + 128)), 255))
    return y, cb, cr


def rgb(ycc):
    """Formula to convert from Y'CbCr to RGB"""
    y, cb, cr = ycc
    red = int(min(max(0, round(y + 1.402*(cr-128))), 255))
    green = int(min(max(0, round(y-(0.114*1.772*(cb-128)+0.299*1.402*(cr-128))/0.587)), 255))
    blue = int(min(max(0, round(y+1.772*(cb-128))), 255))
    return red, green, blue


class BaseStego:
    """A base class for stego implementations"""

    @staticmethod
    def addpadding(img):
        """Adds padding for DCT transform"""
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
        return img

    @staticmethod
    def convertdct(spatialrep):
        """Transforms an image to DCT"""
        # DCT representation axes order is color plane, xcoord of block, ycoord of block, block xcoord, block ycoord
        dimens = (spatialrep.shape[2], spatialrep.shape[1] // BLOCK_SIZE,
                  spatialrep.shape[0] // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        # Transform each block
        dctrep = np.empty(dimens)
        for colorplane in range(spatialrep.shape[2]):
            for xcoord in range(0, spatialrep.shape[1], BLOCK_SIZE):
                for ycoord in range(0, spatialrep.shape[0], BLOCK_SIZE):
                    spatialblock = spatialrep[ycoord:ycoord + BLOCK_SIZE, xcoord:xcoord + BLOCK_SIZE,
                                   colorplane]
                    dctblock = dct2d(spatialblock)
                    dctrep[colorplane, xcoord // BLOCK_SIZE, ycoord // BLOCK_SIZE] = dctblock
        return dctrep

    @staticmethod
    def convertspatial(dctrep):
        """Transforms DCT representation to image"""
        dimens = (dctrep.shape[2] * BLOCK_SIZE, dctrep.shape[1] * BLOCK_SIZE, dctrep.shape[0])
        spatialrep = np.empty(dimens, dtype=np.uint8)
        for xcoord in range(dctrep.shape[1]):
            for ycoord in range(dctrep.shape[2]):
                for colorplane in range(dctrep.shape[0]):
                    dctblock = dctrep[colorplane, xcoord, ycoord]
                    spatialblock = idct2d(dctblock)
                    spatialrep[ycoord * BLOCK_SIZE:(ycoord + 1) * BLOCK_SIZE,
                    xcoord * BLOCK_SIZE:(xcoord + 1) * BLOCK_SIZE, colorplane] = spatialblock
        return spatialrep

    @staticmethod
    def convertycc(bgrrep):
        """Converts RGB (BGR order) to Y'CbCr"""
        yccrep = np.empty(bgrrep.shape, dtype=np.uint8)
        for y in range(bgrrep.shape[0]):
            for x in range(bgrrep.shape[1]):
                blue, green, red = bgrrep[y, x]
                yccrep[y, x, 0], yccrep[y, x, 1], yccrep[y, x, 2] = ycbcr((red, green, blue))
        return yccrep

    @staticmethod
    def convertbgr(yccrep):
        """Converts Y'CbCr to RGB (BGR order)"""
        bgrrep = np.empty(yccrep.shape, dtype=np.uint8)
        for y in range(yccrep.shape[0]):
            for x in range(yccrep.shape[1]):
                yprime, cb, cr = yccrep[y, x]
                bgrrep[y, x, 2], bgrrep[y, x, 1], bgrrep[y, x, 0] = rgb((yprime, cb, cr))
        return bgrrep

    def __init__(self):
        """Initializes instance variables"""
        self.dctrep = None
        self.quantized = False

    def setcover(self, path):
        """Stores a cover image in DCT format"""
        img = cv2.imread(path)
        img = BaseStego.addpadding(img)
        img = BaseStego.convertycc(img)
        self.dctrep = BaseStego.convertdct(img)

    def save(self, path):
        """Saves the current image to a file"""
        if self.dctrep is None:
            print('save(): Please set cover image first')
            return
        if self.quantized:
            self.dequantize()
        spatialrep = BaseStego.convertspatial(self.dctrep)
        spatialrep = BaseStego.convertbgr(spatialrep)
        cv2.imwrite(path, spatialrep)

    def quantize(self):
        """Quantizes the current DCT representation"""
        if self.dctrep is None:
            print('quantize(): Please set cover image first')
            return
        quantizedrep = np.empty(self.dctrep.shape, dtype=int)
        for colorplane in range(self.dctrep.shape[0]):
            for xcoord in range(self.dctrep.shape[1]):
                for ycoord in range(self.dctrep.shape[2]):
                    block = self.dctrep[colorplane, xcoord, ycoord]
                    newblock = quantizedrep[colorplane, xcoord, ycoord]
                    for u in range(block.shape[0]):
                        for v in range(block.shape[1]):
                            # QT is in row-major order
                            newblock[u, v] = round(block[u, v] / QT[v, u])
        self.dctrep = quantizedrep
        self.quantized = True

    def dequantize(self):
        """Multiples by QT to approximate origianl"""
        if self.dctrep is None:
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
test.setcover('hacker.jpg')
test.quantize()
test.save('test.jpg')