from PIL import Image
import numpy as np
from binaryString import *
import matplotlib.pyplot as plt
import math

def converImgToMatrix(route):
    img = Image.open(route).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    return data


def printMatrix(data):
    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].
    for row in data:
        print(' '.join('{:3}'.format(value) for value in row))


def dec2bin(x):  # from decimal number
    return int(bin(x)[2:])


def bin2dec(x):  # from binary number (not string)
    binary = str(x)
    return int(binary, 2)


def stringToImage(str, size):
    x = []
    for i in range(0, size, 8):
        print("i", i)
        s = str[i:i + 8]
        print(s)
        s = bin2dec(s)
        print(s)
        x.append(s)
    return x


def decMatToBinArr(matrix):
    for x in matrix:
        for y in x:
            mat.append(str(format(y, '08b')))


matrix = converImgToMatrix('lena.png')
print(matrix)
mat = []
decMatToBinArr(matrix)
print(mat)
res = "".join(mat)
print(res)
print("the length", len(res))
size = len(res)
ans = stringToImage(res, size)
print(ans)
arr = np.array(ans)
print("arr",arr)
s=int(math.sqrt(size/8))
print("s",s)
matrix = arr.reshape(s, s)
printMatrix(matrix)
array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('try2.png')
