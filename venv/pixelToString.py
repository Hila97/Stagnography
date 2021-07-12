from PIL import Image
import numpy as np
from binaryString import *
import matplotlib.pyplot as plt
import pandas as pd

def converImgToMatrix(route):
    img = Image.open(route).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    return data
def decMatToBinArr(matrix):
    for x in matrix:
        for y in x:
            mat.append(str(format(y, '08b')))

matrix = converImgToMatrix('eggs.png')
#matrix=[[1,1,1],[1,1,1],[1,1,1]]
print(matrix)
mat=[]
decMatToBinArr(matrix)
print(mat)
res = "".join(mat)
print(res)
