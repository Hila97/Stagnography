from PIL import Image
import numpy as np
from binaryString import *
from functions import *

# from methodmus import *
import matplotlib.pyplot as plt


##########################################################################################################
# decode
matrix = converImgToMatrix('after.png')
lengthMatrix = len(matrix)
length = len(matrix)

lengthMsg = extraxtFirstBlock(matrix)
lengthOfSecret = bin2dec(lengthMsg)
print("length after decode")
print(lengthOfSecret)
position = 0
message = ''
for i in range(0, length - 1, 3):
    for j in range(0, length - 1, 3):
        if i == 0 and j == 0:
            print("first block")  # skip the first block
        else:
            # message += startBlockExtraction(matrix, i, j, lenMsgDec, len(message))#the length is the orignelm cant use it
            message += startBlockExtraction(matrix, i, j, lengthOfSecret, len(message))

print("message")
# print(message)
# message = '0' + message
# print(msgBin)
# x = message[0:lenMsgDec]
x = message[0:lengthOfSecret]
print(x)

print(bits2a(x))

