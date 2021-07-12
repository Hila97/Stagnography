from PIL import Image
import numpy as np
from binaryString import *
from methodmus import *
import matplotlib.pyplot as plt





##########################################################################################################
# encode

matrix = converImgToMatrix('eggs.png')
# matrix = [[0, 0, 0, 255, 255, 255, 122, 122, 122], [122, 122, 122, 0, 0, 0, 122, 122, 122],
#           [200, 200, 200, 122, 122, 122, 122, 122, 122], [0, 0, 0, 255, 255, 255, 122, 122, 122],
#           [122, 122, 122, 0, 0, 0, 122, 122, 122], [200, 200, 200, 122, 122, 122, 122, 122, 122],
#           [0, 0, 0, 255, 255, 255, 122, 122, 122], [122, 122, 122, 0, 0, 0, 122, 122, 122],
#           [200, 200, 200, 122, 122, 122, 122, 122, 122]]
print("length of img")
length = len(matrix)
print(length)

if length % 3 != 0:
    print("padding +1 ")
    matrix = paddingOneByOne(matrix)
    length = len(matrix)

if length % 3 != 0:
    print("padding +2 ")
    matrix = paddingOneByOne(matrix)
    length = len(matrix)

# printMatrix(matrix)
array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('before.png')

msgBin, lenMsgBin = getMsg()

# printMatrix(matrix)
embeddFirstBlock(matrix, lenMsgBin)
# printMatrix(matrix)

lenMsgDec=bin2dec(lenMsgBin)
# print("length")
# print(lenMsgDec)

index = 0
for i in range(0, length - 1, 3):
    for j in range(0, length - 1, 3):
        if i == 0 and j == 0:
            print("first block")  # skip the first block
        elif index < len(msgBin):  # only if there is more to embedd
            index = startBlockEmbedding(matrix, i, j, index, msgBin)

# X = matrix
# plt.imshow(X, cmap="gray")
# plt.savefig('after.png')
array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('after.png')




##########################################################################################################
# decode
matrix = converImgToMatrix('after.png')
lengthMatrix = len(matrix)

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

