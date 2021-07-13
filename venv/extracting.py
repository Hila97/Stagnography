from PIL import Image
import numpy as np
from binaryString import *
from encAndDec import *
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
            x=0
            # print("first block")  # skip the first block
        else:
            # message += startBlockExtraction(matrix, i, j, lenMsgDec, len(message))#the length is the orignelm cant use it
            message += startBlockExtraction(matrix, i, j, lengthOfSecret, len(message))

# print("the message is:")
x = message[0:lengthOfSecret]
# print(x)

def getImgBack(x):
    msg = x
    n = 8
    chunks = [msg[i:i + n] for i in range(0, len(msg), n)]
    r = int(math.sqrt(len(chunks)))
    WIDTH = r
    HEIGHT = r
    for i in range(0, len(chunks)):
        chunks[i] = bin2dec(chunks[i])
    d = [chunks[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    # for row in d:
    #     print(' '.join('{:3}'.format(value) for value in row))

    X = d
    # convert the list back to image
    array = np.array(d, dtype='uint8')
    new_image = Image.fromarray(array, 'L')
    new_image.save('secretImgBack.png')
    print("the secret image has been saved un the file name: secretImgBack.png")



flag = int(input("extract msg-0, extract img-1: "))
if flag == 1:
    getImgBack(x)
elif flag == 0:
    msgBits = bits2a(x)
    print(msgBits)
    print("*********************************************************")
    print(decryptMsg(msgBits))
    print("*********************************************************")

else:
    print("error")


