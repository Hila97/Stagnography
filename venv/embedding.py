#from PIL import Image
import numpy as np
from binaryString import *
from functions import *
from encAndDec import *
# from methodmus import *
import matplotlib.pyplot as plt





##########################################################################################################
# encode
img= input("hello, which image do you want to use as cover image? ")
matrix = converImgToMatrix(img)

length = len(matrix)
print("length of img:", length)

if length % 3 != 0:
     # print("padding +1 ")
     matrix = paddingOneByOne(matrix)
     length = len(matrix)

if length % 3 != 0:
    # print("padding +2 ")
    matrix = paddingOneByOne(matrix)
    length = len(matrix)

array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('before.png')

msg=input("please enter the message you want: ")
# msg="This message is embedded in an image!"
enc=encryptMsg(msg)
print("the ecrypting: ",enc )
msgBin, lenMsgBin = getMsg(enc)

embeddFirstBlock(matrix, lenMsgBin)

lenMsgDec=bin2dec(lenMsgBin)

index = 0
for i in range(0, length - 1, 3):
    for j in range(0, length - 1, 3):
        if i == 0 and j == 0:
            x=0
            # print("first block")  # skip the first block
        elif index < len(msgBin):  # only if there is more to embedd
            index = startBlockEmbedding(matrix, i, j, index, msgBin)




array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('after.png')


