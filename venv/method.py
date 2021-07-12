from PIL import Image
import numpy as np
from binaryString import *
import matplotlib.pyplot as plt

size = 9

k = 3


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


def paddingOneByOne(sourceMatrix):
    ##add one line , to be divided by 3
    length = len(sourceMatrix)
    nmpy_matrix = np.matrix(sourceMatrix)  # from matrix to numpy
    new_row = nmpy_matrix[length - 1]  # equal to the last row
    A = np.concatenate((nmpy_matrix, new_row))  # add the last row again
    B = A.tolist()
    # print(B)
    for i in range(len(B)):
        B[i].append(B[i][length - 1])

    # B = np.insert(A, length, values=A[:, length - 1], axis=1)  # add the last col again
    # is numpy object
    return B


def dec2bin(x):  # from decimal number
    return int(bin(x)[2:])


def bin2dec(x):  # from binary number (not string)
    binary = str(x)
    return int(binary, 2)


def kLsb(binaryNumber):  # get binary number and return its k lsb in binary
    lsb = pow(10, k)
    return binaryNumber % lsb


def kMsb(startIndex, size, binaryNumber):  # get binary number and return its k msb in binary
    length = len(str(binaryNumber))
    toDivide = startIndex + size
    x = abs(length - toDivide)
    division = pow(10, x)
    ans = int(binaryNumber) // division
    modulo = pow(10, size)
    return ans % modulo


def rangeTableWithSundivisions(Dm):  # return lower limit, and t
    if 0 <= Dm <= 7:
        return 0, 3
    if 8 <= Dm <= 15:
        return 8, 3
    if 16 <= Dm <= 31:
        return 16, 4
    if 32 <= Dm <= 63:
        return 32, 5
    if 64 <= Dm <= 255:
        return 64, 6


def startBlockEmbedding(matrix, startRow, startCol, position, secretInBinary):  # the embedding process
    length=len(secretInBinary)
    if position>=len(secretInBinary):
        print("it should stop")
    # step1
    pir = matrix[startRow + 1][startCol + 1]  # the middle field= pir

    # step2
    lsbir = bin2dec(kLsb(dec2bin(pir)))  # take k rightmost LSBs of Pir
    x=k
    if position+k>=length:
        x=length-position
    sir = bin2dec(kMsb(position, x, secretInBinary))  # take k left-most bits of secret message
    position += x  # start to take bits from the secret messages

    # step3
    p1ir = pir - lsbir + sir  # p'ir
    # step4
    d = lsbir - sir
    if d > pow(2, k - 1) and 0 <= p1ir + pow(2, k) <= 255:
        p1ir = p1ir + 2 ** k
    elif d < -pow(2, k - 1) and 0 <= p1ir - pow(2, k) <= 255:
        p1ir = p1ir - 2 ** k
    else:
        p1ir = p1ir
    # step5
    P = []  # the length will be 9
    P.append(0)  # to start from index 1
    D = []  # the length will be 9
    D.append(0)  # to start from index 1
    # p1
    P.append(matrix[startRow][startCol])
    # p2
    P.append(matrix[startRow][startCol + 1])
    # p3
    P.append(matrix[startRow][startCol + 2])
    # p4
    P.append(matrix[startRow + 1][startCol])
    # p5
    P.append(matrix[startRow + 1][startCol + 2])  # Pay attention to the skip here beacause the middle box
    # p6
    P.append(matrix[startRow + 2][startCol])
    # p7
    P.append(matrix[startRow + 2][startCol + 1])
    # p8
    P.append(matrix[startRow + 2][startCol + 2])
    # comupte D
    for index in range(1, size):  # run thoreu 8
        D.append(abs(p1ir - P[index]))
    # step6
    L = []
    L.append(0)
    t = []
    t.append(0)
    for index in range(1, size):
        Lj, tj = rangeTableWithSundivisions(D[index])
        L.append(Lj)
        t.append(tj)
    # step7

    D1 = []
    D1.append(0)
    for index in range(1, size):
        x=t[index]
        if position + t[index] >= length:
            x = length - position

        sm = bin2dec(kMsb(position, x, secretInBinary))
        position += x
        D1.append(L[index] + sm)


    # step8
    Pm2 = []
    Pm2.append(0)
    Pm3 = []
    Pm3.append(0)
    for index in range(1, size):
        Pm2.append(p1ir - D1[index])
        Pm3.append(p1ir + D1[index])

    # step9
    Pm1 = []
    Pm1.append(0)
    for m in range(1, size):
        if abs(P[m] - Pm2[m]) < abs(P[m] - Pm3[m]) and 0 <= Pm2[m] <= 255:
            Pm1.append(Pm2[m])
        elif abs(P[m] - Pm3[m]) < abs(P[m] - Pm2[m]) and 0 <= Pm3[m] <= 255:
            Pm1.append(Pm3[m])
        elif Pm3[m] > 255 and Pm2[m] > 0:
            Pm1.append(Pm2[m])
        elif Pm2[m] < 0 and Pm3[m] < 255:
            Pm1.append(Pm3[m])
        else: #need to fix it
            print("else")
            # Pm1.append(122)
            Pm1.append(122)


    # copy and change the block
    matrix[startRow][startCol] = Pm1[1]
    matrix[startRow][startCol + 1] = Pm1[2]
    matrix[startRow][startCol + 2] = Pm1[3]
    matrix[startRow + 1][startCol] = Pm1[4]
    matrix[startRow + 1][startCol + 1] = p1ir
    matrix[startRow + 1][startCol + 2] = Pm1[5]
    matrix[startRow + 2][startCol] = Pm1[6]
    matrix[startRow + 2][startCol + 1] = Pm1[7]
    matrix[startRow + 2][startCol + 2] = Pm1[8]

    return position




# Extraction Process

def tillnow(arr,lenTillNow):
    sum=lenTillNow+k
    for x in arr:
        sum+=len(str(x))
    return sum
def startBlockExtraction(matrix, startRow, startCol,lengthOfSecret,lenTillNow):
    # step1
    P1ir = matrix[startRow + 1][startCol + 1]
    s1ir = bin2dec(kLsb(dec2bin(P1ir)))
    # step2
    P1m = []  # the length will be 9
    P1m.append(0)  # to start from index 1
    # p1
    P1m.append(matrix[startRow][startCol])
    # p2
    P1m.append(matrix[startRow][startCol + 1])
    # p3
    P1m.append(matrix[startRow][startCol + 2])
    # p4
    P1m.append(matrix[startRow + 1][startCol])
    # p5
    P1m.append(matrix[startRow + 1][startCol + 2])  # Pay attention to the skip here beacause the middle box
    # p6
    P1m.append(matrix[startRow + 2][startCol])
    # p7
    P1m.append(matrix[startRow + 2][startCol + 1])
    # p8
    P1m.append(matrix[startRow + 2][startCol + 2])
    D1m = []
    D1m.append(0)
    for m in range(1, size):
        D1m.append(abs(P1ir - P1m[m]))

    # step34
    L = []
    L.append(0)
    t = []
    t.append(0)
    for index in range(1, size):
        Lj, tj = rangeTableWithSundivisions(D1m[index])
        L.append(Lj)
        t.append(tj)


    # step5
    sm = []
    sm.append(0)
    for m in range(1, size):
        sm.append(D1m[m] - L[m])

    # step 6
    # to binary
    smbin = []
    smbin.append(0)
    for m in range(1, size):
        smbin.append(dec2bin(sm[m]))
    # print(smbin)
    # to string
    smbinWitht=[]
    smbinWitht.append(0)

    message = ''
    for m in range(1, size):
        smbinString = str(smbin[m])
        y=t[m]
        if y+tillnow(smbinWitht,lenTillNow)>=lengthOfSecret:
            y= lengthOfSecret-tillnow(smbinWitht,lenTillNow)

        diffrence = y - len(smbinString)
        temp = ''

        if diffrence > 0:
            for i in range(diffrence):
                temp += '0'
        smbinWitht.append(temp + smbinString)
    for m in range(1, size):
        message += smbinWitht[m]
    # print(message)
    # compute the Kmsb of the message by the Klsb of p'ir
    Kmsb = kLsb(dec2bin(P1ir))
    message = str(Kmsb) + message

    return message


def kmsbStr(position, k, msg):
    length = len(msg)
    x = ''
    for i in range(position, position + k):
        x += msg[i]
    return x


def embeddFirstBlock(matrix, lengthMsgInBits):
    print(lengthMsgInBits)
    position = 0
    startCol = 0
    startRow = 0
    # step1
    pir = matrix[startRow + 1][startCol + 1]  # the middle field= pir

    # step2
    lsbir = bin2dec(kLsb(dec2bin(pir)))  # take k rightmost LSBs of Pir

    sir = bin2dec(kmsbStr(position, k, lengthMsgInBits))  # take k left-most bits of secret message
    # step3
    p1ir = pir - lsbir + sir  # p'ir
    # step4
    d = lsbir - sir
    if d > pow(2, k - 1) and 0 <= p1ir + pow(2, k) <= 255:
        p1ir = p1ir + 2 ** k
    elif d < -pow(2, k - 1) and 0 <= p1ir - pow(2, k) <= 255:
        p1ir = p1ir - 2 ** k
    else:
        p1ir = p1ir
    # step5
    P = []  # the length will be 9
    P.append(0)  # to start from index 1
    D = []  # the length will be 9
    D.append(0)  # to start from index 1
    # p1
    P.append(matrix[startRow][startCol])
    # p2
    P.append(matrix[startRow][startCol + 1])
    # p3
    P.append(matrix[startRow][startCol + 2])
    # p4
    P.append(matrix[startRow + 1][startCol])
    # p5
    P.append(matrix[startRow + 1][startCol + 2])  # Pay attention to the skip here beacause the middle box
    # p6
    P.append(matrix[startRow + 2][startCol])
    # p7
    P.append(matrix[startRow + 2][startCol + 1])
    # p8
    P.append(matrix[startRow + 2][startCol + 2])
    # comupte D
    for index in range(1, 5):  # run thoreu 8
        D.append(abs(p1ir - P[index]))
    # step6
    L = []
    L.append(0)
    t = []
    t.append(0)
    for index in range(1, 5):
        Lj, tj = rangeTableWithSundivisions(D[index])
        L.append(Lj)
        t.append(tj)
    # step7
    D1 = []
    D1.append(0)
    for index in range(1, 5):
        sm = bin2dec(kmsbStr(position, 3, lengthMsgInBits))
        position += 3
        D1.append(L[index] + sm)

    # step8
    Pm2 = []
    Pm2.append(0)
    Pm3 = []
    Pm3.append(0)
    for index in range(1, 5):
        Pm2.append(p1ir - D1[index])
        Pm3.append(p1ir + D1[index])
    # step9
    Pm1 = []
    Pm1.append(0)
    for m in range(1, 5):
        if abs(P[m] - Pm2[m]) < abs(P[m] - Pm3[m]) and 0 <= Pm2[m] <= 255:
            Pm1.append(Pm2[m])
        elif abs(P[m] - Pm3[m]) < abs(P[m] - Pm2[m]) and 0 <= Pm3[m] <= 255:
            Pm1.append(Pm3[m])
        elif Pm3[m] > 255 and Pm2[m] > 0:
            Pm1.append(Pm2[m])
        elif Pm2[m] < 0 and Pm3[m] < 255:
            Pm1.append(Pm3[m])
        else:
            # Pm1.append(122) #i added it , need to be fixed
            print("else in first block")
            Pm1.append(122)


    # copy and change the block
    matrix[startRow][startCol] = Pm1[1]
    print(Pm1)
    matrix[startRow][startCol + 1] = Pm1[2]
    matrix[startRow][startCol + 2] = Pm1[3]
    matrix[startRow + 1][startCol] = Pm1[4]
    matrix[startRow + 1][startCol + 1] = p1ir


def extraxtFirstBlock(matrix):
    startRow = 0
    startCol = 0

    # step1
    P1ir = matrix[startRow + 1][startCol + 1]
    s1ir = bin2dec(kLsb(dec2bin(P1ir)))
    # step2
    P1m = []  # the length will be 9
    P1m.append(0)  # to start from index 1
    # p1
    P1m.append(matrix[startRow][startCol])
    # p2
    P1m.append(matrix[startRow][startCol + 1])
    # p3
    P1m.append(matrix[startRow][startCol + 2])
    # p4
    P1m.append(matrix[startRow + 1][startCol])
    # p5
    P1m.append(matrix[startRow + 1][startCol + 2])  # Pay attention to the skip here beacause the middle box
    # p6
    P1m.append(matrix[startRow + 2][startCol])
    # p7
    P1m.append(matrix[startRow + 2][startCol + 1])
    # p8
    P1m.append(matrix[startRow + 2][startCol + 2])
    D1m = []
    D1m.append(0)

    for m in range(1, 5):
        D1m.append(abs(P1ir - P1m[m]))

    # step34
    L = []
    L.append(0)
    t = []
    t.append(0)
    for index in range(1, 5):
        Lj, tj = rangeTableWithSundivisions(D1m[index])
        L.append(Lj)
        t.append(tj)

    # step5
    sm = []
    sm.append(0)
    for m in range(1, 5):
        sm.append(D1m[m] - L[m])

    # step 6
    # to binary
    smbin = []
    smbin.append(0)
    for m in range(1, 5):
        smbin.append(dec2bin(sm[m]))
    # print(smbin)
    # to string
    message = ''
    for m in range(1, 5):
        smbin[m] = str(smbin[m])
        diffrence = 3 - len(smbin[m])
        if diffrence != 0:
            temp = ''
            for i in range(diffrence):
                temp += '0'
            smbin[m] = temp + smbin[m]
    # print(smbin)
    for m in range(1, 5):
        message += smbin[m]
    # print(message)
    # compute the Kmsb of the message by the Klsb of p'ir
    return message


def addZeros(size):
    msg = ''
    for i in range(size):
        msg += '0'
    return msg


def getMsg():  # get a message and find it binary length, no more than 12, if less than padding
    msg = "he"
    msgBin = a2bits(msg)
    print("the message in binary: ")
    # msgBin=str(1100111010100111010111101001011001110110011101010011101011110100101100111011001110101001110101111010010110011101100111010100111010111101001011001110) #check exemple need to delete
    print(bits2a(msgBin))
    print(msgBin)
    lenMsg = len(msgBin)
    print("the length of the message: ")
    print(lenMsg)

    lenBinStr = str(dec2bin(lenMsg))
    if len(lenBinStr) > 12:
        print('the massage is to big')
        exit
    else:
        size = 12 - len(lenBinStr)
        lenBinStr = addZeros(size) + lenBinStr

    return msgBin,lenBinStr

##########################################################################################################
# encode

matrix = converImgToMatrix('lena.png')
# matrix=[[0,0,0,255,255,255,122,122,122],[122,122,122,0,0,0,122,122,122],[200,200,200,122,122,122,122,122,122],[0,0,0,255,255,255,122,122,122],[122,122,122,0,0,0,122,122,122],[200,200,200,122,122,122,122,122,122],[0,0,0,255,255,255,122,122,122],[122,122,122,0,0,0,122,122,122],[200,200,200,122,122,122,122,122,122]]
length = len(matrix)
print("length of img")
print(length)
if length%3!=0:
    print("padding +1 ")
    matrix=paddingOneByOne(matrix)
if length%3!=0:
    print("padding +2 ")
    matrix=paddingOneByOne(matrix)

# printMatrix(matrix)
array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('egbefore.png')


msgBin,lenMsgBin = getMsg()

# printMatrix(matrix)
embeddFirstBlock(matrix, lenMsgBin)
# printMatrix(matrix)


index = 0
for i in range(0, length -1, 3):
    for j in range(0, length -1, 3):
        if i==0 and j==0 :
            print("first block")# skip the first block
        elif index<len(msgBin): #only if there is more to embedd
            index = startBlockEmbedding(matrix, i, j, index, msgBin)

# X = matrix
# plt.imshow(X, cmap="gray")
# plt.savefig('after.png')
array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('egafter.png')

##########################################################################################################
# decode
matrix = converImgToMatrix('egafter.png')
lengthMatrix = len(matrix)


lengthMsg = extraxtFirstBlock(matrix)
lengthOfSecret = bin2dec(lengthMsg)
print("length after decode")
print(lengthOfSecret)
position=0
message=''
for i in range(0, length -1, 3):
    for j in range(0, length -1, 3):
        if i==0 and j==0 :
            print("first block")# skip the first block
        else:
            message += startBlockExtraction(matrix, i, j,lengthOfSecret,len(message))



print("message")
print(message)
message='0'+message
print(message)
x=message[0:lengthOfSecret]
# x=message[0:72]

print(bits2a(x))


array = np.array(matrix, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('egafter2.png')
