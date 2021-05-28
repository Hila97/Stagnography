from PIL import Image
import numpy as np

k=3

def converImgToMatrix(route):
    img = Image.open(route).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata()) # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    return data

def printMatrix(data):
    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].
    for row in data:
        print(' '.join('{:3}'.format(value) for value in row))

def paddingOneByOne(sourceMatrix):
    ##add one line , to be divided by 3
    nmpy_matrix=np.matrix(sourceMatrix) #from matrix to numpy
    new_row=nmpy_matrix[length-1] #equal to the last row
    A = np.concatenate((nmpy_matrix,new_row)) #add the last row again
    B = np.insert(A, length, values=A[:,length-1], axis=1)#add the last col again
    B# is numpy object
    return B.tolist() #python list

def dec2bin(x): #from decimal number
    return int(bin(x)[2:])

def bin2dec(x): #from binary number (not string)
    binary = str(x)
    return int(binary, 2)

def kLsb(binaryNumber):#get binary number and return its k lsb in binary
    lsb=pow(10,k)
    return binaryNumber%lsb

def kMsb(binaryNumber):#get binary number and return its k msb in binary
    length=len(str(binaryNumber))
    x=length-k
    msb=pow(10,x)
    return binaryNumber//msb


def rangeTableWithSundivisions(Dm):#return lower limit, and t
    if 0<=Dm<=7:
        return 0,3
    if 8<=Dm<=15:
        return 8,3
    if 16<=Dm<=31:
        return 16,4
    if 32<=Dm<=63:
        return 32,5
    if 64<=Dm<=255:
        return 64,6



secretInBinary=11001110111010100111010111101001011001110
def startBlockEmbedding(matrix,startRow,startCol): #the embedding process
    #step1
    pir=matrix[startRow+1][startCol+1] #the middle field= pir
    #step2
    lsbir=bin2dec(kLsb(dec2bin(pir)))#take k rightmost LSBs of Pir
    sir=bin2dec(kMsb(dec2bin(secretInBinary)))#WAIT TO TAL----- #take k left-most bits of secret message
    #step3
    p1ir = pir-lsbir+sir #p'ir
    #step4
    d=lsbir-sir
    if d>pow(2,k-1) and 0<=p1ir+pow(2,k) <=255:
        p1ir=p1ir+2**k
    elif d<-pow(2,k-1) and 0<=p1ir-pow(2,k) <=255:
        p1ir=p1ir-2**k
    else:
        p1ir=p1ir
    #step5
    P=[] #the length will be 9
    P.append(0) #to start from index 1
    D=[]#the length will be 9
    D.append(0) #to start from index 1
    #p1
    P.append(matrix[startRow][startCol])
    #p2
    P.append(matrix[startRow][startCol+1])
    #p3
    P.append(matrix[startRow][startCol+2])
    #p4
    P.append(matrix[startRow+1][startCol])
    #p5
    P.append(matrix[startRow+1][startCol+2])#pay attaion to the skip here beacause the middle box
    #p6
    P.append(matrix[startRow+2][startCol])
    #p7
    P.append(matrix[startRow+2][startCol+1])
    #p8
    P.append(matrix[startRow+2][startCol+2])
    #comupte D
    for index in range(1,len(P)):#run thoreu 8
        D.append(abs(p1ir-P[index]))
    #step6
    L=[]
    L.append(0)
    t=[]
    t.append(0)
    for index in range(1,len(D)):
        Lj,tj=rangeTableWithSundivisions(D[index])
        L.append(Lj)
        t.append(tj)
    #step7










    # for i in range(startRow,startRow+3,1):
    # #     for j in range(startCol,startCol+3,1):
    #         print(matrix[i][j],end=" ")
    #        print()





matrix= converImgToMatrix('eggs.png')
length=len(matrix)



# for i in range(length):
#     for j in range(length ):
#         print(matrix[i][j], end =" ")
#     print()




# skip 3
for i in range(0,length-1,3):
    for j in range(0,length-1,3):
        startBlockEmbedding(matrix,i,j)
        # print()





##########################################decoding##############################

import matplotlib.pyplot as plt
import numpy as np

X=matrix
plt.imshow(X, cmap="gray")
# plt.show()
plt.savefig('foo.png')

