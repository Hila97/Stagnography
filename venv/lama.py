from PIL import Image

img = Image.open('lena.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].
# for row in data:
#     print(' '.join('{:3}'.format(value) for value in row))



import numpy as np
length=len(data)
A = np.vstack([data, data[0]])


##add one line , to be divided by 3
nmpy_matrix=np.matrix(data) #from matrix to numpy
new_row=nmpy_matrix[length-1] #equal to the last row
A = np.concatenate((nmpy_matrix,new_row)) #add the last row again
B = np.insert(A, length, values=A[:,length-1], axis=1)#add the last col again

length=len(B)
# print(length)


matrix = B.tolist()
for i in range(0,length-1,3):
    for j in range(length -1):
        print(matrix[i][j], end =" ")
    print()



def startBlock(matrix,startRow,startCol):
    for i in range(startRow,startRow+3,1):
        for j in range(startCol,startCol+3,1):
            print(matrix[i][j],end=" ")
        print()


#skip 3
# for i in range(0,length-1,3):
#     for j in range(0,length-1,3):
#         startBlock(matrix,i,j)
#         print()







##########################################decoding##############################

import matplotlib.pyplot as plt
import numpy as np

X=data
plt.imshow(X, cmap="gray")
# plt.show()
plt.savefig('foo.png')

