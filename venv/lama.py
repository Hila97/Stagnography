from PIL import Image

img = Image.open('lena.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))


import matplotlib.pyplot as plt
import numpy as np

X=data
plt.imshow(X, cmap="gray")
# plt.show()
plt.savefig('foo.png')

