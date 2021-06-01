from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('eggs.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata())  # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# For example:
print("pic 1")
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))

X = data
# convert the list back to image
array = np.array(data, dtype='uint8')
new_image = Image.fromarray(array, 'L')
new_image.save('testing2.png')

# plt.imshow(X, cmap="gray")
# plt.show()
# check testing.png

img2 = Image.open('testing2.png').convert('L')  # convert image to 8-bit grayscale
WIDTH2, HEIGHT2 = img2.size
data2 = list(img.getdata())
data2 = [data2[offset:offset + WIDTH2] for offset in range(0, WIDTH2 * HEIGHT2, WIDTH2)]

print("pic 2")
for row in data2:
    print(' '.join('{:3}'.format(value) for value in row))


def compare(a, b): return a == b


print(compare(data, data2))
