from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('eggs.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata())  # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].
print(data[0][0])
# For example:
print("pic 1")
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))

# # Here's another more compact representation.
# chars = '@%#*+=-:. '  # Change as desired.
# scale = (len(chars)-1)/255.
# print()
# for row in data:
#     print(' '.join(chars[int(value*scale)] for value in row))
import matplotlib.pyplot as plt
import numpy as np

# X = np.random.random((100, 100))  # sample 2D array
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

z = '0100011101100101011001010110101101110011'
y = '10001111100101110010111010111110011'

def compare(a, b): return a == b

def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)] * 8))

print(compare(data, data2))

import base64
import requests

my_url = 'lena.png'  # your url here


def pass_image(image_url):
    output = base64.b64encode(requests.get(image_url).content)
    bin = "".join(format(ord(x), "b") for x in base64.decodestring(output))
    print(bin)
    return bin  # or you could print it


len(pass_image(my_url))  # for the url I used, I got length of 387244

import base64
with open("lena.png", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
print(my_string)
