from PIL import Image

img = Image.open('lena.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata())  # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].
print(data[0][0])
# For example:
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
plt.imshow(X, cmap="gray")
plt.show()

test_str = "this is a try"
print("The original string is : " + str(test_str))
res = ''.join(format(ord(i), '08b') for i in test_str)
print("The string after binary conversion : " + str(res))


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)] * 8))


print(bits2a(res))

z = '0100011101100101011001010110101101110011'
y = '10001111100101110010111010111110011'


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)] * 8))


print(bits2a(z))
print(bits2a(y))
print(bits2a('0110100001100101011011000110110001101111'))
print(bits2a('1100111010100111010111101001011001110'))
