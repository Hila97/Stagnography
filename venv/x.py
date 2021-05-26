#

# file_name = 'lena.bmp';
# cover_image = imread(file_name);
# [row, col] = size(cover_image);
# % secret
# image
# file_name = 'baboon.bmp';
# secret_image = imread(file_name);
# secret_image = imresize(secret_image, [row, col]);
# stego_image = cover_image;
# for ii=1:row
#     for jj=1:col
#     stego_image(ii, jj) = bitset(stego_image(ii, jj), 1, secret_image(ii, jj));
#     end
# end
# imwrite(uint8(stego_image), 'Stegoimage.bmp')

# import pillow

from PIL import image
im=image.open('baboon.bmp','r')
pix_val=list(getdata())
pix_val_flat = [x for sets in pix_val for x in sets]
print(pix_val)
print(pix_val_flat)