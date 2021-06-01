print('Enter the message')
test_str = input()
# test_str = "this is a try"
print("The original string is : " + str(test_str))
res = ''.join(format(ord(i), '08b') for i in test_str)
print("The string after binary conversion : " + str(res))


def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)] * 8))


print(bits2a(res))
