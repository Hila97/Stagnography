import pyDes
vDes = pyDes.des("DESCRYPT", pyDes.CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5)


#from string to binary
def a2bits(a):
    return ''.join(format(ord(i), '08b') for i in str(a))

#from binary to string
def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))



def encrypt(data):#encrypt string by DES
     return vDes.encrypt(data)

def decrypt(data):#decrypt string by DES
    return vDes.decrypt(data)



print('Enter the message')
mystring = input()
print("The original string is : " + mystring)
mystringEncrypt=encrypt(mystring)
print("The original string is +encrypt : " + str(mystringEncrypt))
mystringEncryptBinary=a2bits(mystringEncrypt)
print("The string +encrypt after binary conversion : " +mystringEncryptBinary )
stringnoraml=bits2a(mystringEncryptBinary)
print("The string back to normal : " +stringnoraml )
print("The string back to normal+decrypt : " + str(decrypt(stringnoraml)))




print ("Encrypted: %r" % d)
print ("Decrypted: %r" % k.decrypt(d))
