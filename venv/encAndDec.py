import string
import random

class Encryption():

    def __init__(self,seed):

        # Sets a random seed and a self.seed attribute
        random.seed(seed)

        self.seed = seed

        # Creates an empty string attribute to hold the encrypted phrase
        self.encrypted_message = ""

        # One is the standard alphabet, the other is a shuffled alphabet
        self.true_alphabet = list(string.ascii_lowercase)
        self.random_alphabet = random.sample(self.true_alphabet, len(self.true_alphabet))

    def encrypt(self, message):
        """
        This method will take in a string message and encrypt it.
        """

        output = ""

        # Replaces every letter with a random letter
        for i in range(len(message)):
            output += message[i]
            output += random.sample(self.true_alphabet, 1)[0]

        # Reverses the string
        self.encrypted_message = output[::-1]

        # Uses a random shuffled alphabet for a caesar cipher
        encrypted_message_two = list(range(len(self.encrypted_message)))

        for i,letter in enumerate(self.encrypted_message.lower()):

            if letter in self.true_alphabet:
                index = self.true_alphabet.index(letter)
                encrypted_message_two[i] = self.random_alphabet[index]
            # For punctuation and spaces
            else:
                encrypted_message_two[i] = letter

        self.encrypted_message = "".join(encrypted_message_two)
        return self.encrypted_message
        pass

    def decrypt(self,message,seed):
        """
        This method takes in a messsage and a seed for the random shuffled alphabet.
        It then returns the decrypted alphabet.
        """

        random.seed(seed)
        session_random_alphabet = random.sample(self.true_alphabet, len(self.true_alphabet))

        decrypted_message = list(range(len(message)))

        # Undo randomized cipher
        for i, letter in enumerate(message.lower()):

            if letter in self.true_alphabet:
                index = session_random_alphabet.index(letter)
                decrypted_message[i] = self.true_alphabet[index]
            # For punctuation and spaces
            else:
                decrypted_message[i] = letter

        decrypted_message = "".join(decrypted_message)[::-1][::2]
        return decrypted_message

def bits2string(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)] * 8))
def string2bits(s):
    return ''.join(format(ord(i), '08b') for i in s)

x = Encryption(9)
print("input message:")
secret_message = x.encrypt(input())
print("secret_message:\n" ,secret_message)
secret_message_bin=string2bits(secret_message)
print("secret_message_bin:\n"+secret_message_bin)
secret_message_dec=bits2string(secret_message_bin)
print("secret_message_edc:\n"+secret_message_dec)
open_message = x.decrypt(str(secret_message), 9)
print("deciphering:\n",open_message)


