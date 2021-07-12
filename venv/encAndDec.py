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


x = Encryption(9)

def encryptMsg(msg):
    return x.encrypt(msg)

def decryptMsg(secret_message):
    return x.decrypt(str(secret_message), 9)