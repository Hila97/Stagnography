# Stagnography


## table of contents:
* [Introduction](#Introduction:)
* [technologies](#Technologies:)
* [Launch](#Launch:)
* [Authors](#Authors:)

## Introduction:
The software implements the algorithms of the article **"A Spatial Domain Steganographic Approach Using Pixel Pair Differencing and LSB Substitution"**.<br />
This system is software for encrypting a message or image within a grayscale image as well as extracting an image or message encrypted within an image.<br />
The software has 2 operations: embedding and extraction, in both the cover image is divided into 3x3 blocks as a basis for the work of the algorithm proposed in the article.<br />
There are 2 options: embedding a message (string format) or embedding an image.<br /> 
In the first block (top left corner) the length of the message or alternatively the dimensions of the image will be embedded depending on the type of data embedded and therefore there are restrictions on the length of the message / image size.<br /> 
In the other blocks according to the method proposed in the article, the message / image is embedded according to the pairs of pixels in each block.
<br />If you choose to embed an message, as part of the upgrades it will undergo a double encryption process.
<br />If you choose to embed an image, it will go through a process of conversion to a string that symbolizes the pixels in binary.<br />
At the extraction stage the message or image respectively will appear in their entirety as inserted.


## 💻technologies:
* Language:<br /> 
python
* Workspace:<br /> 
  PyCharm
* Libraries:<br /> 
 numpy <br /> 
 matplotlib.pyplot <br /> 
 PIL <br /> 



## 🛠️ Launch
#### Installation Steps:<br />
1. Open folder for this project and clone this repository use follow command: <br />
 https://github.com/Mussil/Stagnography.git

2. To run this project, install it locally:  <br /> 
``` install numpy ```  <br /> 
``` install matplotlib.pyplot ```  <br /> 
``` install PIL ``` <br /> 

#### Run the app: <br />
###### The embedding process:

* Run the embedding.py file <br />
  - Type the image you want to encrypt.
  - Choose what you want to encrypt:
    - 0- Message- In this case write the message you want to encrypt.
    - 1- Image- In this case enter the name of the image you want to encrypt.
 > After the embedding phase the cover image will be modified and saved in a file named after.png
 ###### The extracting process: 
 * Run the extracting.py file
   - Select extracting of :
     - 0- Message, if a message is extracted, it can be seen.
     - 1- Image, if an image is extracted it will be saved as secretimgBack.png
  
  
## 📗 Authors:
* https://github.com/Mussil
* https://github.com/Hila97
* https://github.com/TalFarhan
* https://github.com/hodaypi
