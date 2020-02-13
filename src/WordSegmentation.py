import math
import cv2
import numpy as np


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0, rapprochCoef=7):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
    
    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.
		
        rapprochCoef  : rapport minimal entre la hauteur max de deux img consecutives et le vide les separant pour etre 
                        considere comme une seule image.
        
    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        res.append((currBox, currImg))

    res=sorted(res, key=lambda entry:entry[0][0])
    result=[]
    listNoire=[]
    for i in range(len(res)):
        if i<len(res)-1 and i not in listNoire:
            (wordBox1, wordImg1) = res[i]
            (x1, y1, w1, h1) = wordBox1
            (wordBox2, wordImg2) = res[i+1]
            (x2, y2, w2, h2) = wordBox2

            #print('ecart-->'+str(x2-x1-w1))
            
            if x2-x1-w1 < max(h1,h2)/rapprochCoef:
                listNoire.append(i+1)
                imageFusion=img[min(y1,y2):max((y1+h1),(y2+h2)),  x1:x2+w2]
                boxFusion=(x1,min(y1,y2), w2+x2-x1,max((y1+h1),(y2+h2))-min(y1,y2))
                result.append((boxFusion,imageFusion))

            else:
                result.append(res[i])
        elif i in listNoire :
            continue
        else:
            result.append(res[i])
    # return list of words, sorted by x-coordinate
    return result    



def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2
    
    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta
    
    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize
            
            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
            
            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

