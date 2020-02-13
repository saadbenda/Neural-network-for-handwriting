from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnCorpus = '../data/corpus.txt'
    fnModelPath='../model/'
    fnInfer = '../data/test.png'


def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
        
        # validate
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0

            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate


def infer(model, fnImg, isSegmented=False): 
    if not isSegmented:
        "recognize text in image provided by file path"
        img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, True)
        print('Recognized:', '"' + recognized[0] + '"')
        print('Probability:', probability[0])
    else:
        print("_____________   Segmentation and creation of files ::: ")
        user=model.getUser()
        picture=os.path.join('../data/',user)+'/test.png'
        userPath=FilePaths.fnTrain
        preImg=cv2.imread(picture)
        (height, _, _ )= preImg.shape
        img = prepareImg(preImg, height)
        
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=height*0.14, minArea=1000, rapprochCoef=7)

        # write output    
        if not os.path.exists(os.path.join(userPath,'outSegmentation')):
            os.mkdir(os.path.join(userPath,'outSegmentation'))
        
        #clear files of the outSegmentation directory
        for fic in os.listdir(os.path.join(userPath,'outSegmentation')):
            os.remove(os.path.join(os.path.join(userPath,'outSegmentation'), fic))
            
        # iterate over all segmented words
        allImages=[]
        print('\nSegmented into %d words'%len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite(os.path.join(userPath,'outSegmentation')+'/%d.png'%j, wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
            allImages.append(os.path.join(userPath,'outSegmentation')+'/%d.png'%j)
        # output summary image with bounding boxes around words
        cv2.imwrite(os.path.join(userPath,'outSegmentation')+'/summary.png', img)
        
        mots=''
        probMoyenne=0
        for i in range(len(allImages)):            
            img = preprocess(cv2.imread(allImages[i], cv2.IMREAD_GRAYSCALE), Model.imgSize)
            batch = Batch(None, [img])
            (recognized, probability) = model.inferBatch(batch, True)
            mots+=recognized[0]+'  '
            probMoyenne+=probability[0]
        print('word recognized :  "'+mots+'"')
        if len(allImages)!=0:
            print('average probabilities :' +str(probMoyenne/len(allImages)))


def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('-u','--user', help='add new user to NN')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    parser.add_argument('-s','--segmentation', help= 'segmentation of word', action='store_true')
    
    args = parser.parse_args()

    currentUser=''
    isUser=False
    
    #indique l'utilisateur choisi: Utilisateur principal par defaut
    #création de l'utilisateur s'il ne l'était pas encore.
    if args.user:
        currentUser=args.user #nom de l'utilisateur 
        isUser=True
        #model
        if not os.path.exists(os.path.join(FilePaths.fnModelPath,args.user)):
            os.makedirs(os.path.join(FilePaths.fnModelPath,args.user))
            for fic in os.listdir(FilePaths.fnModelPath):
                if os.path.isfile(os.path.join(FilePaths.fnModelPath, fic)):
                    shutil.copy2(os.path.join(FilePaths.fnModelPath,fic), os.path.join(FilePaths.fnModelPath,args.user))
        #data
        if not os.path.exists(os.path.join(FilePaths.fnTrain,args.user)):
            os.makedirs(os.path.join(FilePaths.fnTrain,args.user))
            for fic in os.listdir(FilePaths.fnTrain):
                if os.path.isfile(os.path.join(FilePaths.fnTrain, fic)):
                    shutil.copy2(os.path.join(FilePaths.fnTrain,fic), os.path.join(FilePaths.fnTrain,args.user))
        
        #redirection vers les fichiers de l'utilisateur dans le cas d'un utilisateur
        FilePaths.fnTrain=os.path.join(FilePaths.fnTrain, args.user)+'/'
        FilePaths.fnCharList=os.path.join('../model/', args.user)+'/charList.txt'
        FilePaths.fnCorpus=os.path.join('../data/', args.user)+'/corpus.txt'
        FilePaths.fnAccuracy=os.path.join('../model/', args.user)+'/accuracy.txt'
        FilePaths.fnInfer=os.path.join('../data/', args.user)+'/test.png'
        FilePaths.fnModelPath=os.path.join('../model/', args.user)
        FilePaths.fnBatchIndexSave=os.path.join('../model/', args.user)+'/lastBatchIndexFile.txt'
        
        

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

        
    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        print('@@ Data loading ... please wait')
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen,isUser)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
        
        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType, user=currentUser)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True, user=currentUser)
            validate(model, loader)

    # infer text on test image
    else:
        isSegmented=False
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, user=currentUser, dump=args.dump)
        if args.segmentation:
            isSegmented=True
        infer(model, FilePaths.fnInfer,isSegmented)


if __name__ == '__main__':
    main()

