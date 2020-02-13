import os
import shutil
import argparse

class FilePaths:
    "filenames and paths to data"
    fnTrain = '../data/'
    fnModelPath='../model/'
    
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-u','--user', help='select an user to do an action (ex : formated)')
group.add_argument('-r','--removeUser', help='remove an user form NN')
group.add_argument('-ra','--removeAllUsers', help='remove all instances of NN', action='store_true')
group.add_argument('-a','--addUser', help='add new user to NN')
group.add_argument('-pu', '--printUsers', help='print all instances of NN', action='store_true')
parser.add_argument('--formated', help='to create an user not trained', action='store_true')


args = parser.parse_args()

if args.user:
    if args.formated:
        #just model
        modelUserDir=os.path.join(FilePaths.fnModelPath, args.user)
        if os.path.exists(modelUserDir):
            for fic in os.listdir(modelUserDir):
                os.remove(os.path.join(modelUserDir, fic))
        

if args.addUser:
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'a')
    fo.write(args.addUser+'\n')
    fo.close 
    #model    
    if not os.path.exists(os.path.join(FilePaths.fnModelPath,args.addUser)):
        os.makedirs(os.path.join(FilePaths.fnModelPath,args.addUser))  #create directory ../model/user
        if not args.formated:
            for fic in os.listdir(FilePaths.fnModelPath):
                exceptedCopyFiles=['usersList.txt']
                if os.path.isfile(os.path.join(FilePaths.fnModelPath, fic)) and fic not in exceptedCopyFiles:
                    shutil.copy2(os.path.join(FilePaths.fnModelPath,fic), os.path.join(FilePaths.fnModelPath,args.addUser))
    #data
    if not os.path.exists(os.path.join(FilePaths.fnTrain,args.addUser)):
        os.makedirs(os.path.join(FilePaths.fnTrain,args.addUser))   #create directory ../data/user
        for fic in os.listdir(FilePaths.fnTrain):
            fileToCopy=['corpus.txt', 'test.png', 'words.txt', 'words.json']
            if os.path.isfile(os.path.join(FilePaths.fnTrain, fic)) and fic in fileToCopy:
                shutil.copy2(os.path.join(FilePaths.fnTrain,fic), os.path.join(FilePaths.fnTrain,args.addUser))
    
        

if args.removeAllUsers:
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'r')
    f=fo.read().splitlines()
    for line in f:
        #model
        if os.path.exists(os.path.join(FilePaths.fnModelPath, line)):
            shutil.rmtree(os.path.join(FilePaths.fnModelPath, line))
        #data
        if os.path.exists(os.path.join(FilePaths.fnTrain, line)):
            shutil.rmtree(os.path.join(FilePaths.fnTrain, line))
    fo.close
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'w')
    fo.close


if args.removeUser:
    #model
    if os.path.exists(os.path.join(FilePaths.fnModelPath, args.removeUser)):
        shutil.rmtree(os.path.join(FilePaths.fnModelPath, args.removeUser))
        
    #data
    if os.path.exists(os.path.join(FilePaths.fnTrain, args.removeUser)):
        shutil.rmtree(os.path.join(FilePaths.fnTrain, args.removeUser))
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'r')
    f=fo.read().splitlines()
    
    #remove from usersList
    contenu = ""
    for line in f:
        if line!=args.removeUser:
            contenu += line+'\n'
    fo.close
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'w')
    fo.write(contenu)
    fo.close
     
        
if args.printUsers:
    fo=open(FilePaths.fnModelPath+'usersList.txt', 'r')
    f=set(fo.read().splitlines())
    for line in f:
        print(line)
    fo.close
    
    
fo=open(FilePaths.fnModelPath+'usersList.txt', 'r')
f=fo.read().splitlines()
comp=0
for line in f:
    if args.addUser:
        if line==args.addUser:
            comp+=1
        if comp==2:
            print("this user was still present : data's user has been restored")
            break
fo.close
