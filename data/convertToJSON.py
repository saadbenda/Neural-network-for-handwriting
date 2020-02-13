wordsPath='words.txt'
json=wordsPath[:len(wordsPath)-4]+'.json'

fjson=open(json,'a')

#partie des commentaires
fjson.write('# ---  words.json --------------------------------\n'+
            '#database information\n'+
            '#\n'+
            '#format: a01-000u-00-00 : A\n'+
            '#\n'+
            '#a01-000u-00-00  -> word id for line in form a01-000u (in words/a01/a01-000u/a01-000u-00-00.png)\n'+
            '#A               -> the transcription for this word\n'
            )

fjson.write('\n{\n')

f=open(wordsPath)

for line in f:
    excepted=['#','\n']
    if not line or line[0] in excepted:
        continue
    
    lineSplit = line.strip().split(' ')
    assert len(lineSplit) >= 9
    
    fileNameSplit = lineSplit[0]
    gtText=lineSplit[8]
    
    line= fileNameSplit+ ' : '+ gtText+'\n'
    fjson.write(line)

f.close()
fjson.write('\n}')
fjson.close()


