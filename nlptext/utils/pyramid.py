import os
import re
import numpy as np 
import pickle
from datetime import datetime

from .infrastructure import strQ2B, fileReader

##################################################################################################CORPUS-FOLDER
# Important One 
def CorpusFoldersReader(CORPUSPath, iden = None):
    # file is the priority
    if iden:
        corpusFiles = [i for i in os.listdir(CORPUSPath) if iden in i]
        return {os.path.join(CORPUSPath, fd): '' for fd in corpusFiles}, 'File'
    else:
        results = [x for x in os.walk(CORPUSPath) if x[2]]
        return {i[0]: i[2] for i in results},                            'Dir'

        
def geneTextFilePaths(corpusPath, orig_iden = '.txt', anno_iden = None):
    FolderNames = [i for i in np.sort(os.listdir(corpusPath)) if i[0] != '.']
    # print(FolderNames)
    FolderDict = {}
    
    for foldername in FolderNames:
        path = corpusPath + foldername
        # TODO: check the path by os
        OrigFileList = [i for i in os.listdir(path) if orig_iden in i ]
        if anno_iden:
            AnnoFileList = [i.replace(orig_iden, anno_iden)  for i in OrigFileList]
            AnnoFileList = [i if os.path.isfile(path + '/' + i) else '' for i in AnnoFileList]
        else:
            AnnoFileList = [''] * len(OrigFileList)
            
        FolderDict[foldername] = OrigFileList, AnnoFileList
    return FolderDict
##################################################################################################CORPUS-FOLDER

##################################################################################################FOLDER-TEXT
def textFileReader(folderPath, fileNames, anno = False, sep = '\t', notZeroIndex = 1,notRightOpen=0, **kwargs):
    # folderPath is textFile path, one text only
    # only here need to take care of annoLevel: text or sent.
    # sep is SSET sep, not Sentence to Token Sep
    ORIGIden = '.txt'
    ANNOIden = anno
    origTextName = None
    annoTextName = None

    origTextNames = [f for f in fileNames if ORIGIden in f]
    for origTextName in origTextNames:
        SSETText = []
        with open(os.path.join(folderPath, origTextName), 'r', encoding = 'utf-8') as f:
            strText = strQ2B(f.read())
        ########################################################## ANNO
        if anno:
            # step 1
            name = origTextName.replace(ORIGIden, '')
            annoFileNames4Text = [f for f in fileNames if name in f and ANNOIden in f]
            
            if sum(['-sent' in f for f in annoFileNames4Text]) >= 1:
                ########################################################## SENT_ANNO
                # step 2
                # sent anno level
                strAnnoText = ''
                sentId   = 0
                annoTextName = []
                for _ in range(len(annoFileNames4Text)):
                    annoSentName = origTextName.replace(ORIGIden, '-sent' + str(sentId) + ANNOIden)
                    while not annoSentName in annoFileNames4Text:
                        print(sentId)
                        print(annoSentName)
                        sentId = sentId + 1
                        annoSentName = origTextName.replace(ORIGIden, '-sent' + str(sentId) + ANNOIden)
                    strAnnoText  = strAnnoText + '\n' + fileReader(os.path.join(folderPath, annoSentName))
                    annoTextName.append(annoSentName)
                    # print(os.path.join(folderPath, annoSentName))
                    sentId = sentId + 1
                    # print(strAnnoSent)
                    
                SSETText = [sset.split(sep)[-4:] for sset in strAnnoText.split('\n') if sep in sset]
                SSETText = [[sset[0], int(sset[1]) - notZeroIndex, int(sset[2]), sset[3]] for sset in SSETText] 
                
                #- print(strText)
                #- print(SSETText)

                ### something different
                txtCharIdx = 0
                collapse   = 0
                
                for ssetIdx, sset in enumerate(SSETText):
                    string, s, e, t = sset
                    string = string.replace('@', ' ') # TODO
                    lenString = len(string)
                    while string != strText[txtCharIdx: txtCharIdx + lenString]:
                        # Detect and Debug errors here.
                        # in case there is no enough sentences.
                        # print('--', txtCharIdx, '--', string)
                        txtCharIdx = txtCharIdx + 1

                    SSETText[ssetIdx] = [string, txtCharIdx, txtCharIdx + lenString, t ]
                    txtCharIdx = txtCharIdx + lenString 
                ########################################################## SENT_ANNO

            else:
                # step 3
                ########################################################## TEXT_ANNO
                strAnnoText = ''
                annoTextName = origTextName.replace(ORIGIden, ANNOIden)
                if annoTextName in fileNames:
                    strAnnoText  = fileReader(os.path.join(folderPath, annoTextName))
                else:
                    print('Error', os.path.join(folderPath, annoTextName))

                SSETText = [sset.split(sep)[-4:] for sset in strAnnoText.split('\n') if sep in sset]
                L = []
                for sset in SSETText:
                    try:
                        L.append([sset[0], int(sset[1]) - notZeroIndex, int(sset[2]) + notRightOpen, sset[3]])     
                    except:
                        pass
                SSETText = L
                ########################################################## TEXT_ANNO
                
        # SSETText = [[sset,sset,sset,sset[3].split('-')[0]] for sset in SSETText]
        yield strText, SSETText, origTextName, annoTextName


def textLineReader(folderPath, fileNames, anno = False, **kwargs):
    with open(folderPath, 'r', encoding = 'utf-8') as f:
        for line in f:
            line = strQ2B(line)
            SSETText = []
            strText = ''
            if anno == 'embed':
                ST = [(block, 'O') if idx%2==0 else (block.split(':')[-1].strip(), block.split(':')[0]) 
                     for idx, block in enumerate(line.replace("}}", '{{').split('{{'))]
                txtCharIdx = 0
                for st in ST:
                    string, tag = st
                    strText = strText + string
                    sset = [string, txtCharIdx, txtCharIdx + len(string), tag]
                    txtCharIdx = sset[2]
                    if tag == 'O':
                        continue
                    SSETText.append(sset) 
            else:
                strText = line
            # SSETText = [[sset,sset,sset,sset[3].split('-')[0]] for sset in SSETText]
            yield strText, SSETText, None, None


def textBlockReader(folderPath, fileNames, anno = True, **kwargs):
    with open(folderPath, 'r', encoding = 'utf-8') as f:
        L = []
        for line in f:
            line = strQ2B(line)
            if line != '\n':
                L.append(strQ2B(line).replace('\n', '').split(' ')) # TODO: maybe different seps
            else:
                # TODO
                strText = ''.join([ct[0] for ct in L])
                CIT = [[ct[0], idx, ct[1]] for idx, ct in enumerate(L) if ct[1] != 'O']
                startIdxes = [idx for idx in range(len(CIT)) if CIT[idx][-1][0] in ['B', 'S']] + [len(CIT)]
                SSETText = []
                for i in range(len(startIdxes)-1):
                    OneCIT = CIT[startIdxes[i]: startIdxes[i+1]]
                    string = ''.join(cit[0] for cit in OneCIT)
                    start, end = OneCIT[0][1], OneCIT[-1][1] + 1
                    tag = OneCIT[0][-1].split('-')[1]
                    SSETText.append([string, start, end, tag])
                # SSETText = [[sset,sset,sset,sset[3].split('-')[0]] for sset in SSETText]
                yield strText, SSETText, None, None
                L = []
    

def textElementReader(foderPath, fileNames, anno = False, **kwargs):
    with open(foderPath, 'rb') as handle:
        L = pickle.load(handle)
        for strText in L:
            strText = strQ2B(strText)
            yield strText, None, None, None


FolderTextsReaders = {
    'file': textFileReader,
    'line': textLineReader,
    'block':textBlockReader,
    'element': textElementReader,
}
##################################################################################################FOLDER-TEXT


##################################################################################################TEXT-SENT
def reCutText2Sent(text, useSep = False):
    
    
    ###################### Remove some weird chars #######################
    text = re.sub('\xa0', '', text)
    
    ############# The Issue of Spaces
    ###################### Convert the Spaces between two English Letters to 'ⴷ' #################
    # Take care of Spaces
    text = re.sub(r'(?<=[A-Za-z])\s+(?=[|A-Za-z])', 'ⴷ',  text)
    
    ###################### Convert the S+ spaces to '〰' #################
    text = re.sub(' {2}', '〰', text ).strip()
    if useSep == ' ':
        # if using space to sep the words
        text = text.replace('\t','').replace('〰', ' ')
    elif useSep == '\t':
        # if using tab to sep the words, removing all spaces
        text = text.replace(' ','').replace('〰', '')
    else:
        # if there is no sep char for Chinese, remove single space, and then convert space+ to single space
        text = text.replace('\t','').replace(' ', '',).replace('〰', ' ')
        
    # convert the spaces between English letters to single spaces
    text = text.replace('ⴷ', ' ')
    
    # Other Things
    text = re.sub('([。！;；])([^”])',  r"\1\n\2",text) 
    text = re.sub('(\.{6})([^”])',     r"\1\n\2",text) 
    text = re.sub('(\…{2})([^”])',     r"\1\n\2",text)
    
    # The \n within " " is not considered
    text = '"'.join( [ x if i % 2 == 0 else x.replace('\n', '') 
                         for i, x in enumerate(text.split('"'))] )
    text = re.sub( '\n+', '\n', text ).strip() # replace '\n+' to '\n'
    text = text.replace('\\n', '\n')
    text = text.split("\n")
    text = [sent.strip() for sent in text]
    # text = [sent.replace(' ', '').replace('\\n', '') for sent in text]
    return [sent for sent in text if len(sent)>=2]

def lineCutText2Sent(fullfilepath):
    with open(fullfilepath, 'r', encoding = 'utf-8') as f:
        for sent in f:
            yield strQ2B(sent).replace('\n', '')

def segText2Sents(text, method = 'whole', **kwargs):
    
    '''
    text:
        1. textfilepath. 2. text-level string
    method: 
        1. 'whole': when text is a text-level string,then use this text-level string as sent-level string directly.
                    and return text = [sent-level string].
        2. `funct`: when method is a function, whose input is a text-level string,
                    then return text = funct(text) = [..., sent-level string, ...]
        3. 'line' : string. when text is filepath where each line is a sentence
                    then return a generator text = generate(text), item is a sent-level string.        
    '''
    if os.path.isfile(text):
        if method == 'line':
            text = lineCutText2Sent(text)
            return text
        else:
            text = fileReader(text)
    if method == 'whole':
        return [text]
    elif method == 're':
        return reCutText2Sent(text, **kwargs)
    else:
        return method(text, **kwargs)
##################################################################################################TEXT-SENT

##################################################################################################SENT-TOKEN
def segSent2Tokens(sent, method = 'iter'):
    if method == 'iter':
        return [i for i in sent]
    elif method[:4] == 'sep-':
        sep = method.replace('sep-', '')
        return [i for i in sent.split(sep) if i != '']
##################################################################################################SENT-TOKEN


##################################################################################################TEXT-ANNO
def getCITText(strText, SSETText):
    # len(SSETText) > 0
    # for sset in SSETText:
    #    assert strText[sset[1]: sset[2]] == sset[0]
    CITAnnoText = []
    for sset in SSETText:
        # BIOES
        strAnno, s, e, tag = sset
        CIT = [[c, s + idx, tag+ '-I']  for idx, c in enumerate(strAnno)]
        CIT[-1][2] = tag + '-E'
        CIT[ 0][2] = tag + '-B'
        if len(CIT) == 1:
            CIT[0][2] = tag + '-S' 
        CITAnnoText.extend(CIT)

    # print(strAnnoText)
    CITText = [[char, idx, 'O'] for idx, char in enumerate(strText)]
    for citAnno in CITAnnoText:
        c, idx, t = citAnno
        assert CITText[idx][0] == c
        CITText[idx] = citAnno
    return CITText

def getCITSents(strSents, CITText):
    lenLastSent = 0
    collapse    = 0 # don't need to move 
    CITSents = []
    for strSent in strSents:
        CITSent = []
        for sentTokenIdx, c in enumerate(strSent):
            # sentTokenIdx = txtTokenIdx - lenLastSent - collapse
            txtTokenIdx = sentTokenIdx + lenLastSent + collapse
            cT, _, tT = CITText[txtTokenIdx]
            while c != cT:
                collapse = collapse + 1
                txtTokenIdx = sentTokenIdx + lenLastSent + collapse
                cT, _, tT = CITText[txtTokenIdx]
            CITSent.append([c,sentTokenIdx, tT])
        lenLastSent = lenLastSent + len(strSent)
        CITSents.append(CITSent)
    # CITSents
    # Here we get CITSents  
    return CITSents
       
def getSSET_from_CIT(orig_seq, tag_seq, tag_seq_tagScheme = 'BIO', join_char = ''):
    # orig_seq is sentence without start or end
    # tag_seq may have start or end
    if tag_seq[0] == '</start>':
        tag_seq = tag_seq[1:-1]
        
    tagScheme = tag_seq_tagScheme
    if tagScheme == 'BIOES':
        tag_seq = [i.replace('-S', '-B').replace('-E', '-I') for i in tag_seq]
    elif tagScheme == 'BIOE':
        tag_seq = [i.replace('-E', '-I') for i in tag_seq]
    elif tagScheme == 'BIOS':
        tag_seq = [i.replace('-S', '-B') for i in tag_seq]
    elif tagScheme == 'BIO':
        pass
    else:
        print('The tagScheme', tagScheme, 'is not supported yet...')
    
    # use BIO tagScheme
    CIT = list(zip(orig_seq, range(len(orig_seq)), tag_seq))
    taggedCIT = [cit for cit in CIT if cit[2]!= 'O']
    
    startIdx = [idx for idx in range(len(taggedCIT)) if taggedCIT[idx][2][-2:] == '-B']
    startIdx.append(len(taggedCIT))

    entitiesList = []
    for i in range(len(startIdx)-1):
        entityAtom = taggedCIT[startIdx[i]: startIdx[i+1]]
        string = join_char.join([cit[0] for cit in entityAtom])
        start, end = entityAtom[0][1], entityAtom[-1][1] + 1
        tag = entityAtom[0][2].split('-')[0]
        entitiesList.append((string, start, end, tag))

    # if join_char == '*':
    #     a = [set([t.split('-')[0] for t in i[0].split('*')]) for i in entitiesList]
    #     for i in a:
    #         if len(i) > 1:
    #             pprint(list(zip(orig_seq, tag_seq)))
                
    return entitiesList
##################################################################################################TEXT-ANNO
