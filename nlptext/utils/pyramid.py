import os
import re
import numpy as np 
import pickle
import json
import pandas as pd
from datetime import datetime
from smart_open import smart_open
import itertools

from .infrastructure import strQ2B, fileReader, any2unicode

##################################################################################################CORPUS-FOLDER
# Important One 
def CorpusGroupsReader(CORPUSPath, iden = None):
    # file is the priority
    if iden:
        corpusFiles = [i for i in os.listdir(CORPUSPath) if iden in i]
        return {os.path.join(CORPUSPath, fd): '' for fd in corpusFiles}, 'File'
    else:
        results = [x for x in os.walk(CORPUSPath) if x[2]]
        return {i[0]: i[2] for i in results}, 'Dir'

        
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

                ### something different
                txtCharIdx = 0
                collapse   = 0
                
                for ssetIdx, sset in enumerate(SSETText):
                    string, s, e, t = sset
                    string = string.replace('@', ' ') # TODO
                    lenString = len(string)
                    while string != strText[txtCharIdx: txtCharIdx + lenString]:
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
    with smart_open(folderPath) as fin:
        for line in itertools.islice(fin, None):
            # print(line)
            line = strQ2B(any2unicode(line))
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
    

def textElementReader(folderPath, fileNames, anno = False, **kwargs):
    with open(folderPath, 'rb') as handle:
        L = pickle.load(handle)
        for strText in L:
            strText = strQ2B(strText)
            yield strText, None, None, None

def textJsonReader(folderPath, fileNames, anno = '.json', **kwargs):
    with open(folderPath, 'rb') as file:
        for line in file.readlines():
            dic = json.loads(line)
            strText = dic['content']
            SSETText = []
            tokens = dic['annotation']
            for i in tokens:
                token = [i['points'][0]['text'], i['points'][0]['start'], i['points'][0]['end']+1, ''.join(i['label'])]
                SSETText.append(token)
            yield strText, SSETText, None, None

def textWordReader(folderPath, fileNames, anno = '.xlsx', **kwargs):
    if anno == '.xlsx' or anno == '.csv':
        if anno == '.xlsx': 
            data = pd.read_excel(folderPath)
        if anno == '.csv':
            data = pd.read_csv(folderPath)
        strText = []
        SSETText = []
        flag = data['Sentence #'].isnull()
        for i in range(len(data['Word'])):
            if flag[i] == False and i != 0:
                strText.append(' ')
                SSETText.append([' ', -1, 'T2S'])
            word = data['Word'][i]
            label = data['Tag'][i]
            tag = [word, i, label]
            SSETText.append(tag)
            strText.append(word)
    else:       
        with open (folderPath, 'r') as file:
            lines = file.readlines()
            strText = []
            SSETText = []
            idx = 0
            for line in lines:
                if line == '\n':
                    strText.append(' ')
                    SSETText.append([' ', -1, 'T2S'])
                else:
                    line = line.split('\n')[0]
                    temp = line.split('\t')
                    strText.append(temp[0])
                    SSETText.append([temp[0], idx, temp[-1]])
                    idx += 1
    yield strText, SSETText, None, None



FolderTextsReaders = {
    'file': textFileReader,
    'line': textLineReader,
    'block':textBlockReader,
    'element': textElementReader,
    'json' : textJsonReader,
    'word': textWordReader
}
##################################################################################################FOLDER-TEXT


##################################################################################################TEXT-SENT
def reChnCutText2Sent(text, useSep = False):
    # first, eventually, the space will be used to seperate the tokens
    # in this case, if there are spaces in the original text and is not the delimiter
    # they will be removed away.

    # the formats of the text
    # 1. text is a Chinese string, in this case there are no spaces
    #    '汉字表示的数学一词大约产生于中国宋元时期。'
    #    in this case, if there is a space, regard it as an original seperator in the future.
    #    the annoying case if the sentence together with numbers and English letters.
    # 2. text is a Chinese string seperated by space
    #    '汉字 表示 的 数 学 一 词 大约 产生 于 中国 宋元 时期 。'
    # 3. tokens in the sentence contain hyper information, but still separated by space.
    #    '汉字_nz 表示_v 的_u 数_n 学_n 一_m 词_n 大约_d 产生_v 于_p 中国_ns 宋元_t 时期_n 。_w'

    # I didn't find good way to deal with the English. The common way to use is whole. 
    # we will use whole line as a sentence for English file.
    ###################### Remove some weird chars #######################
    text = re.sub('\xa0', '', text)

    # keep the spaces between two English letters.
    # text = re.sub(r'(?<=[A-Za-z])\s+(?=[|A-Za-z])', 'ⴷ',  text)
    
    ###################### Convert the S+ spaces to '〰' #################
    # text = re.sub(' {2}', '〰', text ).strip()
    # if useSep == ' ':
    #     # if using space to sep the words
    #     text = text.replace('\t','').replace('〰', ' ')
    # elif useSep == '\t':
    #     # if using tab to sep the words, removing all spaces
    #     text = text.replace(' ','').replace('〰', '')
    # else:
    #     # if there is no sep char for Chinese, remove single space, and then convert space+ to single space
    #     text = text.replace('\t','').replace(' ', '',).replace('〰', ' ')

    text = text.replace('\t','')# .replace(' ', '',).replace('〰', ' ')
    # convert the spaces between English letters to single spaces
    # text = text.replace('ⴷ', ' ')
    
    text = re.sub('([。！;；])([^”])',  r"\1\n\2",text) 
    text = re.sub('(\.{6})([^”])',     r"\1\n\2",text) 
    text = re.sub('(\…{2})([^”])',     r"\1\n\2",text)
    
    # The \n within " " is not considered
    text = '"'.join( [ x if i % 2 == 0 else x.replace('\n', '') for i, x in enumerate(text.split('"'))] )
    
    text = re.sub( '\n+', '\n', text ).strip() # replace '\n+' to '\n'
    text = text.replace('\\n', '\n')
    text = text.split("\n")
    text = [sent.strip() for sent in text]
    # text = [sent.replace(' ', '').replace('\\n', '') for sent in text]

    return text # [sent for sent in text if len(sent)>=2]

def lineCutText2Sent(text):
    return text.split('\n')


def tokenText2Sent(text):
    sents = []
    idx = 0
    for i in range(len(text)):
        if text[i] == ' ':
            sents.append(text[idx:i])
            idx =  i + 1
    sents.append(text[idx:])
    return sents


def preprocess_text(text):
    # for example: 
    # 1 removing the no-chinese character
    # 2 removing the empty parenthesis
    # 3 replacing the weird punctunations.
    return text

def segText2Sents(text, method = 'whole', **kwargs):
    # the formats of the text
    # 1. text is a Chinese string, in this case there are no spaces
    #    '汉字表示的数学一词大约产生于中国宋元时期。'
    #    in this case, if there is a space, regard it as an original seperator in the future.
    #    the annoying case if the sentence together with numbers and English letters.
    #    to deal with the spaces in this format, we simply keep them.
    # 2. text is a Chinese string seperated by space
    #    '汉字 表示 的 数 学 一 词 大约 产生 于 中国 宋元 时期 。'
    # 3. tokens in the sentence contain hyper information, but still separated by space.
    #    '汉字_nz 表示_v 的_u 数_n 学_n 一_m 词_n 大约_d 产生_v 于_p 中国_ns 宋元_t 时期_n 。_w'
    # 2. and 3. are basically the same. but we wouldn't consider it.

    # for English
    # 1. the text is naturally seperated by the spaces. we seg the text to sents by its only '\n' only.

    # TODO: here should be some checks for texts.

    if os.path.isfile(text):
        text = fileReader(text)
            
    # preprocessing the text
    text = preprocess_text(text)

    if method == 'whole':
        # this is commonly used for wikipedia data
        # for English corpus, we use this only.
        sents = [text]

    elif method == 're':
        # re still need more method to consider the final results.
        sents = reChnCutText2Sent(text, **kwargs)

    elif method == 'line':
        sents = lineCutText2Sent(text)

    elif method == 'token':
        # what's the meaning?
        sents = tokenText2Sent(text)

    else:
        sents = method(text, **kwargs)

    # postprocessing
    # after this, there is no '\n' in sents
    sents = [sent.replace('\n', '') for sent in sents]

    return sents


##################################################################################################SENT-TOKEN

def clean_token(token):
    # token contains no spaces.
    token = token.replace(' ', '').replace('\n', '').replace('\t', '')

    # punctuation must be seperated away from token.
    # if token contains special marks, seperate this token to two or more token.

    # last check, token should not be '', if so, pass it.
    # remove some special tokens.
    return [token]


def segSent2Tokens(sent, seg_method = 'iter', tokenLevel = 'char', Channel_Dep_Methods = {}):

    hyper_info = {}

    # seg_method is: 1 iter; 2 ' ' or '\t'; 3 'pos' or other channel that are in Channel_Dep_Methods
    if seg_method == 'iter' and tokenLevel == 'char':
        # sent can be
        # 1. '产生于中国宋元时期。'           --->   '产 生 于 中 国 宋 元 时 期 。' 
        # 2. '产生 于 中国 宋元 时期 。'      --->   '产 生 于 中 国 宋 元 时 期 。' 
        # 3. '产 生 于 中 国 宋 元 时 期 。'  --->   '产 生 于 中 国 宋 元 时 期 。' 
        strTokens = [i for i in sent if i != ' ']
    
    elif seg_method in [' ', '\t']:
        # sent can be
        # 1. '产生 于 中国 宋元 时期 。'      --->   '产生 于 中国 宋元 时期 。' 
        # 2. '产 生 于 中 国 宋 元 时 期 。'  --->   '产 生 于 中 国 宋 元 时 期 。'
        # both word and char level is OK.
        strTokens =  [i.replace(' ', '') for i in sent.split(seg_method) if i != ' ']

        # check and clean the token
        final_tokens = []
        for idx, token in enumerate(strTokens):
            new_tokens = clean_token(token)
            final_tokens.extend(new_tokens)
        strTokens = final_tokens

    elif seg_method in Channel_Dep_Methods:
        ch_grain_sent, strTokens = Channel_Dep_Methods[seg_method](sent, tokenLevel = tokenLevel, tagScheme = 'BIOES')
        # check and clean the token
        final_tokens = []
        new_grain_sent = []
        for idx, token in enumerate(strTokens):
            new_tokens = clean_token(token)
            final_tokens.extend(new_tokens)
            if len(new_tokens) == 1:
                new_grain_sent.append(ch_grain_sent[idx])
            else:
                new_grain_sent.extend([ch_grain_sent[idx]] * len(new_tokens))

        strTokens = final_tokens
        ch_grain_sent = new_grain_sent
        hyper_info[seg_method] = ch_grain_sent

    else:
        raise('No good segmentaion method...')

    # in general, we only use it for Chinese Char, and won't for Chinese Word
    # actually, we can also use it for English Word. but English char is too verbose.
    for ch, hyper_method in Channel_Dep_Methods.items():
        if ch == seg_method:
            continue
        # these hyper fields are all fields with larger granularity, and changing will be taken care inside the method.
        ch_grain_sent, _ = hyper_method(strTokens, tokenLevel = tokenLevel, tagScheme = 'BIOES') 
        # this assert should be inside the hyper_field_method
        # assert len(ch_grain_sent) == len(final_tokens)
        hyper_info[ch] = ch_grain_sent

    return strTokens, hyper_info


##################################################################################################TEXT-ANNO
def getCITText(strText, SSETText, TOKENLevel='char'):
    len(SSETText) > 0 
    if TOKENLevel == 'char':
        for sset in SSETText:
            try:
                assert strText[sset[1]: sset[2]] == sset[0]
            except:
                print('strText:', strText[sset[1] : sset[2]])
                print('SSETText:', sset[0])
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

    elif TOKENLevel == 'word':
        CITText = []
        for idx, sset in enumerate(SSETText):
            try:
                assert sset[0] == strText[idx]
            except:
                print(strText)[idx]
                print(sset[0])

            CITText.append(sset)
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
            while c != cT and c != ' ':
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


def get_line_with_position(path, start_position):
    with open(path, 'r', encoding = 'utf-8') as f:
        f.seek(start_position)
        line = f.readline()
    return line[:-1]

def get_lines_with_position(path, start_position, num_lines):
    lines = ''
    i = 0
    with open(path, 'r', encoding = 'utf-8') as f:
        f.seek(start_position)
        for line in f.readlines():
            lines = lines + line 
            i = i + 1
            if i >= num_lines:
                break
    return lines[:-1]

def read_file_chunk_string(path, startbyteidx, endbyteidx):
    with open(path, 'r',  encoding = 'utf-8') as fin:
        fin.seek(startbyteidx)
        data = ''
        while True:
            data = data + fin.readline()
            # fin.tell() equal to endbyteidx is very rigid
            # we can use >= instead.
            if fin.tell() >= endbyteidx:
                break
    return data[:-1]
