import os
import re
import numpy as np 
import pickle
import json
import pandas as pd
from datetime import datetime
from smart_open import smart_open
import itertools

from .anno import getSSET_from_CIT
from .infrastructure import strQ2B, fileReader, any2unicode

##################################################################################################CORPUS-FOLDER
# Important One 
def CorpusGroupsReader(CORPUSPath, iden = 'Dir'):
    # file is the priority
    if iden != 'Dir':
        corpusFiles = [i for i in os.listdir(CORPUSPath) if iden in i]
        if corpusFiles == []:
            results = [x for x in os.walk(CORPUSPath) if x[2]]
            results = {i[0]: i[2] for i in results}
            corpusFiles = sum([[pre + '/' + i for i in results[pre]] for pre in results], [])
            return {fd: ''      for fd in corpusFiles}, 'File'
        return {os.path.join(CORPUSPath, fd): ''      for fd in corpusFiles}, 'File'
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

##################################################################################################FOLDER-TEXT
anno = 'annofile4text'
anno_keywords = {
    'ANNOIden': '.NER',
    'anno_sep' : '\t', 
    'notZeroIndex' : 1, 
    'notRightOpen' : 0,
}

def annofile4text(strText, folderPath, origTextName, fileNames, ORIGIden, ANNOIden, anno_sep = '\t', notZeroIndex = 1, notRightOpen=0, **kwargs):
    SSETText = []
    name = origTextName.replace(ORIGIden, '')
    annoFileNames4Text = [f for f in fileNames if name in f and ANNOIden in f]
    
    if sum(['-sent' in f for f in annoFileNames4Text]) == 0:
        # each file stores text annotation
        strAnnoText = ''
        annoTextName = origTextName.replace(ORIGIden, ANNOIden)
        if annoTextName in fileNames:
            strAnnoText  = fileReader(os.path.join(folderPath, annoTextName))
        else:
            print('Error', os.path.join(folderPath, annoTextName))

        SSETText = [sset.split(anno_sep)[-4:] for sset in strAnnoText.split('\n') if anno_sep in sset]
        L = []
        for sset in SSETText:
            try:
                L.append([sset[0], int(sset[1]) - notZeroIndex, int(sset[2]) + notRightOpen, sset[3]])     
            except:
                pass
        SSETText = L
    return SSETText, annoTextName


anno = 'annofile4sent'
anno_keywords = {
    'ANNOIden': '.UMLSTag',
    'anno_sep' : '\t', 
    'notZeroIndex' : 0, 
    'notRightOpen' : 0,
}

def annofile4sent(strText, folderPath, origTextName, fileNames, ORIGIden, ANNOIden, anno_sep = '\t', notZeroIndex = 1, notRightOpen=0, **kwargs):
    name = origTextName.replace(ORIGIden, '')
    annoFileNames4Text = [f for f in fileNames if name in f and ANNOIden in f]
    # sentence based
    SSETText = []

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
        
    SSETText = [sset.split(anno_sep)[-4:] for sset in strAnnoText.split('\n') if anno_sep in sset]
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

    return SSETText, annoTextName

def textFileReader(folderPath, fileNames, anno = False, **kwargs):
    ORIGIden = '.txt'
    # ANNOIden = anno
    origTextName = None
    annoTextName = None

    origTextNames = [f for f in fileNames if ORIGIden in f]
    for origTextName in origTextNames:
        
        # here we process each text, and want to get strText, SSET, orig, anno
        SSETText = []
        path = os.path.join(folderPath, origTextName)
        with open(path, 'r', encoding = 'utf-8') as f:
            try:
                strText = strQ2B(f.read())
            except:
                strText = ''
                print('Error in file:', path)

        if anno == 'annofile4text':
            SSETText, annoTextName = annofile4text(strText, folderPath, origTextName, fileNames, ORIGIden, **kwargs)

        elif anno == 'annofile4sent':
            SSETText, annoTextName = annofile4sent(strText, folderPath, origTextName, fileNames, ORIGIden, **kwargs)

        yield strText, SSETText, origTextName, annoTextName



anno = 'anno_embed_in_text'
anno_keywords = {}

def anno_embed_in_text(line):
    SSETText = []
    strText = ''
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
    return strText, SSETText


# 其实/o 非/o 汉/o 非/o 唐/o ，/o 又是/o 什么/o 与/o 什么/o 呢/o ？/o 
anno = 'anno_embed_along_token' 
anno_keywords = {
    'sep_between_tokens': ' ',
    'sep_between_token_label': '/', 
}

def anno_embed_along_token(line, **anno_keywords):
    line = line.replace('\n', '')
    sep_between_tokens = anno_keywords['sep_between_tokens']
    sep_between_token_label = anno_keywords['sep_between_token_label']
    tokenlabel_seq = line.split(sep_between_tokens)
    tokenlabel_split_seq = [i.split(sep_between_token_label) for i in tokenlabel_seq if sep_between_token_label in i]
    tokenlabel_split_seq = [i for i in tokenlabel_split_seq if len(i) == 2 ]
    tokenlabel_split_seq = [i for i in tokenlabel_split_seq if i[0] != '' ]
    # print(tokenlabel_split_seq)
    strText = ''
    SSETText = []
    for tokensfrag_label in tokenlabel_split_seq:
        try:
            tokensfrag, label = tokensfrag_label
        except:
            print(tokensfrag_label)
            continue
        startidx = len(strText)
        strText = strText + tokensfrag
        endidx = len(strText)
        if label.lower() != 'o':
            SSETText.append([tokensfrag, startidx, endidx, label])

    return strText, SSETText


def textLineReader(folderPath, fileNames, anno = False, **anno_keywords):
    with smart_open(folderPath) as fin:
        for line in itertools.islice(fin, None):
            line = strQ2B(any2unicode(line))
            strText = line
            SSETText = []
            if anno == 'anno_embed_in_text':
                strText, SSETText = anno_embed_in_text(line)
            elif anno == 'anno_embed_along_token':
                strText, SSETText = anno_embed_along_token(line, **anno_keywords)
            # print(strText)
            # print(SSETText)
            yield strText, SSETText, None, None

anno = 'conll_block'
anno_keywords = {
    'anno_sep': '\t',
    'connector': '',
    'suffix': False,
    'change_tags': False, # If False, B, I, I, I, ...; If True: I, I, B, ...
}

def textBlockReader(folderPath, fileNames, anno = 'conll_block', change_tags = False, 
                    connector = '', suffix = True, anno_sep = ' ', **kwargs):
    assert anno == 'conll_block'
    if suffix:
        indicator, labelidx = -1, 0
    else:
        indicator, labelidx = 0, -1
    with smart_open(folderPath) as f:
        L = []
        for line in f:
            line = strQ2B(any2unicode(line))
            if line != '\n':
                L.append(strQ2B(line).replace('\n', '').split(anno_sep)) # TODO: maybe different seps
            else:
                strText = [ct[0] for ct in L]
                strText = connector.join(strText)
                
                # print(CIT)
                # this is used to deal with conll-2003 only.
                if change_tags:
                    TotalCIT = [[ct[0], idx, ct[-1]] for idx, ct in enumerate(L)]
                    # print(TotalCIT)
                    for idx, cit in enumerate(TotalCIT):
                        currentTag = cit[-1]
                        lastTag = TotalCIT[idx - 1][-1] if idx > 1 else 'O'
                        if 'I-' not in currentTag:
                            continue
                        # lastTag = TotalCIT[idx - 1][-1] if idx > 1 else 'null'
                        if lastTag == 'O' and 'I-' in currentTag:
                            TotalCIT[idx][-1] = cit[-1].replace('I-', 'B-')
                    L = TotalCIT

                CIT = [[ct[0], idx, ct[-1]] for idx, ct in enumerate(L) if ct[-1] != 'O']

                startIdxes = [idx for idx in range(len(CIT)) if CIT[idx][-1][indicator] in ['B', 'S']] + [len(CIT)]
                # print(startIdxes)
                SSETText = []
                for i in range(len(startIdxes)-1):
                    OneCIT = CIT[startIdxes[i]: startIdxes[i+1]]
                    string = connector.join(cit[0] for cit in OneCIT)
                    start, end = OneCIT[0][1], OneCIT[-1][1] + 1
                    tag = OneCIT[0][-1].split('-')[labelidx]
                    SSETText.append([string, start, end, tag])
                # SSETText = [[sset,sset,sset,sset[3].split('-')[0]] for sset in SSETText]
                # print('In Block Reader')
                # print(strText)
                # print(SSETText)
                yield strText, SSETText, None, None
                L = []
    
# no annotation currently
def textElementReader(folderPath, fileNames, anno = False, **kwargs):
    with open(folderPath, 'rb') as handle:
        L = pickle.load(handle)
        for strText in L:
            strText = strQ2B(strText)
            yield strText, None, None, None

# for json annotation
anno = 'json_annotation'
anno_keywords = {
    'strText': 'content',
    'labels': 'annotation',
}
def textJsonReader(folderPath, fileNames, anno = 'json_annotation', strText = 'content', labels = 'annotation', **kwargs):
    assert anno == 'json_annotation'
    with open(folderPath, 'r') as file:
        for line in file.readlines():
            dic = json.loads(line)
            strText = dic[strText]
            SSETText = []
            labels = dic[labels]
            for i in labels:
                sset = [i['points'][0]['text'], i['points'][0]['start'], i['points'][0]['end']+1, ''.join(i['label'])]
                SSETText.append(sset)
            yield strText, SSETText, None, None

anno = 'csv_annotation'
anno_keywords = {
    'ANNOIden': '.xlsx',
}
def textCSVReader(folderPath, fileNames, anno = 'csv_annotation',  ANNOIden = '.csv', **kwargs):
    assert anno == 'csv_annotation'
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
    'csv': textCSVReader
}



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

    ###################### Remove some weird chars #######################
    text = re.sub('\xa0', '', text)
    text = re.sub('\\n', '', text)

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
    sents = [sent.strip() for sent in text]
    # text = [sent.replace('\\n', '') for sent in text]

    return sents # [sent for sent in text if len(sent)>=2]

def lineCutText2Sent(text):
    return text.split('\n')


def tokenText2Sent(text):
    # how to understand this?
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
        # sents = [text.replace('\\n', '').replace('\n', '').replace(' ', '').replace('\t', '').replace('\xa0', '')]
        sents = [text.replace('\\n', '').replace('\n', '').replace('\t', '')]

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
    sents = [sent.replace('\n', '').replace('\\n', '') for sent in sents]
    sents = [sent for sent in sents if len(sent) > 0]
    # print(sents)
    return sents

##################################################################################################SENT-TOKEN

def clean_token(token):
    # token contains no spaces.
    # what is token is a white splace itself?
    token = token.replace(' ', '').replace('\n', '').replace('\t', '')

    # punctuation must be seperated away from token.
    # if token contains special marks, seperate this token to two or more token.

    # last check, token should not be '', if so, pass it.
    # remove some special tokens.
    return [token]


def segSent2Tokens(sent, seg_method = 'iter', tokenLevel = 'char', Channel_Dep_Methods = {}):
    # print(Channel_Dep_Methods)
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
        # print(Channel_Dep_Methods[seg_method])
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
        raise ValueError('No good segmentaion method...')


    # remove ' ' inside strToken
    # in this case, ' ' will not be a token, we will solve this problem in the future.
    # how about, in Chinese, the ' ' between English words.
    # TODO

    # in general, we only use it for Chinese Char, and won't for Chinese Word
    # actually, we can also use it for English Word. but English char is too verbose.
    # print(strTokens)
    for ch, hyper_method in Channel_Dep_Methods.items():
        if ch == seg_method:
            continue
        # these hyper fields are all fields with larger granularity, and changing will be taken care inside the method.
        ch_grain_sent, _ = hyper_method(strTokens, tokenLevel = tokenLevel, tagScheme = 'BIOES') 
        # this assert should be inside the hyper_field_method
        # assert len(ch_grain_sent) == len(final_tokens)
        hyper_info[ch] = ch_grain_sent
        assert len(strTokens) == len(ch_grain_sent)

    return strTokens, hyper_info

#########################################################################33
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
