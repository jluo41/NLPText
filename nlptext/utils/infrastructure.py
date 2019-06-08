import pickle



PAD   = '</pad>'
START = '</start>'
END   = '</end>'
UNK   = '</unk>'
specialTokens     = [ PAD, START, END, UNK]
specialTokensDict = {PAD: 0, START: 1, END: 2, UNK : 3, }

START_ID = specialTokensDict[START]
END_ID   = specialTokensDict[END]
UNK_ID   = specialTokensDict[UNK]


def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def fileReader(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        text = f.read()
    return strQ2B(text)


def writeGrainList2File(channel_name_path, ListGrainUnique):
    with open(channel_name_path, 'w', encoding = 'utf-8') as f:
        for i in ListGrainUnique:
            i = '\\n' if i == '\n' else i
            f.write(i + '\n')

def readFile2GrainList(channel_name_path):
    ListGrainUnique = []
    with open(channel_name_path, 'r', encoding = 'utf-8') as f:
        for gr in f.readlines():
            gr = '\n' if  '\\n' in gr[:-1] else gr[:-1]
            ListGrainUnique.append(gr)
    return ListGrainUnique

def readPickleFile2GrainUnique(channel_name_path):
    with open(channel_name_path, 'rb') as handle:
        v = pickle.load(handle)
    return v  #(LGU, DGU)


def modify_wordBoundary_with_hyperBoundary(pos_sent, anno_sent):
    # return the new word_boundary with BIO tagScheme.
    pos_sent = [i.replace('-S', '-B').replace('-E', '-I') for i in pos_sent]
    
    for idx, tag in enumerate(anno_sent):
        if ('-B' == anno_sent[idx][-2:]  or '-S' == anno_sent[idx][-2:]):
            if pos_sent[idx][-2:] != '-B':
                # pprint([anno_sent[idx], pos_sent[idx]])
                pos_sent[idx] = pos_sent[idx].split('-')[0] + '-B'

        if ('-E' == anno_sent[idx][-2:]  or '-S' == anno_sent[idx][-2:]) and idx + 1 < len(pos_sent):
            if pos_sent[idx+1][-2:] != '-B':
                # pprint([anno_sent[idx], pos_sent[idx], pos_sent[idx+1]])
                # pprint(list(zip(anno_sent, pos_sent)))
                pos_sent[idx+1] = pos_sent[idx+1].split('-')[0] + '-B'
                
    return pos_sent



def trans_charLabels_to_wordLabels(string, join_char = '*'):
    labels = string.split('*')
    if ['O'] == list(set(labels)):
        return 'O'
    else:
        labels = [i.split('-') for i in labels]
        tag = list(set([i[0] for i in labels]))
        if len(tag) > 1:
            return tag, loc_indicator
        else:
            tag = tag[0]
            loc_indicator = [i[1] for i in labels]
            # print(string)
            if 'B'== loc_indicator[0] and 'E' == loc_indicator[-1]:
                # print(loc_indicator, '-->', 'S')
                return tag + '-S'
            elif ['S']  == loc_indicator:
                # print(loc_indicator, '-->', 'S')
                return tag + '-S'
            elif 'B' == loc_indicator[0]:
                # print(loc_indicator, '-->', 'B')
                return tag + '-B'
            elif 'E' == loc_indicator[-1]:
                # print(loc_indicator, '-->', 'E')
                return tag + '-E'
            elif ['I'] == list(set(loc_indicator)):
                # print(loc_indicator, '-->', 'I')
                return tag + '-I'
            else:
                # print('Jie -->', tag, loc_indicator)
                return tag, loc_indicator



########### NER ###########

CORPUSPath = 'corpus/ner/'
corpusFileIden = None
textType   = 'file'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = '.Entity'
annoKW = {
    'sep': '\t',
    'notZeroIndex': 1,
}


########### MedPOS ###########

CORPUSPath = 'corpus/medpos/'
textType   = 'file'
corpusFileIden = None
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = '.UMLSTag'
annoKW = {
    'sep': '\t',
    'notZeroIndex': 0,
}

########### Weibo Test ###########
CORPUSPath = 'corpus/weibotest/'
corpusFileIden = None
textType   = 'file'
Text2SentMethod  = 're'
Sent2TokenMethod = 'sep-\t'
TOKENLevel = 'word'
anno = False
annoKW = {}


########### Wiki ###########
CORPUSPath = 'corpus/wiki/'
corpusFileIden = '.txt'

textType   = 'line'

Text2SentMethod  = 're'
Sent2TokenMethod = 'sep- '
TOKENLevel = 'word'

anno = False
annoKW = {}



########### ResumeNER ###########
CORPUSPath = 'corpus/ResumeNER/'
corpusFileIden = '.bmes'
textType   = 'block'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = 'embed' # TODO
annoKW = {}

