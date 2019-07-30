import pickle

############### PART SpecialTokens

PAD   = '</pad>'
START = '</start>'
END   = '</end>'
UNK   = '</unk>'
specialTokens     = [ PAD, START, END, UNK]
specialTokensDict = {PAD: 0, START: 1, END: 2, UNK : 3, }
START_ID = specialTokensDict[START]
END_ID   = specialTokensDict[END]
UNK_ID   = specialTokensDict[UNK]


############### PART CODE
def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)

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
############### PART Input and Output 

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

############### PART BIOES Labels and Transformation.
# labels
# input: ['不确定', '修饰',  '检查', '疾病', '症状'] 
# tags
# ouput: ['O', '不确定-B', '不确定-I', '修饰-B', '修饰-I', '检查-B', '检查-I', '疾病-B', '疾病-I', '症状-B', '症状-I']
def getTagDict(labels, tagScheme = 'BIOES'):
    L = []
    suffices = ['-B', '-I']
    if 'O' in tagScheme:
        pref = ['O'] # remove speicial token
    else:
        pref = []
    if 'E' in tagScheme:
        suffices = suffices + ['-E']
    if 'S' in tagScheme:
        suffices = suffices + ['-S']
    tags = [i for i in labels if i !='O']
    for tag in tags:
        L.extend([tag+suff for suff in suffices])
    L.sort()
    L = pref + L
    return L

def trans_bioesTag(channel, bioesTag, tagScheme):

    if bioesTag in specialTokens:
        return bioesTag

    if 'S' not in tagScheme and 'E' not in tagScheme:
        i = bioesTag.replace('-S', '-B').replace('-E', '-I')
    elif 'S' not in tagScheme:
        i = bioesTag.replace('-S', '-B')
    elif 'E' not in tagScheme:
        i = bioesTag.replace('-E', '-I')
    else:
        i = bioesTag

    if channel == 'annoR':
        return i.split('-')[-1] 
    else:
        return i
        
############### PART Char To Word
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

############### PART OTHERS

def extractSET(tag_seq, exist_SE = False):
    '''
        SET: start, end, tag
        tag_seq: the hyper field sequence for this sentence
    '''
    if exist_SE:
        tag_seq = tag_seq[1:-1]

    IT = list(zip(range(len(tag_seq)), tag_seq))
    taggedIT = [it for it in IT if it[1]!= 'O']
    
    startIdx = [idx for idx in range(len(taggedIT)) if taggedIT[idx][1][-2:] == '-B' or taggedIT[idx][1][-2:] == '-S']
    startIdx.append(len(taggedIT))

    entitiesList = []
    for i in range(len(startIdx)-1):
        entityAtom = taggedIT[startIdx[i]: startIdx[i+1]]
        # string = ''.join([cit[0] for cit in entityAtom])
        start, end = entityAtom[0][0], entityAtom[-1][0] + 1
        tag = entityAtom[0][1].split('-')[0]
        entitiesList.append((start, end, tag))
    return entitiesList


# def extractEmbedPath2Info(embed_path, channel = None):
#     if not os.path.isfile(embed_path):
#         return None
#     path_comp = embed_path.split('/')
#     TokenNum_Dir = '/'.join(['channel'] + path_comp[1:4])
#     # print(path_comp[-1].split('.')[0].split('_')[-1].lower())
#     if channel:
#         assert channel == path_comp[-1].split('.')[0].split('_')[-1].lower()
#     else:
#         channel = path_comp[-1].split('.')[0].split('_')[-1].lower()

#     channel_abbr = CHANNEL_ABBR[channel]
#     channel_name_abbr = [i for i in path_comp[4].split('_') if channel_abbr in i][0]

#     MN_E = channel_name_abbr[len(channel_abbr): ]

#     if MN_E == '':
#          channel_name = channel
#     elif channel in CONTEXT_IND_CHANNELS:
#         channel_name = channel + MN_E
#     else:
#         if int(MN_E) == 5:
#             channel_name = channel + '-bioes'
#         elif int(MN_E) == 4:
#             channel_name = channel + '-bioe'
#         else:
#             print('Fail to Extract information for embed:', embed_path, channel, MN_E)

#     return TokenNum_Dir, channel_name


