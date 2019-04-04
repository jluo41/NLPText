import pickle



PAD   = '</pad>'
START = '</start>'
END   = '</end>'
UNK   = '</unk>'
specialTokens     = [ PAD, START, END, UNK]
specialTokensDict = {PAD: 0, START: 1, END: 2, UNK : 3, }
UNK_ID = specialTokensDict[UNK]


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

