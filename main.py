from pprint import pprint
from nlptext.base import BasicObject

# ########### Wiki All ###########
CORPUSPath = 'dataset/WikiTotal/'
corpusFileIden = '.txt'
textType   = 'line'
Text2SentMethod  = 're'
Sent2TokenMethod = 'sep- '
TOKENLevel = 'word'
anno = False
annoKW = {}
MaxTextIdx = False
MaxTokenUnique = 447166


BasicObject.INIT(CORPUSPath, corpusFileIden, textType,
                 Text2SentMethod, Sent2TokenMethod, TOKENLevel,
                 anno, annoKW, MaxTextIdx, MaxTokenUnique)



# BOB = 'data/WikiTotal/word/Token1000004/Pyramid'
# LGU = 'data/WikiTotal/word/Token1000004/GrainUnique'
# BasicObject.INIT_FROM_PICKLE(BOB, LGU)

# print('BasicObject is Ready!')
##################################


CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,},
    'char':    {'use': True,'Max_Ngram': 1, 'end_grain': False},
    'basic':   {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'medical': {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'radical': {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
    'stroke':  {'use': True,'Max_Ngram': 5, 'end_grain': True},
    # CTX_DEP
    'pos':     {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
    # ANNO
    'annoR':   {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
    'annoE':   {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
}


CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,},
    'char':    {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'basic':   {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'medical': {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'radical': {'use': True,'Max_Ngram': 2, 'end_grain': False},
    'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
    'stroke':  {'use': True,'Max_Ngram': 5, 'end_grain': True},
    # CTX_DEP
    'pos':     {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
    # ANNO
    'annoR':   {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
    'annoE':   {'use': False,'Max_Ngram': 1, 'end_grain': False, 'tagScheme':   'BIO',},
}



BasicObject.BUILD_GRAIN_UNI_AND_LOOKUP(CHANNEL_SETTINGS_TEMPLATE)




