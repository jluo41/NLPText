from pprint import pprint
from nlptext.base import BasicObject


########### Wiki ###########
# CORPUSPath = 'corpus/WikiEnglish/'

# Corpus2GroupMethod = '.txt'

# Group2TextMethod   = 'line'

# Text2SentMethod  = 'whole' # this caused the problem

# Sent2TokenMethod =  'pos_en' # ' '
# TOKENLevel = 'word'
# min_token_freq = 10

# use_hyper =  ['pos_en'] # ' '

# anno = False
# anno_keywords = {}

# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = False, anno_keywords = anno_keywords)

# from nlptext.base import BasicObject

# ########### Wiki ###########
# CORPUSPath = 'corpus/WikiChinese/'

# Corpus2GroupMethod = '.txt'

# Group2TextMethod   = 'line'

# Text2SentMethod  = 'whole'

# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'
# min_token_freq = 1

# use_hyper = {'pos'}

# anno = False
# anno_keywords = {}


# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = False, anno_keywords = anno_keywords)



from pprint import pprint
from nlptext.base import BasicObject

CORPUSPath = 'corpus/LuohuNER750Neat/'

Corpus2GroupMethod = 'Dir' # TODO

Group2TextMethod   = 'file'

Text2SentMethod  = 're'

Sent2TokenMethod = 'iter'
TOKENLevel = 'char'

min_token_freq = 1

use_hyper = {'pos', 'medpos'}

anno = 'annofile4text'
anno_keywords = {
    'ANNOIden': '.NER',
    'anno_sep' : '\t', 
    'notZeroIndex' : 0, # if 0, indexed from zero
    'notRightOpen' : 0, # if 0, Right is Open, is 1 not Open
}


BasicObject.INIT(CORPUSPath, 
                 Corpus2GroupMethod, 
                 Group2TextMethod, 
                 Text2SentMethod, 
                 Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
                 use_hyper = use_hyper, 
                 anno = anno, anno_keywords = anno_keywords)

