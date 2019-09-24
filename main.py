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

CORPUSPath = 'corpus/MSRA/'

Corpus2GroupMethod = '.txt'

Group2TextMethod   = 'line'

Text2SentMethod  = 'whole'

Sent2TokenMethod = 'iter'
TOKENLevel = 'char'

min_token_freq = 1

use_hyper = []

# 其实/o 非/o 汉/o 非/o 唐/o ，/o 又是/o 什么/o 与/o 什么/o 呢/o ？/o 
anno = 'anno_embed_along_token' 
anno_keywords = {
    'sep_between_tokens': ' ',
    'sep_between_token_label': '/', 
}

BasicObject.INIT(CORPUSPath, 
                 Corpus2GroupMethod, 
                 Group2TextMethod, 
                 Text2SentMethod, 
                 Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
                 use_hyper = use_hyper, 
                 anno = anno, anno_keywords = anno_keywords)
