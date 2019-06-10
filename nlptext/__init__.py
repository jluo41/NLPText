# for f in *_tag.txt; do mv -- "$f" "${f//_tag.txt/.NER}"; done

# ########### BOSON ###########
# CORPUSPath = 'corpus/boson/'
# corpusFileIden = '.txt'
# textType   = 'line'
# Text2SentMethod  = 're'
# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'
# anno = 'embed'
# annoKW = {}


# ########### NER ###########
# CORPUSPath = 'corpus/ner/'
# corpusFileIden = None
# textType   = 'file'
# Text2SentMethod  = 're'
# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'
# anno = '.Entity'
# annoKW = {
#     'sep': '\t',
#     'notZeroIndex': 1,
# }


# ########### MedPOS ###########
# CORPUSPath = 'corpus/medpos/'
# textType   = 'file'
# corpusFileIden = None
# Text2SentMethod  = 're'
# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'
# anno = '.UMLSTag'
# annoKW = {
#     'sep': '\t',
#     'notZeroIndex': 0,
# }

# ########### Weibo Test ###########
# CORPUSPath = 'corpus/weibotest/'
# corpusFileIden = None
# textType   = 'file'
# Text2SentMethod  = 're'
# Sent2TokenMethod = 'sep-\t'
# TOKENLevel = 'word'
# anno = False
# annoKW = {}


# ########### Wiki ###########
# CORPUSPath = 'corpus/wiki/'
# corpusFileIden = '.txt'

# textType   = 'line'

# Text2SentMethod  = 're'
# Sent2TokenMethod = 'sep- '
# TOKENLevel = 'word'

# anno = False
# annoKW = {}



# ########### ResumeNER ###########
# CORPUSPath = 'corpus/ResumeNER/'
# corpusFileIden = '.bmes'
# textType   = 'block'
# Text2SentMethod  = 're'
# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'
# anno = 'embed' # TODO
# annoKW = {}

