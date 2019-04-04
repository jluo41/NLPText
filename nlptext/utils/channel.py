import re
import pickle
import string
from jieba import posseg

from .infrastructure import specialTokens

def basicGrainChar(char, end_grain = False):
    '''char level only!'''
    punStr = string.punctuation + '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    engReg = r'[A-Za-z]{1}'
    if '%' in char or '%' in char:
        info =  'PERC'
    elif re.match(r'[0-9]{1}', char):
        info = "NUM"
    elif char in punStr:
        info = "PUNC"
    elif char >= '\u4e00' and char <= '\u9fff':
        info = "CHN"
    elif re.match(engReg, char):
        info = 'ENG'
    #elif word in string.whitespace:
        #return 'SPA'
    elif char == '@':
        info = 'SPA'
    else:
        info = 'OTHER'
    info = [info]
    if end_grain:
        info = info + ['b0']
    return info

def medicalGrainChar(word, end_grain = False):
    '''char level only!'''
    units = 'kBq kbq mg Mg UG Ug ug MG ml ML Ml GM iu IU u U g G l L cm CM mm s S T % % mol mml mmol MMOL HP hp mmHg umol ng'.split(' ')
    chn_units = '毫升 毫克 单位 升 克 第 粒 颗粒 支 件 散 丸 瓶 袋 板 盒 合 包 贴 张 泡 国际单位 万 特充 个 分 次'.split(' ')
    med_units = 'qd bid tid qid qh q2h q4h q6h qn qod biw hs am pm St DC prn sos ac pc gtt IM IV po iH'.split(' ')
    all_units = units + chn_units + med_units

    site_units = '上 下 左 右 间 片 部 内 外 前 侧 后'.split(' ')
    sym_units = '大 小 增 减 多 少 升 降 高 低 宽 厚 粗 两 双 延 长 短 疼 痛 终 炎 咳'.split(' ')
    part_units = '脑 心 肝 脾 肺 肾 胸 脏 口 腹 胆 眼 耳 鼻 颈 手 足 脚 指 壁 膜 管 窦 室 管 髋 头 骨 膝 肘 肢 腰 背 脊 腿 茎 囊 精 唇 咽'.split(' ')
    break_units = '呈 示 见 伴 的 因'.split(' ')
    more_units = '较 稍 约 频 偶 偏'.split(' ')
    non_units = '无 不 非 未 否'.split(' ')
    tr_units = '服 予 行'.split(' ')

    if word in units:
        info =  'UNIT'
    elif word in chn_units:
        info =  'CHN_UNIT'
    elif word in med_units:
        info =  'MED_UNIT'
    elif word in site_units:
        info =  'SITE_UNIT'
    elif word in sym_units:
        info =  'SYM_UNIT'
    elif word in part_units:
        info =  'PART_UNIT'
    elif word in break_units:
        info =  'BREAK_UNIT'
    elif word in more_units:
        info =  'more_UNIT'
    elif word in non_units:
        info =  'NON_UNIT'
    elif word in tr_units:
        info =  'TR_UNIT'
    else:
        info =  'OTHER'
        
    # return [info]
    info = [info]
    if end_grain:
        info = info + ['m0']
    return info
    
with open('nlptext/sources/CharSubComp.p', 'rb') as handle:
    CharSubCompInfos = pickle.load(handle)

def subcompGrainChar(char, end_grain = False):
    '''char level only!'''
    if char in CharSubCompInfos:
        info = CharSubCompInfos[char]
        if info:
            info = ['c' + i for i in info ]
        else:
            info = ['c' + char] 
    else:
        info = [char]
        
    if end_grain:
        info = info + ['c0']
    return info


with open('nlptext/sources/CharStroke.p', 'rb') as handle:
    CharStrokeInfos = pickle.load(handle)
    
    
def strokeGrainChar(char, end_grain = False):
    '''char level only!'''
    if char in CharStrokeInfos:
        info = CharStrokeInfos[char]
        if info:
            info = ['s' + i for i in info ]
        else:
            info = ['s' + char] 
    else:
        info = [char]
        
    if end_grain:
        info = info + ['s0']
    return info

with open('nlptext/sources/CharRadical.p', 'rb') as handle:
    CharRadicalInfos = pickle.load(handle)

def radicalGrainChar(char, end_grain = False):
    ''''char level only!'''
    if char in CharRadicalInfos:
        info = CharRadicalInfos[char]
        if info:
            info = [info] # here Radical Data is in str not list
        else:
            info = [char]
    else:
        info = [char]
        
    if end_grain:
        info = info + ['r0']
    return info


def charGrainChar(char, end_grain = False):
    '''char level only!'''
    info = [char]
    if end_grain:
        info = info + ['ch0']
    return info



################## FOR THE CONTEXT-DEPENDENT CHANNELS ################
posTag = ['a', 'ad', 'ag', 'an', 'b', 'c', 'd', 'df', 'dg', 'e', 'eng', 
          'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'mg', 'mq', 'n', 'ng', 'nr', 'nrfg', 
          'nrt', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'rg', 'rr','rz', 's', 't', 
          'tg', 'u', 'ud', 'ug', 'uj', 'ul', 
          'uv', 'uz', 'v', 'vd', 'vg', 'vi', 'vn', 'vq', 'x', 'y', 'yg', 'z', 'zg']

def POSGrainSent(sent, tokenLevel = 'word', useStartEnd = False, tagScheme = 'BIOES'):
    '''
        Here only for Atom is Char Based
        This method should be enriched
        sent: List of Token(String), with or without Start or End
    '''
    if useStartEnd:
        sent = sent[1:-1]
    segs = list(posseg.cut(''.join(sent)))
    
    GrainSent = []
    
    if tokenLevel == 'word':
        for i in range(len(segs)):
            pair  = segs[i]
            label = pair.flag
            GrainSent.append([label])
            # print(pair.word)
        if useStartEnd:
            return [[START]] + GrainSent + [[END]]
        return GrainSent
    
    elif tokenLevel == 'char':
        for i in range(len(segs)):
            pair  = segs[i]
            leng  = len(pair.word)
            label = pair.flag
            labels= [label + '-I' ]*leng
            labels[0] = label + '-B'
            if 'E' in tagScheme and leng >= 2:
                labels[-1] = label + '-E'
            if 'S' in tagScheme and leng == 1:
                labels[0] = label + '-S'
            GrainSent.extend([[i] for i in labels])
        if useStartEnd:
            return [[START]] + GrainSent + [[END]]
        return GrainSent
##################################################################

################################################################################################
CONTEXT_IND_CHANNELS    = ['basic', 'medical', 'radical', 'token', 'char', 'subcomp', 'stroke']
CONTEXT_DEP_CHANNELS    = ['pos']
ANNO_CHANNELS           = ['annoR', 'annoE']

CONTEXT_IND_CHANNELS_AB = ['b', 'm', 'r',  'T', 'C', 'c', 's']
CONTEXT_DEP_CHANNELS_AB = ['P']
ANNO_CHANNELS_AB        = ['R', 'E']

CHANNEL_ABBR = dict(zip(CONTEXT_IND_CHANNELS + CONTEXT_DEP_CHANNELS+ANNO_CHANNELS , 
                        CONTEXT_IND_CHANNELS_AB+CONTEXT_DEP_CHANNELS_AB + ANNO_CHANNELS_AB ))
################################################################################################



################################################################################################
Channel_Ind_Methods ={
    'char': charGrainChar, 
    'basic': basicGrainChar,
    'medical':medicalGrainChar,
    'radical':radicalGrainChar,
    'subcomp':subcompGrainChar,
    'stroke':strokeGrainChar,
}

Channel_Dep_Methods = {'pos': POSGrainSent}
Channel_Dep_TagSets = {'pos': posTag}
################################################################################################



def getChannelName(channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, abbr = False):

    channel = CHANNEL_ABBR[channel] if abbr else channel

    MN = str(Max_Ngram) if Max_Ngram > 1 else ''
    e  = 'e'            if end_grain else ''

    tS = str(len(tagScheme)) if abbr and tagScheme != 'BIO' else '-' + tagScheme.lower() if tagScheme != 'BIO' else ''  

    return channel + MN + e + tS

def getTagDict(TagList, tagScheme = 'BIO'):
    L = []
    suffices = ['-B', '-I']
    if 'O' in tagScheme:
        pref = specialTokens[:-1] + ['O']
    
    else:
        pref = specialTokens[:-1] # without UNK
    if 'E' in tagScheme:
        suffices = suffices + ['-E']
    if 'S' in tagScheme:
        suffices = suffices + ['-S']

    TagList = [i for i in TagList if i !='O']
    for tag in TagList:

        L.extend([tag+suff for suff in suffices])
    L.sort()
    L = pref + L
    
    return L


def extractEmbedPath2Info(embed_path, channel = None):
    
    if not os.path.isfile(embed_path):
        return None
    path_comp = embed_path.split('/')
    TokenNum_Dir = '/'.join(['channel'] + path_comp[1:4])
    # print(path_comp[-1].split('.')[0].split('_')[-1].lower())
    if channel:
        assert channel == path_comp[-1].split('.')[0].split('_')[-1].lower()
    else:
        channel = path_comp[-1].split('.')[0].split('_')[-1].lower()

    channel_abbr = CHANNEL_ABBR[channel]
    channel_name_abbr = [i for i in path_comp[4].split('_') if channel_abbr in i][0]

    MN_E = channel_name_abbr[len(channel_abbr): ]

    if MN_E == '':
         channel_name = channel
    elif channel in CONTEXT_IND_CHANNELS:
        channel_name = channel + MN_E
    else:
        if int(MN_E) == 5:
            channel_name = channel + '-bioes'
        elif int(MN_E) == 4:
            channel_name = channel + '-bioe'
        else:
            print('Fail to Extract information for embed:', embed_path, channel, MN_E)

    return TokenNum_Dir, channel_name