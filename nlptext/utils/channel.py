import os
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



with open('nlptext/sources/CharPinyin.p', 'rb') as handle:
    CharPinyinInfos = pickle.load(handle)

def pinyinGrainChar(char, end_grain = False):
    '''char level only!'''
    if char in CharPinyinInfos:
        info = CharPinyinInfos[char]
        if info:
            info = [i for i in info ]
        else:
            info = ['po'] 
    else:
        info = [char]
        
    if end_grain:
        info = info + ['y0']
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
CONTEXT_IND_CHANNELS    = ['basic', 'medical', 'radical', 'token', 'char', 'subcomp', 'stroke', 'pinyin']
CONTEXT_DEP_CHANNELS    = ['pos']
ANNO_CHANNELS           = ['annoR', 'annoE']

CONTEXT_IND_CHANNELS_AB = ['b', 'm', 'r',  'T', 'C', 'c', 's', 'y']
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
    'pinyin':pinyinGrainChar,
}

Channel_Dep_Methods = {'pos': POSGrainSent}
Channel_Dep_TagSets = {'pos': posTag}
################################################################################################


def getGrainNgrams(subword_infos, n):
    # Here N is the Num for n_gram
    #     subword_infos: [subcomp1, subcomp2, ...] or [stroke1, stroke2, ...]
    #                 n: the targeted n gram
    if n == 1:
        return [i for i in subword_infos]
    if n > len(subword_infos):
        # How to deal this when the length is not so long
        # Condition: where n is larger than the infos
        return [] 
    l = [subword_infos[i:n+i] for i in range(len(subword_infos) - n + 1)]
    l = ['-'.join(i) for i in l]
    return l

def grainToken(token, grainCharFunction, Ngram = 1,Max_Ngram = None, end_grain = True):
    '''
        token level only!
        The input token is not in Special Tokens. The input token is a string!
        TODO: handle the `ngram` problems here. Content-Idenpendent Only
    '''
    if token not in specialTokens:
        infos = sum([grainCharFunction(char, end_grain) for char in token], [])
        if not Max_Ngram:
            return getGrainNgrams(infos, Ngram)
        else:
            return sum([getGrainNgrams(infos, idx+1) for idx in range(Max_Ngram)], [])
    else:
        return getGrainNgrams([token], Ngram) # deal with the special tokens

def getChannelGrain4Token(token, channel, Ngram = 1, Max_Ngram = None,  end_grain = False):
    '''
        token level only!
        The input token is not in Special Tokens
        The input token is a string!
        TODO: handle the `ngram` problems here.
        Content-Idenpendent Only
    '''
    if channel == 'token':
        return [token]
    elif channel in Channel_Ind_Methods:
        return grainToken(token, Channel_Ind_Methods[channel], Ngram = Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain)
    else:
        print('The Channel "', channel, '" is not available currently!')

###############################################################################################################
def grainSent_ctxInd(sent, channel, Ngram = 1, Max_Ngram = None,  end_grain = False):
    return [getChannelGrain4Token(token, channel, Ngram, Max_Ngram, end_grain) for token in sent]
    
def grainSent_ctxDep(sent, channelGrainSent, tokenLevel = 'word', tagScheme = 'BIO', useStartEnd = True):
    return channelGrainSent(sent, tokenLevel=tokenLevel, tagScheme=tagScheme, useStartEnd = useStartEnd)

def getChannelGrain4Sent(sent, channel, Ngram = 1, Max_Ngram = None, tokenLevel = 'char', tagScheme =  'BIO', useStartEnd = True, end_grain = False):
    '''
        token level only! The input token is not in Special Tokens. The input token is a string!
        TODO: handle the `ngram` problems here. Content-Idenpendent Only
    '''
    if channel in Channel_Ind_Methods:
        return grainSent_ctxInd(sent, channel, Ngram = Ngram, Max_Ngram = Max_Ngram,  end_grain = end_grain)
    elif channel in Channel_Dep_Methods:
        return grainSent_ctxDep(sent, Channel_Dep_Methods[channel], tokenLevel =tokenLevel, tagScheme = tagScheme, useStartEnd = useStartEnd)
    else:
        print('The Channel "', channel, '" is not available currently!')
###############################################################################################################


############### PART Channel Name
def getChannelName(channel, Max_Ngram = 1,  end_grain = False, tagScheme = 'BIO', style = 'normal', channel_name = None, channel_name_abbr = None, **kwargs):

    if style == 'normal':
        MN = str(Max_Ngram) if Max_Ngram > 1 else ''
        e  = 'e'            if end_grain else ''
        tS = '-' + tagScheme.replace('BIO', '').lower() if tagScheme != 'BIO' else ''
        return channel + MN + e + tS

    elif style == 'abbr':
        channel = CHANNEL_ABBR[channel] # if abbr else channel
        MN = str(Max_Ngram) if Max_Ngram > 1 else ''
        e  = 'e'            if end_grain else ''
        tS = '-' + tagScheme.replace('BIO', '').lower() if tagScheme != 'BIO' else ''
        return channel + MN + e + tS

    elif channel_name and style == 'extract':
        assert channel in channel_name
        MN_e_tS = channel_name[len(channel):]
        if len(MN_e_tS) == 0:
            return channel, Max_Ngram, end_grain, tagScheme
        if MN_e_tS[0] in '23456789':
            Max_Ngram = int(MN_e_tS[0])
            e_ts = MN_e_tS[1:]
            if len(e_ts) == 0:
                return channel, Max_Ngram, end_grain, tagScheme
        else:
            Max_Ngram = 1
            e_ts = MN_e_tS
        
        if e_ts[0] == 'e':
            end_grain = True
            ts = e_ts[1:]
        else:
            end_grain = False
            ts = e_ts
        if ts.upper() in ['-ES', '-E', '-S']:
            tagScheme = 'BIO' + ts.upper()[1:]
        else:
            tagScheme = 'BIO'
        return channel, Max_Ngram, end_grain, tagScheme
        
    elif channel_name_abbr and style == 'extract':
        channel_abbr = CHANNEL_ABBR[channel]
        MN_e_tS = channel_name_abbr[len(channel_abbr): ]
        if len(MN_e_tS) == 0:
            return channel, Max_Ngram, end_grain, tagScheme
        if MN_e_tS[0] in '23456789':
            Max_Ngram = int(MN_e_tS[0])
            e_ts = MN_e_tS[1:]
            if len(e_ts) == 0:
                return channel, Max_Ngram, end_grain, tagScheme
        else:
            Max_Ngram = 1
            e_ts = MN_e_tS
        
        if e_ts[0] == 'e':
            end_grain = True
            ts = e_ts[1:]
        else:
            end_grain = False
            ts = e_ts
        if ts.upper() in ['-ES', '-E', '-S']:
            tagScheme = 'BIO' + ts.upper()[1:]
        else:
            tagScheme = 'BIO'
        
        return channel, Max_Ngram, end_grain, tagScheme

    else:
        print('Error in getChannelName')

def get_Channel_Settings(CHANNEL_SETTINGS_TEMPLATE):
    d = CHANNEL_SETTINGS_TEMPLATE.copy()
    try:
        CHANNEL_SETTINGS = {channel: d[channel] for channel in d 
                            if d[channel].pop('use') == True}
    except:
        CHANNEL_SETTINGS = {k:v for k, v in d.items()}

    nameList = []

    for channel in CHANNEL_SETTINGS:
        # CHANNEL_SETTINGS[]
        channel_setting = CHANNEL_SETTINGS[channel]
        #channel_setting.pop('use')
        Max_Ngram    = channel_setting.get('Max_Ngram', 1)
        end_grain    = channel_setting.get('end_grain', False)
        tagSet       = channel_setting.get('tagSet',    None)
        tagScheme    = channel_setting.get('tagScheme', 'BIO')

        channel_name_abbr = getChannelName(channel, Max_Ngram, end_grain, tagScheme, style = 'abbr')
        nameList.append(channel_name_abbr)
    
    folderName = '_'.join(nameList)
    
    return CHANNEL_SETTINGS, folderName
    