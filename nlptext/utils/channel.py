import os
import re
import pickle
import string
from jieba import posseg
import nltk

import pyphen
from .infrastructure import specialTokens
try:
    from medpos.crfpp.tagger import medpos_tagger
except:
    print('MedPos is not available yet')
################## FOR THE CONTEXT-INDEPENDENT CHANNELS ################

##### basic
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

def basicGrainToken(token, end_grain = False):
    info = sum([basicGrainChar(char, end_grain) for char in token], [])
    return info

#### medical

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

def medicalGrainToken(token, end_grain = False):
    info = sum([medicalGrainChar(char, end_grain) for char in token], [])
    return info
    



current_dir_old = os.path.abspath(os.path.dirname(__file__))
current_dir, _ = os.path.split(current_dir_old)

# print(current_dir_old)
# print(current_dir)
# data_dir = os.path.join(current_dir, 'data')



#### subcomp

with open( os.path.join(current_dir, 'sources/CharSubComp.p'), 'rb') as handle:
    CharSubCompInfos = pickle.load(handle)

#### stroke

with open( os.path.join(current_dir, 'sources/CharStroke.p'), 'rb') as handle:
    CharStrokeInfos = pickle.load(handle)
    
#### radical

with open( os.path.join(current_dir, 'sources/CharRadical.p'), 'rb') as handle:
    CharRadicalInfos = pickle.load(handle)

#### pinyin

with open( os.path.join(current_dir, 'sources/CharPinyin.p'), 'rb') as handle:
    CharPinyinInfos = pickle.load(handle)


## phonemes
with open( os.path.join(current_dir, 'sources/WordPhoneme.p'), 'rb') as handle:
    WordPhenomeInfo = pickle.load(handle)


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

def subcompGrainToken(token, end_grain = False):
    info = sum([subcompGrainChar(char, end_grain) for char in token], [])
    return info


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

def strokeGrainToken(token, end_grain = False):
    info = sum([strokeGrainChar(char, end_grain) for char in token], [])
    return info

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

def radicalGrainToken(token, end_grain = False):
    info = sum([radicalGrainChar(char, end_grain) for char in token], [])
    return info

#### char
def charGrainChar(char, end_grain = False):
    '''char level only!'''
    info = [char]
    if end_grain:
        info = info + ['ch0']
    return info

def charGrainToken(token, end_grain = False):
    info = sum([charGrainChar(char, end_grain) for char in token], [])
    return info


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

def pinyinGrainToken(token, end_grain = False):
    info = sum([pinyinGrainChar(char, end_grain) for char in token], [])
    return info


## syllable
def syllableGrainToken(token, end_grain = False):
    # pyphen.LANGUAGES
    try:
        dic = pyphen.Pyphen(lang='en')
        info = dic.inserted(token).split('-')
    except:
        info = ['']
    return info


def phonemeGrainToken(token, end_grain = False): 
    try:
        phonemes = WordPhenomeInfo[token.lower()]
    except:
        phonemes = ['']
    return phonemes

################## FOR THE CONTEXT-DEPENDENT CHANNELS ################
posTag = ['a', 'ad', 'ag', 'an', 'b', 'c', 'd', 'df', 'dg', 'e', 'eng', 
          'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'mg', 'mq', 'n', 'ng', 'nr', 'nrfg', 
          'nrt', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'rg', 'rr','rz', 's', 't', 
          'tg', 'u', 'ud', 'ug', 'uj', 'ul', 
          'uv', 'uz', 'v', 'vd', 'vg', 'vi', 'vn', 'vq', 'x', 'y', 'yg', 'z', 'zg']

def POSGrainSent(sent, tokenLevel = 'word', tagScheme = 'BIOES'):
    '''
        # tokenLevel = 'word'
            1: sent = '北京 是 中国 的 首都' # be cautious about this sentence, final segmentation may be different from the input
            2: sent = '北京是中国的首都'     # this is ok
        Here only for Atom is Char Based
        This method should be enriched
        sent: List of Token(String), with or without Start or End
        sent: [str]
    '''
    if type(sent) == str:
        sent = sent.split(' ') 
    segs = list(posseg.cut(''.join(sent)))
    tokens = []
    GrainSent = []

    if tokenLevel == 'char':
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
            GrainSent.extend(labels)
            tokens.extend([i for i in pair.word])
        return GrainSent, tokens

    elif tokenLevel == 'word':
        for i in range(len(segs)):
            pair  = segs[i]
            label = pair.flag
            GrainSent.append(label+ '-B')
            tokens.append(pair.word )
            # print(pair.word)
        return GrainSent, tokens



MedPosTag = ['notsure', '临床属性', '事件','人群', '代词', '体征与症状',
 '医疗仪器', '医疗行为', '单位', '因果', '地名机构名', '定性', '或许','数字', '数学符号',
 '无', '时间单位', '时间概念', '有', '标点', '检查项目', '治疗项目', '物体',
 '生物', '疾病', '空间概念', '药物', '虚词','身体功能', '身体物质',  '身体部位', '连词']




def MedPOSGrainSent(sent, tokenLevel = 'char', tagScheme = 'BIOES'):
    

    result_entities = medpos_tagger(sent)
    sentence = [i for i in sent]

    entities = []
    for i in result_entities:
        s, e, label = i
        entities.append((''.join(sentence[s:e]), s, e, label))
    # entities
    
  
    assert tokenLevel == 'char'
    tokens = []
    GrainSent = []

    try:
        for sset in entities:
            string, s, e, label = sset
            leng  = len(string)
            labels= [label + '-I' ]*leng
            labels[0] = label + '-B'
            if 'E' in tagScheme and leng >= 2:
                labels[-1] = label + '-E'
            if 'S' in tagScheme and leng == 1:
                labels[0] = label + '-S'
            GrainSent.extend(labels)
            tokens.extend([i for i in string])
    except:
        print(entities)


    # print(list(zip(tokens, GrainSent)))
    
    return GrainSent, tokens




posEnTag = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 
            'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', 
            '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 
            'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX',
            'NNS', 'SYM', 'CC', 'CD', 'POS']

def POSENGrainSent(sent, tokenLevel = 'word', tagScheme = 'BIOES'):
    '''
        Here only for Atom is Char Based
        This method should be enriched
        sent: List of Token(String), with or without Start or End
        sent: [str]
    '''
    assert tokenLevel == 'word'
    if type(sent) == list:
        tokens = sent
    else:
        tokens = nltk.word_tokenize(sent)
    # segs = list(posseg.cut(''.join(sent)))
    GrainSent = [i[-1] +'-B' for i in nltk.pos_tag(tokens)]
    return GrainSent, tokens


##################################################################

################################################################################################
CONTEXT_IND_CHANNELS    = ['basic', 'medical', 'radical', 'token', 'char', 'subcomp', 'stroke', 'pinyin', 'syllable', 'phoneme', ]
CONTEXT_DEP_CHANNELS    = ['pos', 'pos_en', 'medpos']
ANNO_CHANNELS           = ['annoR', 'annoE']

CONTEXT_IND_CHANNELS_AB = ['b', 'm', 'r',  'T', 'C', 'c', 's', 'y', 'sl', 'ph']
CONTEXT_DEP_CHANNELS_AB = ['P', 'Pe', 'MP']
ANNO_CHANNELS_AB        = ['R', 'E']

CHANNEL_ABBR = dict(zip(CONTEXT_IND_CHANNELS + CONTEXT_DEP_CHANNELS+ANNO_CHANNELS , 
                        CONTEXT_IND_CHANNELS_AB+CONTEXT_DEP_CHANNELS_AB + ANNO_CHANNELS_AB ))


Channel_Ind_Methods ={
    'char': charGrainToken, 
    'basic': basicGrainToken,
    'medical':medicalGrainToken,
    'radical':radicalGrainToken,
    'subcomp':subcompGrainToken,
    'stroke':strokeGrainToken,
    'pinyin':pinyinGrainToken,
    'syllable': syllableGrainToken,
    'phoneme': phonemeGrainToken, 
}

Channel_Dep_Methods = {'pos': POSGrainSent, 
                       'pos_en': POSENGrainSent, 
                       'medpos': MedPOSGrainSent}

Channel_Dep_TagSets = {'pos': posTag, 
                       'pos_en': posEnTag, 
                       'medpos': MedPosTag}


################################################################################################

def getGrainNgrams(subword_infos, n):
    if n == 1:
        return [i for i in subword_infos]
    if n > len(subword_infos):
        # How to deal this when the length is not so long
        # Condition: where n is larger than the infos
        return [] 
    l = [subword_infos[i:n+i] for i in range(len(subword_infos) - n + 1)]
    l = ['-'.join(i) for i in l]
    return l


def grainToken(token, grainTokenFunction, Ngram = None, Min_Ngram = 1, Max_Ngram = 1, end_grain = True):
    infos =  grainTokenFunction(token, end_grain = end_grain) 
    if Ngram: Min_Ngram, Max_Ngram = Ngram, Ngram
    return sum([getGrainNgrams(infos, idx) for idx in range(Min_Ngram, Max_Ngram + 1)], [])


def getChannelGrain4Token(token, channel, Ngram = None, Min_Ngram = 1, Max_Ngram = 1,  end_grain = False):
    if channel == 'token':
        return [token]
    elif channel in Channel_Ind_Methods:
        return grainToken(token, Channel_Ind_Methods[channel], Ngram = Ngram, Min_Ngram = Min_Ngram,Max_Ngram = Max_Ngram, end_grain = end_grain)
    else:
        print('The Channel "', channel, '" is not available currently!')


def getChannelGrain4Sent(sent, channel, Ngram = None, Min_Ngram = 1, Max_Ngram = 1, tokenLevel = 'char', tagScheme =  'BIO', end_grain = False):
    if channel == 'token':
        return [[tk] for tk in sent.split(' ')]
    elif channel in Channel_Ind_Methods:
        return [getChannelGrain4Token(token, channel, Ngram, Min_Ngram, Max_Ngram, end_grain) for token in sent.split(' ')]
        # return grainSent_ctxInd(sent, channel, Ngram = Ngram, Max_Ngram = Max_Ngram,  end_grain = end_grain)
    elif channel in Channel_Dep_Methods:
        return [[gr] for gr in Channel_Dep_Methods[channel](sent, tokenLevel = tokenLevel, tagScheme = tagScheme)[0]]
        # return grainSent_ctxDep(sent, Channel_Dep_Methods[channel], tokenLevel =tokenLevel, tagScheme = tagScheme, useStartEnd = useStartEnd)
    else:
        print('The Channel "', channel, '" is not available currently!')


###############################################################################################################

###############################################################################################################
def getChannelName(channel, Min_Ngram = 1, Max_Ngram = 1,  end_grain = False, tagScheme = 'BIO', min_grain_freq = 1,
                   style = 'normal', channel_name = None, channel_name_abbr = None, **kwargs):
    # hyper: channel + '-bio'
    # sub  : channel + '-n1t5' + 'e' or '' + 'f5'
    if style == 'normal':
        if channel == 'token':
            return channel 
        elif channel in CONTEXT_IND_CHANNELS:
            MN = '-n' + str(Min_Ngram) + 't' + str(Max_Ngram)
            e  = 'e' if end_grain else ''
            f  = '-f' + str(min_grain_freq)
            return channel + MN + e + f
        else:
            tS = '-' + tagScheme.lower() 
            return channel + tS

    elif style == 'abbr':
        channel_abbr = CHANNEL_ABBR[channel] 
        if channel == 'token':
            return channel_abbr
        elif channel in CONTEXT_IND_CHANNELS:
            MN = '-n' + str(Min_Ngram) + 't' + str(Max_Ngram)
            e  = 'e' if end_grain else ''
            f  = '-f' + str(min_grain_freq)
            return channel_abbr + MN + e + f
        else:
            tS = '-' + tagScheme.lower() 
            return channel_abbr + tS

    elif channel_name and style == 'extract':
        assert channel in channel_name
        if channel == 'token':
            return channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme

        elif channel in CONTEXT_IND_CHANNELS:
            channel_name, freq = channel_name.split('-f')
            min_grain_freq = int(freq)
            MN_e = channel_name.split('-n')[1]
            # print(MN_e)
            if 'e' in MN_e:
                end_grain = True
                MN = MN_e[:-1]
            else:
                end_grain = False
                MN = MN_e
            Min_Ngram, Max_Ngram = [int(i) for i in MN.split('t')]
            return channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme
        
        else:
            tagScheme = channel_name.split('-')[-1].upper()
            return channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme
        
    elif channel_name_abbr and style == 'extract':
        channel_abbr = CHANNEL_ABBR[channel]
        assert channel_abbr in channel_name_abbr
        if channel == 'token':
            return channel_abbr, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme

        elif channel in CONTEXT_IND_CHANNELS:
            channel_name, freq = channel_name_abbr.split('-f')
            min_grain_freq = int(freq)
            MN_e = channel_name.split('-n') [1]
            if 'e' in MN_e:
                end_grain = True
                MN = MN_e[:-1]
            else:
                end_grain = False
                MN = MN_e
            Min_Ngram, Max_Ngram = [int(i) for i in MN.split('t')]
            return channel_abbr, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme
        
        else:
            tagScheme = channel_name.split('-')[-1].upper()
            return channel_abbr, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme
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
    