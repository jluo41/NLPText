import random
import collections
import numpy as np

from .channel import Channel_Ind_Methods, Channel_Dep_Methods
from .infrastructure import specialTokens


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



def generate_batch(data_word, data_index, batch_size, num_skips, skip_window, model = 'Skip-Gram', **kwargs):
    # data_word is inside this function
    # data_word = D[4]
    # global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    assert model == 'CBOW' or model == 'Skip-Gram'
    
    batch  = [0]*batch_size # np.ndarray( shape=(batch_size), dtype=np.int32)
    labels = [0]*batch_size # np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    
    buffer = collections.deque(maxlen = span)
    
    if data_index + span > len(data_word):
        data_index = 0
    buffer.extend(data_word[data_index:data_index + span])
    data_index += span
    
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch [i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[context_word]
        if data_index == len(data_word):
            buffer.extend(data_word[0:span])
            data_index = span
        else:
            buffer.append(data_word[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data_word) - span) % len(data_word)
    # Two lines are added below to change the order of batch and labels
    # To get CBOW model
    # Comment them out to return to Skip-Gram.
    if model == 'CBOW':
        # batch.shape, labels.shape = labels.shape,batch.shape
        labels, batch = batch, labels
    # print(batch_size)
    return batch, labels, data_index



# get this from Channel_Settings
def getTokenBatchInfo(batch, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, TokenNum_Dir = None):
    # TODO
    # TODO: to identify the type of the elements in the batch
    #   Under the context of BasicObject
    #       Maybe Token
    #       Maybe StrToken
    #   Without the context of BasicObject
    #       Maybe strToken
    # data_type = type(batch[0])
    # if data_type == Token:
    #     pass
    # elif data_type == int:
    #     batch = [Token(tk_idx) for tk_idx in batch]
    # elif data_type == str:
    #     batch = [Token(token = tk) for tk in batch]

    Info_Leng = [tk.getGrainTensor(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme, 
                                   end_grain = end_grain, TokenNum_Dir= TokenNum_Dir )  for tk in batch]

    channelLeng = np.array([info_leng[1] for info_leng in Info_Leng])
    maxGrainLeng = int(np.max(channelLeng))
    channelInfo = [info_leng[0] for info_leng in Info_Leng]
    channelInfo = np.array([np.pad(info,[0,maxGrainLeng - len(info)], 'constant') for info in channelInfo])

    return channelInfo, channelLeng # , maxGrainLeng


def getSentBatchInfo(batch, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, useStartEnd = False, TokenNum_Dir = None):
    
    Info_GrTk_TkSt_mGr_mTk = [st.getGrainTensor(channel, 
                                           Max_Ngram = Max_Ngram, 
                                           useStartEnd = useStartEnd,
                                           tagScheme = tagScheme, 
                                           end_grain = end_grain, 
                                           TokenNum_Dir= TokenNum_Dir ) for st in batch]

    mGr  = np.max([i[3] for i in Info_GrTk_TkSt_mGr_mTk])
    mTk  = np.max([i[4] for i in Info_GrTk_TkSt_mGr_mTk])
    Info = np.array([np.pad(i[0], ((0,mTk-i[0].shape[0]), (0, mGr-i[0].shape[1])),  'constant', constant_values=0) 
                     for i in Info_GrTk_TkSt_mGr_mTk])
    GrTk = np.array([np.pad(i[1], (0,mTk-i[1].shape[0]),  'constant', constant_values=0) 
                     for i in Info_GrTk_TkSt_mGr_mTk])

    TkSt  = np.array([i[2] for i in Info_GrTk_TkSt_mGr_mTk])
    
    return Info, GrTk, TkSt


