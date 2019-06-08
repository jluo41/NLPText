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

