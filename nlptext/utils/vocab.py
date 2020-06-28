import numpy as np
import collections
from datetime import datetime
from .channel import getChannelGrain4Token
from .infrastructure import UNK, UNK_ID, specialTokens, specialTokensDict
import pandas as pd

##################################################################################################TOKEN_LTU
def buildTokens(tokenList):
    """
        'deprecated: not suitable to handle big corpus'
        Process raw inputs into a dataset.
        words: a list of the whole corpus
    """
    #########################################################################COUNT
    total_len_token = len(tokenList)
    print('The Total Number of Tokens:', total_len_token)
    print('Counting the number unique Tokens...          \t', datetime.now())
    count = collections.Counter(tokenList).most_common()
    print('\t\tDone!')
    #########################################################################COUNT

    print('Generating Dictionary of Token Unique...\t', datetime.now())
    # grain2index
    # DTU_freq = {sp_tk: 0 for sp_tk in specialTokens}
    DTU = {}
    index2freq = []
    for token, freq in count:
        DTU[token] = len(DTU)
        index2freq.append(freq)
        # if token is not specialTokens:
        #     DTU[token] = len(DTU)
        #     DTU_freq[token] = freq
        # else:
        #     DTU_freq[token] = DTU_freq[token] + 1

    print('\t\tThe length of original vocabulary is:', len(DTU), '\t', datetime.now())
    print('Generating the ORIGTokenIndex...       \t', datetime.now())
    data = np.zeros(len(tokenList), dtype = np.uint32)
    # data = []
    for idx, token in enumerate(tokenList):
        data[idx] = DTU[token]
        if idx % 5000000 == 0:
            print('\t\tThe idx of token is:', idx, '\t', datetime.now())
    print('\t\tDone!')
    LTU = list(DTU.keys())
    # if MaxTokenUnique:
    #     print('Only Keep First', MaxTokenUnique, 'Tokens.')
    #     print('The coverage rate is:', np.bincount(data)[UNK_ID]/total_len_token)
    # data = np.array(data)
    return data, LTU, DTU, index2freq
##################################################################################################TOKEN_LTU



def get_num_freq(idx2freq, min_token_freq = 1):
    max_vocab_token_num = len(idx2freq[idx2freq >= min_token_freq])
    return max_vocab_token_num

def get_GU_or_LKP(TokenVocab, 
                  tkidx2freq = None, 
                  channel = 'char', 
                  Min_Ngram = 1, 
                  Max_Ngram = 1, 
                  end_grain = False, 
                  min_grain_freq = 1):

    '''
        tkidx2freq: if tkidx2freq is None, then we treat each token as equal. then the grain weights are generated based Vocab instead of Corpus.

    '''
    # ListGrainUnique = []
    LTU, DTU = TokenVocab
    # max_vocab_token_num = get_num_freq(tkidx2freq, min_token_freq = min_token_freq)
    # LTU = LTU[:max_vocab_token_num]

    if tkidx2freq == None:
        tkidx2freq = [1] * len(LTU)
    
    # the containers to store our results
    oldLGU = []
    oldDGU = {}
    oldidx2freq = []
    LKP = []
    
    print('For channel: |', channel, '| build GrainUnique and LookUp')
    for idx, token in enumerate(LTU):
        token_freq  = tkidx2freq[DTU[token]]
        ChN = getChannelGrain4Token(token, channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain)
        grain2number = dict(collections.Counter(ChN).most_common())
        for gr in grain2number:
            if gr in oldDGU:
                oldidx2freq[oldDGU[gr]] = oldidx2freq[oldDGU[gr]] + grain2number[gr] * token_freq
                # oldidx2freq[oldDGU[gr]] = oldidx2freq[oldDGU[gr]] + grain2number[gr] 
            else:
                oldDGU[gr] = len(oldDGU)
                oldLGU.append(gr)
                oldidx2freq.append(grain2number[gr] * token_freq)
                # oldidx2freq.append(grain2number[gr])

        LKP.append([oldDGU[gr] for gr in ChN])
        if idx % 100000 == 0:
            print('\t\tFor Channel:', channel, '\t', idx, datetime.now())

    # remove some high and low frequency grains.
    # how to deal with the high freqency grains?
    # notice that the grain freq is based on vocab instead of corpus.
    del oldDGU 
    assert len(LKP) == len(LTU)
    
    # sort the LGU, DGU and renew LKP
    oldidx2freq = np.array(oldidx2freq)
    max_grain_num = len(oldidx2freq[oldidx2freq >= min_grain_freq])
    
    grainidx2freq = np.sort(oldidx2freq)[::-1]
    newidx2oldidx = np.argsort(oldidx2freq)[::-1]
    del oldidx2freq

    oldidx2newidx = np.zeros(len(newidx2oldidx), dtype= int) 
    for new_idx, old_idx in enumerate(newidx2oldidx):
        oldidx2newidx[old_idx] = new_idx
    
    for tkidx, grainlist in enumerate(LKP):
        new_grainlist = []
        for oldidx in grainlist:
            newidx = oldidx2newidx[oldidx]
            # throw away the low frequency grains
            if grainidx2freq[newidx] < min_grain_freq:
                continue
            new_grainlist.append(newidx)
        LKP[tkidx] = new_grainlist 
    del oldidx2newidx

    LGU = []
    for new_idx in range(max_grain_num):
        # to filter some grains
        LGU.append(oldLGU[newidx2oldidx[new_idx]])
    del oldLGU
    del newidx2oldidx

    DGU = {}
    for new_idx, token in enumerate(LGU):
        DGU[token] = new_idx
        
    grainidx2freq = grainidx2freq[:max_grain_num]
    
    return (LGU, DGU), LKP, grainidx2freq