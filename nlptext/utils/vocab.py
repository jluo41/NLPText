import numpy as np
import collections
from datetime import datetime
from .channel import getChannelGrain4Token
from .infrastructure import UNK, UNK_ID, specialTokens, specialTokensDict


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


def get_num_freq(idx2freq, max_vocab_token_num = None, min_token_freq = 1):
    if min_token_freq:
        max_vocab_token_num = len(idx2freq[idx2freq > min_token_freq])
        print('coverage rate is:', max_vocab_token_num / len(idx2freq))
        return max_vocab_token_num,  min_token_freq
    elif max_vocab_token_num:
        if max_vocab_token_num > len(idx2freq):
            return len(idx2freq), 1
        else:
            min_token_freq = max_vocab_token_num[max_vocab_token_num]
            return max_vocab_token_num, min_token_freq
    else:
        raise('Error in max_vocab_token_num and min_token_freq')




##################################################################################################LTU_LGU-LT

def get_GU_or_LKP(TokenVocab, tkidx2freq, channel= 'char', Max_Ngram = 1, end_grain = False, 
                  max_vocab_token_num = None, min_token_freq = 1,
                  max_vocab_grain_num = None, min_grain_freq = 1):

    # ListGrainUnique = []
    LTU, DTU = TokenVocab
    max_vocab_token_num, min_token_freq = get_num_freq(tkidx2freq, max_vocab_token_num = max_vocab_token_num, 
                                                       min_token_freq = min_token_freq)
    LTU = LTU[:max_vocab_token_num]
    
    # the containers to store our results
    oldLGU = []
    oldDGU = {}
    oldidx2freq = []
    LKP = []
    
    print('For channel: |', channel, '| build GrainUnique and LookUp')
    for idx, token in enumerate(LTU):
        token_freq  = idx2freq[DTU[token]]
        ChN = getChannelGrain4Token(token, channel, Max_Ngram = Max_Ngram, end_grain = end_grain)
        grain2number = dict(collections.Counter(ChN).most_common())
        for gr in grain2number:
            if gr in oldDGU:
                oldidx2freq[oldDGU[gr]] = oldidx2freq[oldDGU[gr]] + grain2number[gr] * token_freq
            else:
                oldDGU[gr] = len(oldDGU)
                oldLGU.append(gr)
                oldidx2freq.append(grain2number[gr] * token_freq)

        LKP.append([oldDGU[gr] for gr in ChN])
        if idx % 100000 == 0:
            print('\t\tFor Channel:', channel, '\t', idx, datetime.now())

    # remove some high and low frequency grains.
    # how to deal with the high freqency grains?
    # notice that the grain freq is based on vocab instead of corpus.
    assert len(LKP) == len(LTU)
    
    # sort the LGU, DGU and renew LKP
    del oldDGU 
    
    grainidx2freq = np.sort(oldidx2freq)[::-1]
    newidx2oldidx = np.argsort(oldidx2freq)[::-1]
    del oldidx2freq

    oldidx2newidx = np.zeros(len(newidx2oldidx), dtype= int) 
    for new_idx, old_idx in enumerate(newidx2oldidx):
        oldidx2newidx[old_idx] = new_idx
    for tkidx, grainlist in enumerate(LKP):
        LKP[tkidx] = [oldidx2newidx[oldix] for oldix in grainlist]
    del oldidx2newidx

    LGU = []
    for new_idx in range(len(oldLGU)):
        LGU.append(oldLGU[newidx2oldidx[new_idx]])
    del oldLGU
    del newidx2oldidx

    DGU = {}
    for new_idx, token in enumerate(LGU):
        DGU[token] = new_idx
        
    return (LGU, DGU), LKP, grainidx2freq