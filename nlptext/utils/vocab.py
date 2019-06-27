import numpy as np
import collections
from datetime import datetime
from .channel import getChannelGrain4Token
from .infrastructure import UNK, UNK_ID, specialTokens, specialTokensDict


##################################################################################################TOKEN_LTU
def buildTokens(tokenList):
    """
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


##################################################################################################LTU_LGU-LT
def get_GU_or_LKP(TokenUnique, channel= 'char', Max_Ngram = 1, end_grain = False, specialTokens = specialTokens):
    # ListGrainUnique = []
    LTU, DTU = TokenUnique
    # num_specialtokens = len(specialTokens)
    # assert LTU[:num_specialtokens] == specialTokens
    DGU = {}
    new_grains = []
    # ListGrainUnique = ListGrainUnique + new_grains
    # LKP = [[idx] for idx in range(num_specialtokens)]
    LKP = []
    print('For channel: |', channel, '| build GrainUnique and LookUp')
    for idx, token in enumerate(LTU):
        ChN = getChannelGrain4Token(token, channel, Max_Ngram= Max_Ngram, end_grain = end_grain)
        new_grains = [i for i in set(ChN) if i not in DGU]
        # make sure to keep the order
        new_grains.sort() 
        for gr in new_grains:
            DGU[gr] = len(DGU)
        LKP.append([DGU[gr] for gr in ChN])
        if idx % 100000 == 0:
            print('\t\tFor Channel:', channel, '\t', idx, datetime.now())
            
    LGU = list(DGU.keys())
    # assert ListGrainUnique[:num_specialtokens] == specialTokens
    assert len(LKP) == len(LTU)
    return (LGU, DGU), LookUp
##################################################################################################LTU_LGU-LT
