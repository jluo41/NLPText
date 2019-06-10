import numpy as np
import collections
from datetime import datetime
from .channel import getChannelGrain4Token
from .infrastructure import UNK, UNK_ID, specialTokens, specialTokensDict


##################################################################################################TOKEN_LTU
def buildTokens(tokenList, MaxTokenUnique = None):
    """
        Process raw inputs into a dataset.
        words: a list of the whole corpus
    """
    #########################################################################COUNT
    total_len_token = len(tokenList)
    print('The Total Number of Tokens:', total_len_token)
    print('Counting the number unique Tokens...          \t', datetime.now())
    if MaxTokenUnique:
        count = collections.Counter(tokenList).most_common(MaxTokenUnique)
    else:
        count = collections.Counter(tokenList).most_common()
    print('\t\tDone!')
    #########################################################################COUNT

    print('Generating Dictionary of Token Unique...\t', datetime.now())
    DTU = specialTokensDict.copy()
    DTU_freq = {sp_tk: 0 for sp_tk in specialTokens}
    for token, freq in count:
        if token is not specialTokens:
            DTU[token] = len(DTU)
            DTU_freq[token] = freq
        else:
            DTU_freq[token] = DTU_freq[token] + 1

    print('\t\tThe length of DTU is:', len(DTU), '\t', datetime.now())
    print('Generating the ORIGTokenIndex...       \t', datetime.now())
    data = np.zeros(len(tokenList), dtype = np.uint32)
    # data = []
    for idx, token in enumerate(tokenList):
        voc_id = DTU.get(token, UNK_ID)
        data[idx] = voc_id
        if voc_id == UNK_ID:
            DTU_freq[UNK] = DTU_freq[UNK] + 1

        # data.append(DTU.get(token,UNK_ID))
        if idx % 5000000 == 0:
            print('\t\tThe idx of token is:', idx, '\t', datetime.now())
    print('\t\tDone!')
    LTU = list(DTU.keys())

    if MaxTokenUnique:
        print('Only Keep First', MaxTokenUnique, 'Tokens.')
        print('The coverage rate is:', np.bincount(data)[UNK_ID]/total_len_token)
    # data = np.array(data)
    return data, LTU, DTU, DTU_freq
##################################################################################################TOKEN_LTU


##################################################################################################LTU_LGU-LT
def get_GU_or_LKP(TokenUnique, channel= 'char', Max_Ngram = 1, end_grain = False, specialTokens = specialTokens):
    # ListGrainUnique = []
    LTU, DTU = TokenUnique
    num_specialtokens = len(specialTokens)
    assert LTU[:num_specialtokens] == specialTokens
    DictGrainUnique = {}
    new_grains = [stk for stk in specialTokens]
    # ListGrainUnique = ListGrainUnique + new_grains
    for stk in new_grains:
        DictGrainUnique[stk] = len(DictGrainUnique)

    LookUp = [[idx] for idx in range(num_specialtokens)]
    
    print('For channel: |', channel, '| build GrainUnique and LookUp')
    for idx, token in enumerate(LTU[num_specialtokens:]):
        ChN = getChannelGrain4Token(token, channel, Max_Ngram= Max_Ngram, end_grain = end_grain)
        new_grains = [i for i in set(ChN) if i not in DictGrainUnique]
        new_grains.sort() # make sure to keep the order

        for stk in new_grains:
            DictGrainUnique[stk] = len(DictGrainUnique)
        LookUp.append([DictGrainUnique.get(gr) for gr in ChN])
        if (idx + num_specialtokens) % 100000 == 0:
            print('\t\tFor Channel:', channel, '\t', idx + num_specialtokens, datetime.now())
            
    ListGrainUnique = list(DictGrainUnique.keys())
    assert ListGrainUnique[:num_specialtokens] == specialTokens
    assert len(LookUp) == len(LTU)
    return (ListGrainUnique, DictGrainUnique), LookUp
##################################################################################################LTU_LGU-LT
