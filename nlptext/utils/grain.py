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


# def generate_batch(data_word, data_index, batch_size, num_skips, skip_window, model = 'Skip-Gram', **kwargs):
#     # data_word is inside this function
#     # data_word = D[4]
#     # global data_index
#     assert batch_size % num_skips == 0
#     assert num_skips <= 2 * skip_window
#     assert model == 'CBOW' or model == 'Skip-Gram'
    
#     batch  = [0]*batch_size # np.ndarray( shape=(batch_size), dtype=np.int32)
#     labels = [0]*batch_size # np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
#     span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    
#     buffer = collections.deque(maxlen = span)
    
#     if data_index + span > len(data_word):
#         data_index = 0
#     buffer.extend(data_word[data_index:data_index + span])
#     data_index += span
    
#     for i in range(batch_size // num_skips):
#         context_words = [w for w in range(span) if w != skip_window]
#         words_to_use = random.sample(context_words, num_skips)
#         for j, context_word in enumerate(words_to_use):
#             batch [i * num_skips + j] = buffer[skip_window]
#             labels[i * num_skips + j] = buffer[context_word]
#         if data_index == len(data_word):
#             buffer.extend(data_word[0:span])
#             data_index = span
#         else:
#             buffer.append(data_word[data_index])
#             data_index += 1
#     # Backtrack a little bit to avoid skipping words in the end of a batch
#     data_index = (data_index + len(data_word) - span) % len(data_word)
#     # Two lines are added below to change the order of batch and labels
#     # To get CBOW model
#     # Comment them out to return to Skip-Gram.
#     if model == 'CBOW':
#         # batch.shape, labels.shape = labels.shape,batch.shape
#         labels, batch = batch, labels
#     # print(batch_size)
#     return batch, labels, data_index


# def generate_batch_idx(tgt_start_idx, batch_token_num, maxlen = 100, skip_window=5, 
#                        left2right='tgt2ctx', size2size = '1-1', tgtInctx=False, **kwargs):
    

#     # batch_size = batch_token_num * [(2*skip_window + I{tagInctx})if 1to1 else 1] 
#     next_tgt_start_idx = tgt_start_idx + batch_token_num
#     maxIdx = maxlen - skip_window
#     if next_tgt_start_idx > maxIdx:
#         next_tgt_start_idx  = next_tgt_start_idx -  maxIdx + skip_window
#         tgt_idx_list = list(range(tgt_start_idx, maxlen - skip_window)) + list(range(skip_window, next_tgt_start_idx))
#     else:
#         tgt_idx_list = list(range(tgt_start_idx, next_tgt_start_idx))
#     # print(tgt_idx_list, len(tgt_idx_list))
#     # return next_tgt_start_idx
#     # do as tgt-ctx
#     left, right = [], []
#     if size2size == '1-1':
#         if tgtInctx == True:
#             for tgt_idx in tgt_idx_list:
#                 # print(tgt_idx)
#                 context_ahead = list(range(tgt_idx - skip_window, tgt_idx))
#                 context_after = list(range(tgt_idx+1, tgt_idx + skip_window+1))
#                 right_new = context_ahead + context_after + [tgt_idx]
#                 # print(right)
#                 right.extend([[i] for i in right_new])
#                 left.extend([[tgt_idx]]*len(right_new))
#         else:
#             for tgt_idx in tgt_idx_list:
#                 # print(tgt_idx)
#                 context_ahead = list(range(tgt_idx - skip_window, tgt_idx))
#                 context_after = list(range(tgt_idx+1, tgt_idx + skip_window+1))
#                 right_new = context_ahead + context_after
#                 right.extend([[i] for i in right_new])
#                 left.extend([[tgt_idx]]*len(right_new))
#     if size2size in ['1-All', 'All-1']:
#         if tgtInctx == True:
#             for tgt_idx in tgt_idx_list:
#                 # print(tgt_idx)
#                 context_ahead = list(range(tgt_idx - skip_window, tgt_idx))
#                 context_after = list(range(tgt_idx+1, tgt_idx + skip_window+1))
#                 right_new = context_ahead + context_after + [tgt_idx]
#                 right.append(right_new)
#                 left.append([tgt_idx])
#         else:
#             for tgt_idx in tgt_idx_list:
#                 # print(tgt_idx)
#                 context_ahead = list(range(tgt_idx - skip_window, tgt_idx))
#                 context_after = list(range(tgt_idx+1, tgt_idx + skip_window+1))
#                 right_new = context_ahead + context_after 
#                 right.append(right_new)
#                 left.append([tgt_idx])

#     if left2right == 'ctx-tgt':
#         right, left = left, right

#     right, left = np.array(right), np.array(left)
#     return left, right, next_tgt_start_idx


# # get this from Channel_Settings
# def getTokenBatchInfo(batch, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, TokenNum_Dir = None):

#     elen, tkn = batch.shape
    
#     batch = np.reshape(batch, (elen*tkn))# .shape
#     Info_Leng = [tk.getGrainTensor(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme, 
#                                    end_grain = end_grain, TokenNum_Dir= TokenNum_Dir )  for tk in batch]


#     channelLeng = np.array([info_leng[1] for info_leng in Info_Leng], dtype='float32')
#     max_gr = int(np.max(channelLeng))
    
#     channelInfo = [info_leng[0] for info_leng in Info_Leng]
    
#     channelInfo_Final =np.zeros([len(channelInfo), max_gr], dtype=int)
#     for idx, info in enumerate(channelInfo):
#         channelInfo_Final[idx, :len(info)] = channelInfo[idx]

#     channelInfo_Final.reshape(elen, tkn, max_gr)
#     channelLeng.reshape(elen, tkn)

#     return channelInfo_Final, channelLeng # , max_gr



