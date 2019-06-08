import os
from bisect import bisect
import numpy as np

from .base import BasicObject
from .token import Token

from .utils.grain import getChannelGrain4Sent 
from .utils.channel import getChannelName
from .utils.pyramid import segSent2Tokens
from .utils.infrastructure import START, END, START_ID, END_ID, UNK_ID


START_TK = Token(token = START)
END_TK   = Token(token = END)


class Sentence(BasicObject):

    _sentence = None
    _tokenLevel = None

    def __init__(self, Idx = None, sentence = None, tokenLevel = None):
        self.Idx = Idx
        if sentence:
            self._sentence = sentence # could be a string, or a list of token string.
        if tokenLevel:
            self._tokenLevel = tokenLevel

    @property
    def sentence(self):
        if self._sentence:
            if type(self._sentence) == str:
                return self._sentence
            elif type(self._sentence) == list:
                return ''.join(self._sentence)
        else:
            return ''.join([tk.token for tk in self.Tokens])


    def getChannelGrain(self, channel,  Max_Ngram = 1, tagScheme = 'BIO',  useStartEnd = False, end_grain = False):
        channelToken = 'ANNOTokenIndex' if 'anno' in  channel else channel + 'TokenIndex'
        
        if channel == 'token':
            channelGrain = [[tk.token] for tk in self.Tokens]
        
        elif channelToken in self.TOKEN and not self._sentence:
            #print('Use Pyramid')
            LGU, DGU = self.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
            channel_grain_index = self.build_ctx_dep_grain(channel, tagScheme)
            channelGrain = [[LGU[i]] for i in channel_grain_index]

        else:
            sent = [tk.token for tk in self.Tokens] # using `self.Tokens` is better.
            tokenLevel = self._tokenLevel if self._tokenLevel else self.TOKEN['TOKENLevel']
            channelGrain =  getChannelGrain4Sent(sent, channel, Max_Ngram = Max_Ngram, tokenLevel = tokenLevel, 
                                                 tagScheme  = tagScheme, useStartEnd= False, end_grain = end_grain)

        if useStartEnd:
            return [[START]] + channelGrain + [[END]]
        else:
            return channelGrain

        
    def getGrainTensor(self, channel, Max_Ngram = 1, tagScheme = 'BIO', useStartEnd = False, end_grain = False, 
                       TokenNum_Dir = None, channel_name = None, dontUseLookUp = False):
        
        # BUG to Fix: Error with useStartEnd = True
        if channel == 'token':
            if TokenNum_Dir:
                LTU, DTU = self.getGrainUnique(channel, TokenNum_Dir =TokenNum_Dir)
                info = [[DTU.get(tk.token, UNK_ID)] for tk in self.Tokens]
            else:
                info = [[tk.index] for tk in self.Tokens]
            if useStartEnd:
                info = [[START_ID]] + [info] + [[END_ID]]

            leng_st = len(info)
            return info, leng_st, [1] * leng_st, 1

        if not channel_name:
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)

        ########################################################################## Speical Case for IND and DEP
        if channel in self.CONTEXT_IND_CHANNELS and not dontUseLookUp:
            ####################################################################### LookUp Table
            try:
                sent = self.Tokens # using `self.Tokens` is better.
                info_leng = [tk.getGrainTensor(channel, Max_Ngram, end_grain = end_grain, 
                                               tagScheme = tagScheme, TokenNum_Dir =TokenNum_Dir) for tk in sent]
                info    = [i[0] for i in info_leng]
                leng_tk = [i[1] for i in info_leng]
                info, leng_st, leng_tk, max_gr = self.padding_info(info, leng_tk, useStartEnd = useStartEnd)
                return info, leng_st, leng_tk, max_gr
            except:
                print('\tNo LookUp Table is found for channel:  ', channel_name, 'Turn to the orignal way... (st.getGrainTensor)')
            ####################################################################### LookUp Table

        elif channel not in self.CONTEXT_IND_CHANNELS and not self._sentence:
            try:
                info    = self.build_ctx_dep_grain(channel, tagScheme, TokenNum_Dir = TokenNum_Dir)
                info    = [[i] for i in info]
                info, leng_st, leng_tk, max_gr = self.padding_info(info, useStartEnd = useStartEnd)
                return info, leng_st, leng_tk, max_gr
            except:
                print('\tIn Pyramid, there is no CTX_DEP channel:', channel, 'Turn to the orignal way... (st.getGrainTensor)') 
        ########################################################################## Speical Case for IND and DEP


        # option 2: 
        ### case 1: deal with the case: CTX_IND channel and this channel doesn't have LOOKUP.p
        ### case 2: deal with the case: CTX_DEP channel and ANNO_CHANNEL
        LGU, DGU = self.getGrainUnique(channel, channel_name = channel_name, TokenNum_Dir =TokenNum_Dir)

        info = self.getChannelGrain(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme,  
                                    useStartEnd = False, end_grain = end_grain)
        info = [[DGU.get(gr, UNK_ID) for gr in tk] for tk in info]
        
        info, leng_st, leng_tk, max_gr = self.padding_info(info, useStartEnd = useStartEnd)
        return info, leng_st, leng_tk, max_gr


    def padding_info(self, info, leng_tk = None, useStartEnd = True):
        if not leng_tk:
            leng_tk = [len(i) for i in info]

        if useStartEnd:
            info = [[START_ID]] + [info] + [[END_ID]]
            leng_tk = [1] + leng_tk + [1]

        leng_st = len(info)
        max_gr  = max(leng_tk)
        
        info_final =np.zeros([leng_st, max_gr], dtype=int)
        
        # for idx, tk_info in enumerate(info):
        #     info_final[idx, :len(tk_info)] = info[idx]
        info = [tk + [0] * (max_gr - len(tk)) for tk in info] 
        
        return info, leng_st, leng_tk, max_gr
        

    def build_ctx_dep_grain(self, channel, tagScheme, TokenNum_Dir = None, to_tmp = False):
        channelToken = 'ANNOTokenIndex' if 'anno' in  channel else channel + 'TokenIndex'
        s, e = self.IdxTokenStartEnd
        bioes_grain_index = self.TOKEN[channelToken][s:e]
        bioes2tag = self.get_BIOES_Trans(channel, tagScheme, TokenNum_Dir = TokenNum_Dir)
        channel_grain_index = [bioes2tag[gr_idx] for gr_idx in bioes_grain_index]
        if to_tmp:
            self.CTX_DEP_TMP[channel+tagScheme] = {s+i:gr_idx for i, gr_idx in enumerate(channel_grain_index)}
        return channel_grain_index

    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxFolder)

    @property
    def IdxFolder(self):
        return bisect(self.FOLDER['EndIDXTexts'] , self.IdxText)

    @property
    def IdxText(self):
        return bisect(self.TEXT['EndIDXSents'] , self.Idx)

    @property
    def IdxTokenStartEnd(self):
        s, e = self.Idx, self.Idx + 1
        s = self.SENT['EndIDXTokens'][s-1] if s != 0 else 0
        e = self.SENT['EndIDXTokens'][e-1] if e != 0 else 0
        return s, e 

    @property
    def Corpus(self):
        from .corpus import Corpus 
        return Corpus(self.IdxCorpus)

    @property
    def Folder(self):
        from .folder import Folder 
        return Folder(self.IdxFolder)

    @property
    def Text(self):
        from .text import Text 
        return Text(self.IdxText)


    @property
    def Tokens(self):
        # without START or END in this setting.
        if self._sentence:
            sent = segSent2Tokens(self._sentence, method = self.SENT['Sent2TokenMethod'])
            sent = [Token(token = token, Sentence = self, IdxTokenInSent = idx) for idx, token in enumerate(sent)]

        else:
            sent = [Token(Idx) for Idx in range(*self.IdxTokenStartEnd)]

        return sent

    @property
    def TokensStartEnd(self):
        return [START_TK]  + self.Tokens + [END_TK]

    @property
    def length(self):
        if self._sentence:
            length = len(self.Tokens)
        else:
            s, e = self.IdxTokenStartEnd 
            length = e-s
        return length

    def __repr__(self):
        

        ctx =  str(self.Idx)  if type(self.Idx) == int else 'New'

        return "<st " + ctx + " (tokenNum: " + str(self.length) + ") >"
