import os
from bisect import bisect
import numpy as np

from .base import BasicObject
from .token import Token


# from .utils import segSent2Tokens 
# from .utils import getChannelName 
# from .channel import getChannelGrain4Sent 
# from .channel import START, END, UNK_ID  


from .utils.grain import getChannelGrain4Sent 
from .utils.channel import getChannelName
from .utils.pyramid import segSent2Tokens
from .utils.infrastructure import START, END, UNK_ID


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
        # if 'anno' in channel
        channelToken = 'ANNOTokenIndex' if 'anno' in  channel else channel + 'TokenIndex'
        
        if channelToken in self.TOKEN and not self._sentence:
            # ANNO_CHANNEL and POSCHANNEL
            #- print('use', channelToken)
            ch = 'annoE' if channel == 'annoR' else channel
            bioes_LGU, bioes_DGU = self.getGrainUnique(ch, Max_Ngram, end_grain = end_grain, tagScheme = 'BIOES')
            s, e = self.IdxTokenStartEnd
            channelGrainIndex = self.TOKEN[channelToken][s:e] # tagScheme of BIOES
            channelGrain = [bioes_LGU[index] for index in channelGrainIndex]

            if 'S' not in tagScheme and 'E' not in tagScheme:
                channelGrain = [i.replace('-S', '-B').replace('-E', '-I')  for i in channelGrain]
            elif 'S' not in tagScheme:
                channelGrain = [i.replace('-S', '-B')  for i in channelGrain]
            elif 'E' not in tagScheme:
                channelGrain = [i.replace('-E', '-I')  for i in channelGrain]

            if channel == 'annoR':
                channelGrain = [i.split('-')[-1]  for i in channelGrain]
            if useStartEnd:
                return [[START]] + [[tk] for tk in channelGrain] + [[END]]
            else:
                return [[tk] for tk in channelGrain] # only Ngram = 1 is available


        elif channel in self.CONTEXT_IND_CHANNELS or channel in self.CONTEXT_DEP_CHANNELS:
            sent = [tk.token for tk in self.Tokens] # using `self.Tokens` is better.
            if useStartEnd:
                sent = [START] + sent + [END]
            tokenLevel = self._tokenLevel if self._tokenLevel else self.TOKEN['TOKENLevel']
            channelGrain =  getChannelGrain4Sent(sent, channel, Max_Ngram = Max_Ngram, tokenLevel = tokenLevel, 
                                                 tagScheme  = tagScheme, useStartEnd= useStartEnd, end_grain = end_grain)
            return channelGrain

        else:
            print('There is no channel:', channel, '!')
            return None

        
    def getGrainTensor(self, channel, Max_Ngram = 1, tagScheme = 'BIO', useStartEnd = False, end_grain = False, TokenNum_Dir = None,  dontUseLookUp = False):
        
        channel_name = getChannelName(channel, Max_Ngram, tagScheme, end_grain)
        
        tmp_TokenNum_Dir = TokenNum_Dir if TokenNum_Dir else self.TokenNum_Dir
        lookup_channel_name_path = os.path.join(tmp_TokenNum_Dir, 'LookUp', channel_name + '.p')
        
        if os.path.isfile(lookup_channel_name_path) and not dontUseLookUp:
            # option 1:
            ### case 1: deal with the case: CTX_IND channel and this channel has LOOKUP.p
            sent = [START_TK]  + self.Tokens + [END_TK] if useStartEnd else self.Tokens # using `self.Tokens` is better.
            info = [tk.getGrainTensor(channel, Max_Ngram, end_grain = end_grain, 
                                      tagScheme = tagScheme, TokenNum_Dir =TokenNum_Dir)[0] for tk in sent]
            info = [list(tk) for tk in info]

        else:
            # option 2: 
            ### case 1: deal with the case: CTX_IND channel and this channel doesn't have LOOKUP.p
            ### case 2: deal with the case: CTX_DEP channel and ANNO_CHANNEL
            LGU, DGU = self.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, 
                                           tagScheme = tagScheme, TokenNum_Dir =TokenNum_Dir)

            info = self.getChannelGrain(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme,  
                                        useStartEnd = useStartEnd, end_grain = end_grain)
            info = [[DGU.get(gr, UNK_ID) for gr in tk] for tk in info]
        
        leng = [len(tk) for tk in info]
        
        max_GrInTk  = max(leng)
        info = [tk + [0] * (max_GrInTk - len(tk)) for tk in info]# raw information

        Info        = np.array(info)
        leng_GrInTk = np.array(leng, dtype='float32') # GrInTk
        leng_TkInSt = np.array(len(info),dtype= 'float32')
        max_TkInSt  = len(info)
        return Info, leng_GrInTk, leng_TkInSt, max_GrInTk, max_TkInSt
        

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

        return "<Sentence " + ctx + " (tokenNum: " + str(self.length) + ") >"
