import os
from bisect import bisect
import numpy as np 

from .base import BasicObject

# from .channel import getChannelGrain4Token 
# from .utils import getChannelName 
# from .channel import START, END, UNK_ID   

from .utils.grain import getChannelGrain4Token
from .utils.channel import getChannelName
from .utils.infrastructure import UNK_ID

class Token(BasicObject):

    _token = None
    _Sentence = None 
    _IdxTokenInSent = None

    def __init__(self, Idx = None, token = None, Sentence = None, IdxTokenInSent = None):
        self.Idx = Idx
        if token:
            self._token = token
        if Sentence:
            self._Sentence = Sentence # type is nlptext.Sentence
            self._IdxTokenInSent = IdxTokenInSent

    # index: is the index in 
    @property
    def index(self):
        if self._token:
            return self.TokenUnique[1].get(self._token, UNK_ID)
        else:
            return self.TOKEN['ORIGTokenIndex'][self.Idx]

    @property
    def token(self):
        if self._token:
            return self._token
        else:
            return self.TokenUnique[0][self.index] # quick? or slow?

    def getChannelGrain(self, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False):

        # channel: channelNe;
        if channel in self.CONTEXT_IND_CHANNELS:
            # here the channel is only from the CTX_IND channels.
            return getChannelGrain4Token(self.token, channel, Max_Ngram = Max_Ngram, end_grain = end_grain)
        else:
            # self._token exists.
            channelGrain = self.Sentence.getChannelGrain(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme,
                                                         useStartEnd = False, end_grain = end_grain)
            return channelGrain[self.IdxTokenInSent]

    def getGrainTensor(self, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, TokenNum_Dir = None, dontUseLookUp = False):
        '''Inputs: channel_name, and GrainUnique'''

        ########################################################################################################
        # option 1
        channel_name = getChannelName(channel, Max_Ngram, tagScheme, end_grain)
        tmp_TokenNum_Dir = TokenNum_Dir if TokenNum_Dir else self.TokenNum_Dir
        lookup_channel_name_path = os.path.join(tmp_TokenNum_Dir, 'LookUp', channel_name + '.p')
        if os.path.isfile(lookup_channel_name_path) and not dontUseLookUp:
            
            ### case 1: deal with the case: CTX_IND channel and this channel has LOOKUP.p
            LOOKUP, TokenUnqiue = self.getLookUp(channel, Max_Ngram, end_grain = end_grain, 
                                                 tagScheme = tagScheme, TokenNum_Dir = TokenNum_Dir)
            LTU, DTU = TokenUnqiue
            index = DTU.get(self.token, UNK_ID)

            if index != UNK_ID:
                # print('\tUse LookUp from:'+lookup_channel_name_path)
                info = LOOKUP[index] # TODO, to do what?
                Info = np.array(info)
                Leng = np.array(len(info), dtype='float32')
                return Info, Leng 
        ########################################################################################################

        ########################################################################################################
        # option 2: 
        ### case 1: deal with the case: CTX_IND channel and this channel doesn't have LOOKUP.p
        ### case 2: deal with the case: CTX_DEP channel and ANNO_CHANNEL
        # print(TokenNum_Dir)
        LGU, DGU = self.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, 
                                       tagScheme = tagScheme, TokenNum_Dir =TokenNum_Dir)
        info = [DGU.get(i, UNK_ID) for i in self.getChannelGrain(channel,Max_Ngram = Max_Ngram, 
                                                                       end_grain = end_grain, tagScheme = tagScheme)]

        Info = np.array(info)
        Leng = np.array(len(info), dtype='float32')
        return Info, Leng 
        ########################################################################################################


    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxFolder)

    @property
    def IdxFolder(self):
        return bisect(self.FOLDER['EndIDXTexts'] , self.IdxText)

    @property
    def IdxText(self):
        return bisect(self.TEXT['EndIDXSents'] , self.IdxSent)

    @property
    def IdxSent(self):
        return bisect(self.SENT['EndIDXTokens'] , self.Idx)

    @property
    def IdxTokenInSent(self):
        if type(self._IdxTokenInSent) == int:
            return self._IdxTokenInSent
        else:
            return self.Idx - self.Sentence.IdxTokenStartEnd[0]
    
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
    def Sentence(self):
        if self._Sentence:
            return self._Sentence
        else:
            from .sentence import Sentence
            return Sentence(self.IdxSent)

    def __repr__(self):
        return "<Token "  + self.token + " >"

    