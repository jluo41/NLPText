import os
from bisect import bisect
import numpy as np 

from .base import BasicObject
from .utils.channel import getChannelGrain4Token, getChannelName
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

        if channel in self.CONTEXT_IND_CHANNELS:
            return getChannelGrain4Token(self.token, channel, Max_Ngram = Max_Ngram, end_grain = end_grain)

        else:
            ##################################################################################### Speical Case for CTX_DEP
            if not self._token:
                try:
                    LGU, DGU = self.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
                    grain_idx = self.get_ctx_dep_grain(channel, tagScheme = tagScheme)
                    return [LGU[grain_idx]]
                except:
                    print('\tIn Pyramid, there is no CTX_DEP channel:', channel, 'Turn to the orignal way... (tk.getChannelGrain)') 
            ##################################################################################### Speical Case for CTX_DEP

            channelGrain = self.Sentence.getChannelGrain(channel, Max_Ngram = Max_Ngram, tagScheme = tagScheme,
                                                         useStartEnd = False, end_grain = end_grain)
            return channelGrain[self.IdxTokenInSent]

    def getGrainTensor(self, channel, Max_Ngram = 1, tagScheme = 'BIO', end_grain = False, TokenNum_Dir = None, 
                       channel_name = None, dontUseLookUp = False):
        
        if channel == 'token':
            if TokenNum_Dir:
                LTU, DTU = self.getGrainUnique(channel, TokenNum_Dir =TokenNum_Dir)
                return [DTU.get(self.token, UNK_ID)], 1
            else:
                return [self.index], 1

        if not channel_name:
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)

        ########################################################################## Speical Case for IND and DEP
        if channel in self.CONTEXT_IND_CHANNELS and not dontUseLookUp:
            ####################################################################### LookUp Table
            try:
                LOOKUP, TokenUnqiue = self.getLookUp(channel_name = channel_name, TokenNum_Dir = TokenNum_Dir)
                LTU, DTU = TokenUnqiue
                index = DTU.get(self.token, UNK_ID) if TokenNum_Dir else self.index
                if index != UNK_ID:
                    info = LOOKUP[index]
                    leng = len(info)
                    return info, leng
                else:
                    # print('\tThough there is a LookUp Table, the token here is UNK...', self.token )
                    pass
            except:
                print('\tNo LookUp Table is found for channel:  ', channel_name, 'Turn to the orignal way... (tk.getGrainTensor)')
            ####################################################################### LookUp Table
                
        elif channel not in self.CONTEXT_IND_CHANNELS and not self._token:
            ####################################################################### In Pyramid
            try:
                info = [self.get_ctx_dep_grain(channel, tagScheme = tagScheme, TokenNum_Dir = TokenNum_Dir)] # based on cls.TOKEN
                return info, 1
            except:
                print('\tIn Pyramid, there is no CTX_DEP channel:', channel, 'Turn to the orignal way... (tk.getGrainTensor)') 
            ####################################################################### In Pyramid
        ########################################################################## Speical Case for IND and DEP


        ########################################################################################################
        # option 2: 
        ### case 1: deal with the case: CTX_IND channel and this channel doesn't have LOOKUP.p or the token is an UNK
        ### case 2: deal with the case: CTX_DEP channel and ANNO_CHANNEL which not in pyramid.
        LGU, DGU = self.getGrainUnique(channel, channel_name = channel_name, TokenNum_Dir =TokenNum_Dir)
        info = [DGU.get(i, UNK_ID) for i in self.getChannelGrain(channel, Max_Ngram = Max_Ngram, 
                                                                 end_grain = end_grain, tagScheme = tagScheme)]
        leng = len(info)
        return info, leng
        ########################################################################################################

    def get_ctx_dep_grain(self, channel, tagScheme, TokenNum_Dir = None):
        Idx = self.Idx
        try:
            return self.CTX_DEP_TMP[channel+tagScheme][Idx]
        except:
            self.Sentence.build_ctx_dep_grain(channel, tagScheme, TokenNum_Dir = TokenNum_Dir, to_tmp = True)
            return self.CTX_DEP_TMP[channel+tagScheme][Idx]


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
        return "<tk "  + self.token + " >"

    