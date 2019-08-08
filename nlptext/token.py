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
    # @property
    # def idx(self):
    #     if self._token:
    #         return self.TokenVocab[1].get(self._token, len(TokenVocab[1]))
    #     else:
    #         # remove this out 
    #         start_position = self.start_position('token')
    #         return self.TOKEN['ORIGTokenIndex'][self.Idx]

    # @property
    # def token(self):
    #     if self._token:
    #         return self._token
    #     else:
    #         # there is another quick method
    #         # logic: get the sentence first, then get the token
    #         return self.TokenUnique[0][self.index] # quick? or slow?

    def start_position(self, channel):
        # the byte idx where this token's sentence starts from.
        # Idx = self.IdxText
        start_position = self.SENT[self.Channel_Hyper_Path[channel]][self.IdxText-1] if self.IdxText != 0 else 0
        return start_position

    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxFolder)

    @property
    def IdxFolder(self):
        return bisect(self.GROUP['EndIDXTexts'] , self.IdxText)

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

    