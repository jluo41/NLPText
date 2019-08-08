from .base import BasicObject
# from .corpus import Corpus
# from .folder import Folder 
from .text import Text
from .sentence import Sentence
from .token import Token

from bisect import bisect


class Folder(BasicObject):
    def __init__(self, Idx = 0):
        self.Idx = Idx

    @property
    def name(self):
        return self.GROUP['group_names'][self.Idx]

    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.Idx)


    @property
    def IdxTextStartEnd(self):
        s, e = self.Idx, self.Idx + 1
        s = self.GROUP['EndIDXTexts'][s-1] if s != 0 else 0
        e = self.GROUP['EndIDXTexts'][e-1]
        return s, e 

    @property
    def IdxSentStartEnd(self):
        s, e = self.IdxTextStartEnd
        s = self.TEXT['EndIDXSents'][s-1]   if s != 0 else 0
        e = self.TEXT['EndIDXSents'][e-1]
        return s, e 

    @property
    def IdxTokenStartEnd(self):
        s, e = self.IdxSentStartEnd
        s = self.SENT['EndIDXTokens'][s-1]  if s != 0 else 0
        e = self.SENT['EndIDXTokens'][e-1]
        return s, e 

    @property
    def Corpus(self):
        from .corpus import Corpus 
        return Corpus(self.IdxCorpus)
    
    @property
    def Texts(self):
        return [Text(Idx)     for Idx in range(*self.IdxTextStartEnd)]

    @property
    def Sentences(self):
        return [Sentence(Idx) for Idx in range(*self.IdxSentStartEnd)]

    # this seems to handle
    @property
    def Tokens(self):
        return [Token(Idx)    for Idx in range(*self.IdxTokenStartEnd)]


    @property
    def TokensStartEnd(self):
        return sum([st.TokensStartEnd for st in self.Sentences], [])

    def __repr__(self):
        return "<gr " + self.name + ' ' + str(self.Idx) + ">"
