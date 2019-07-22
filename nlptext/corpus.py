from .base import BasicObject
# from .corpus import Corpus
from .folder import Folder 
from .text import Text
from .sentence import Sentence
from .token import Token


class Corpus(BasicObject):


    def __init__(self, Idx = 0):
        self.Idx = 0  # Idx always be 0
        

    @property
    def name(self):
        return self.CORPUS['CORPUSPath'].split('/')[-2]


    @property 
    def IdxFolderStartEnd(self):
        s, e = self.Idx, self.Idx + 1
        s = self.CORPUS['EndIDXGroups'][s-1] if s != 0 else 0
        e = self.CORPUS['EndIDXGroups'][e-1]
        return s, e 
    
    @property
    def IdxTextStartEnd(self):
        s, e = self.IdxFolderStartEnd
        s = self.GROUP['EndIDXTexts'][s-1] if s != 0 else 0
        e = self.GROUP['EndIDXTexts'][e-1]
        return s, e 

    @property
    def IdxSentStartEnd(self):
        s, e = self.IdxTextStartEnd
        s = self.TEXT['EndIDXSents'][s-1] if s != 0 else 0
        # print(e-1)
        e = self.TEXT['EndIDXSents'][e-1]
        return s, e 

    @property
    def IdxTokenStartEnd(self):
        s, e = self.IdxSentStartEnd
        s = self.SENT['EndIDXTokens'][s-1] if s != 0 else 0
        e = self.SENT['EndIDXTokens'][e-1]
        return s, e 

    @property
    def Groups(self):
        return [Folder(Idx)   for Idx in range(*self.IdxFolderStartEnd)]
    
    @property
    def Texts(self):
        return [Text(Idx)     for Idx in range(*self.IdxTextStartEnd)]

    @property
    def Sentences(self):
        return [Sentence(Idx) for Idx in range(*self.IdxSentStartEnd)]

    @property
    def Tokens(self):
        return [Token(Idx)    for Idx in range(*self.IdxTokenStartEnd)]

    @property
    def TokensStartEnd(self):
        return sum([st.TokensStartEnd for st in self.Sentences], [])

    def __repr__(self):
        return "<Corpus " + self.name + ">"

