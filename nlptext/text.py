
# update 2018.07.14

########################## text.py #######################

from bisect import bisect

from .base import BasicObject
# from .corpus import Corpus
# from .folder import Folder 
# from .text import Text
from .sentence import Sentence
from .token import Token

from .utils.pyramid import segText2Sents

class Text(BasicObject):

    _text = None

    def __init__(self, Idx = None, text = None):
        self.Idx = Idx
        if text:
            self._text = text

    @property
    def name(self):
        return self.TEXT['ORIGFileName'][self.Idx] if 'ORIGFileName' in self.TEXT else 'TxT' 

    @property
    def text(self):
        if self._text:
            return self._text
        else:
            return ''.join([tk.token for tk in self.Tokens])

    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxFolder)

    @property
    def IdxFolder(self):
        return bisect(self.FOLDER['EndIDXTexts'] , self.Idx)

    @property
    def IdxSentStartEnd(self):
        s, e = self.Idx, self.Idx+1
        s = self.TEXT['EndIDXSents'][s-1] if s != 0 else 0
        e = self.TEXT['EndIDXSents'][e-1]
        return s, e 

    @property
    def IdxTokenStartEnd(self):
        s, e = self.IdxSentStartEnd
        s = self.SENT['EndIDXTokens'][s-1] if s != 0 else 0
        e = self.SENT['EndIDXTokens'][e-1]
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
    def Sentences(self):
        if self._text:
            text = segText2Sents(self._text, method = self.TEXT['Text2SentMethod'])
            return [Sentence(sentence = sent) for sent in text]
        else:
            return [Sentence(Idx) for Idx in range(*self.IdxSentStartEnd)]

    @property
    def Tokens(self):
        if self._text:
            return sum([st.Tokens for st in self.Sentences], [])
        else:
            return [Token(Idx)    for Idx in range(*self.IdxTokenStartEnd)]

    @property
    def TokensStartEnd(self):
        return sum([st.TokensStartEnd for st in self.Sentences], [])
    

    def __repr__(self):
        ctx =  self.name + " " + str(self.Idx)  if type(self.Idx) == int else 'New'
        return "<txt " + ctx + ">"

