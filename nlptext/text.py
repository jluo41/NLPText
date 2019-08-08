
########################## text.py #######################
import re
from bisect import bisect

from .base import BasicObject
# from .corpus import Corpus
# from .folder import Folder 
# from .text import Text
from .sentence import Sentence
from .token import Token

from .utils.pyramid import segText2Sents, get_lines_with_position, read_file_chunk_string

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
            start_position, end_position = self.start_end_position('token')
            # s, e = self.IdxSentStartEnd
            # num_lines = e - s 
            return read_file_chunk_string(self.Channel_Hyper_Path['token'], start_position, end_position)

    def get_stored_hyper(self, channel):
        start_position, end_position = self.start_end_position(channel)
        return read_file_chunk_string(self.Channel_Hyper_Path[channel], start_position, end_position)

    def get_stored_hypertagscheme(self, channel, tagScheme):
        # here channel should exclude token
        grain_idx = re.split(' |\n', self.get_stored_hyper(channel))
        bioes2tag = self.getTrans(channel, tagScheme)
        # GV = self.getGrainVocab(channel, tagScheme)
        # shall we check its insanity?
        return [bioes2tag[vocidx] for vocidx in grain_idx]

    def get_stored_hyperstring(self, channel, tagScheme):
        vocidx2grain = self.getGrainVocab(channel, tagScheme = tagScheme)[0]
        grain_idx = self.get_stored_hypertagscheme(channel, tagScheme)
        return [vocidx2grain[vocidx] for vocidx in grain_idx]

    def start_end_position(self, channel):
        startsentIdx, endsentIdx = self.IdxSentStartEnd
        start_position = self.SENT[self.Channel_Hyper_Path[channel]][startsentIdx-1] if startsentIdx != 0 else 0
        end_position   = self.SENT[self.Channel_Hyper_Path[channel]][endsentIdx - 1]
        return start_position, end_position

    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxFolder)

    @property
    def IdxFolder(self):
        return bisect(self.GROUP['EndIDXTexts'] , self.Idx)

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
    def length(self):
        if self._text:
            length = len(self.Tokens)
        else:
            s, e = self.IdxTokenStartEnd 
            length = e - s
        return length
    
    def __repr__(self):
        ctx =  self.name + " " + str(self.Idx)  if type(self.Idx) == int else 'New'
        return "<txt " + ctx + ">"

