import os
import logging
from bisect import bisect
import numpy as np

from .base import BasicObject
from .token import Token

from .utils.channel import getChannelGrain4Sent, getChannelGrain4Token, getChannelName
from .utils.pyramid import segSent2Tokens, get_line_with_position
from .utils.infrastructure import START, END, START_ID, END_ID, UNK_ID
import re 


START_TK = Token(token = START)
END_TK   = Token(token = END)

class Sentence(BasicObject):
    _sentence   = None
    _tokenLevel = None

    def __init__(self, Idx = None, sentence = None, tokenLevel = None):
        # Idx is the location index
        self.Idx = Idx
        # the sentence must be split with spaces
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
                return ' '.join(self._sentence)
        else:
            return self.get_stored_hyper('token')

    # get_stored methods is applicable for pyramid sentence
    def get_stored_hyper(self, channel):
        start_position, _ = self.start_end_position(channel)
        return get_line_with_position(self.Channel_Hyper_Path[channel], start_position)

    def get_stored_hypertagscheme(self, channel, tagScheme):
        # here channel should exclude token
        grain_idx = re.split(' |\n', self.get_stored_hyper(channel))
        bioes2tag = self.getTrans(channel, tagScheme)
        # shall we check its insanity?
        try:
            return [bioes2tag[vocidx] for vocidx in grain_idx]
        except:
            print(grain_idx)

    def get_stored_hyperstring(self, channel, tagScheme):
        vocidx2grain = self.getGrainVocab(channel, tagScheme = tagScheme)[0]
        grain_idx = self.get_stored_hypertagscheme(channel, tagScheme)
        return [vocidx2grain[vocidx] for vocidx in grain_idx]

    def get_grain_str(self, channel, Min_Ngram = 1, Max_Ngram = 1, end_grain = False, min_grain_freq = 1, tagScheme = 'BIO',  channel_name = None):
        # channelToken = 'ANNOTokenIndex' if 'anno' in  channel else channel + 'TokenIndex'
        
        if channel == 'token':
            channelGrain = [[tk] for tk in self.sentence.split(' ')]
            return channelGrain


        if not channel_name:
            channel_name = getChannelName(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, 
                                          min_grain_freq = min_grain_freq, tagScheme = tagScheme)
        else:
            # print(channel_name)
            channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel = channel, channel_name = channel_name, style = 'extract')


        if not self._sentence and channel in self.Channel_Hyper_Path:
            channelGrain = [[tk] for tk in self.get_stored_hyperstring(channel, tagScheme)]
            return channelGrain

        else:
            # sent = self.sentence.split(' ')
            tokenLevel = self._tokenLevel if self._tokenLevel else self.TOKEN['TOKENLevel']
            # the sentence input is separated by spaces
            # print(tagScheme)
            channelGrain =  getChannelGrain4Sent(self.sentence, channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, 
                                                 end_grain = end_grain, tagScheme  = tagScheme, tokenLevel = tokenLevel)

            return channelGrain

    def get_grain_idx(self, channel, Min_Ngram = 1, Max_Ngram = 1, end_grain = False, min_grain_freq = 1, tagScheme = 'BIO', channel_name = None, 
                      Data_Dir = None,
                      GU = None, TU = None, LKP = None, TRANS = None, unk_id = None):

        '''
            1. Once GU TU LKP and TRANS are settled, channel_name is settled before.
            2. If give us channel_name and Data_Dir, we furtherly determine what GU TU LKP and TRANS are.
        '''

        if channel == 'token':
            # find the TU at the first time
            # assert GU == TU.
            if TU:
                GU = TU
            elif GU:
                TU = GU 
            else:
                if not Data_Dir:
                    TU = self.TokenVocab; GU = TU
                else:
                    TU = self.getGrainVocab('token', Data_Dir = Data_Dir); GU = TU

            LTU, DTU = TU
            if unk_id is None: unk_id = len(DTU)
            info = [[DTU.get(tk, unk_id)] for tk in self.sentence.split(' ')]
            leng_st = len(info)
            leng = [1] * leng_st
            return info, leng_st, leng, 1


        if not channel_name:
            channel_name = getChannelName(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, 
                                          min_grain_freq = min_grain_freq, tagScheme = tagScheme)
        else:
            channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel = channel, channel_name = channel_name, style = 'extract')

        
        if channel in self.Channel_Hyper_Path:
            # deal with hyper fields
            if not GU: GU = self.getGrainVocab(channel, channel_name = channel_name, Data_Dir =Data_Dir)
            LGU, DGU = GU
            if unk_id is None: unk_id = len(DGU)

            if not self._sentence:
                # Generally, we will use this one.
                grain_idx = re.split(' |\n', self.get_stored_hyper(channel))
                TRANS = self.getTrans(channel, tagScheme) if not TRANS else TRANS
                info = [[TRANS[vocidx]] for vocidx in grain_idx]
                leng_st = len(info)
                leng = [1] * leng_st
                return info, leng_st, leng, 1
            else:
                info = self.get_grain_str(channel, channel_name = channel_name)
                # print(info)
                info = [[DGU.get(gr, unk_id) for gr in tk] for tk in info]
                info, leng_st, leng_tk, max_gr = self.padding_info(info)
                return info, leng_st, leng_tk, max_gr

        else: 
            # deal with sub fields
            if not GU: GU = self.getGrainVocab(channel, channel_name = channel_name, Data_Dir =Data_Dir)
            LGU, DGU = GU
            if unk_id is None: unk_id = len(LGU)

            if not (LKP and TU):
                try:
                    LKP, TU = self.getLookUp(channel, channel_name = channel_name, Data_Dir = Data_Dir)
                except:
                    pass

            if (LKP and TU):
                # if we can get LKP and TU
                sentence_tokens = self.sentence.split(' ')
                LTU, DTU = TU
                if unk_id is None: 
                    tk_unk_id = len(DTU)
                else:
                    tk_unk_id = unk_id # NOTICE: we assume that both grain_unkown and token unknown are the same.

                tk_voc_info = [DTU.get(tk, tk_unk_id) for tk in sentence_tokens]
                # print(tk_voc_info)

                info = []
                for idx, tk_voc in enumerate(tk_voc_info):
                    if tk_voc != tk_unk_id:
                        info.append(LKP[tk_voc])
                    else:
                        # deal with the low freq words
                        token_str = sentence_tokens[idx]
                        print(token_str) # TODO: remove this to log
                        grains = getChannelGrain4Token(token_str, channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain)
                        info.append([DGU.get(gr, unk_id) for gr in grains])

                info, leng_st, leng_tk, max_gr = self.padding_info(info)
                return info, leng_st, leng_tk, max_gr
            else:
                # if we cannot get LKP and TU
                info = self.get_grain_str(channel, channel_name = channel_name)
                info = [[DGU.get(gr, unk_id) for gr in tk] for tk in info]
                info, leng_st, leng_tk, max_gr = self.padding_info(info)
                return info, leng_st, leng_tk, max_gr

    
    def padding_info(self, info, leng_tk = None):
        if not leng_tk: leng_tk = [len(i) for i in info]
        leng_st = len(info)
        max_gr  = max(leng_tk)
        info_final =np.zeros([leng_st, max_gr], dtype=int)
        info = [tk + [0] * (max_gr - len(tk)) for tk in info] 
        return info, leng_st, leng_tk, max_gr
        
    @property 
    def IdxCorpus(self):
        return bisect(self.CORPUS['EndIDXFolders'] , self.IdxGroup)

    @property
    def IdxGroup(self):
        return bisect(self.GROUP['EndIDXTexts'] , self.IdxText)

    @property
    def IdxText(self):
        return bisect(self.TEXT['EndIDXSents'] , self.Idx)

    @property
    def IdxTokenStartEnd(self):
        s, e = self.Idx, self.Idx + 1
        s = self.SENT['EndIDXTokens'][s-1] if s != 0 else 0
        e = self.SENT['EndIDXTokens'][e-1] 
        return s, e 

    def start_end_position(self, channel):
        s, e = self.Idx, self.Idx + 1
        s = self.SENT[self.Channel_Hyper_Path[channel]][s-1] if s != 0 else 0
        e = self.SENT[self.Channel_Hyper_Path[channel]][e-1] 
        return s, e

    @property
    def Corpus(self):
        from .corpus import Corpus 
        return Corpus(self.IdxCorpus)

    @property
    def Folder(self):
        from .group import Group 
        return Group(self.IdxGroup)

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
    def length(self):
        if self._sentence:
            length = len(self.sentence.split(' '))
        else:
            s, e = self.IdxTokenStartEnd 
            length = e-s
        return length

    def __repr__(self):
        

        ctx =  str(self.Idx)  if type(self.Idx) == int else 'New'

        return "<st " + ctx + " (tokenNum: " + str(self.length) + ") >"
