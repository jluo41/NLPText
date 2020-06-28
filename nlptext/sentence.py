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


# START_TK = Token(token = START)
# END_TK   = Token(token = END)

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
            print('://nlptext.Sentence.get_stored_hypertagscheme', grain_idx)

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
            # instead of using lookup table, it run the function directly.
            # print(tagScheme)
            channelGrain =  getChannelGrain4Sent(self.sentence, channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, 
                                                 end_grain = end_grain, tagScheme  = tagScheme, tokenLevel = tokenLevel)

            return channelGrain

    def get_grain_idx(self, channel,    
                      # series1.
                      Min_Ngram = 1, Max_Ngram = 1, end_grain = False, min_grain_freq = 1, tagScheme = 'BIO', 
                      # series2. -- (channel_name is the abrr for series1)
                      channel_name = None,  
                      # series3. -- (where to get GU, TU, LKP, and TRANS, not frequently used, associated with "series1" and "Corpus")
                      Data_Dir = None,
                      # series4. -- (actually, associated with "series1" and "Corpus")
                      GU = None, TU = None, LKP = None, TRANS = None, 
                      # important items.
                      unk_id = None, tk_unk_id = None, mask_unk_tk_info = False):

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
                    # print('[nlptext.Sentence.get_grain_idx]:', 'get its own GU')
                    TU = self.getGrainVocab('token', Data_Dir = Data_Dir); GU = TU

            LTU, DTU = TU
            flag = False
            if tk_unk_id is None: 
                flag = True
                tk_unk_id = len(DTU)
                
            info = [[DTU.get(tk, tk_unk_id)] for tk in self.sentence.split(' ')]

            # check whether the unk token in current sentence.
            if [tk_unk_id] in info and flag:
                print('[nlptext.Sentence.get_grain_idx]: temporary unk_id for (', channel, ') is:', tk_unk_id)

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
            if not GU: 
                GU = self.getGrainVocab(channel, channel_name = channel_name, Data_Dir =Data_Dir)
            LGU, DGU = GU

            # if unk_id is None: 
            #     print('[nlptext.Sentence.get_grain_idx]: temporary unk_id for (', channel, ') is:', len(unk_id))
            #     unk_id = len(DGU)

            if unk_id is None: 
                unk_id = len(LGU)
                padding_idx = unk_id + 1
            else:
                padding_idx = 0


            if not self._sentence:
                # Generally, we will use this one.
                grain_idx = re.split(' |\n', self.get_stored_hyper(channel))
                TRANS = self.getTrans(channel, tagScheme) if not TRANS else TRANS
                info = [[TRANS[vocidx]] for vocidx in grain_idx]
                leng_st = len(info)
                leng = [1] * leng_st
                return info, leng_st, leng, 1
            else:
                print('[nlptext.Sentence.get_grain_idx]: get hyper fields by functions:', tk_unk_id)
                info = self.get_grain_str(channel, channel_name = channel_name)
                # print(info)
                info = [[DGU.get(gr, unk_id) for gr in tk] for tk in info]
                info, leng_st, leng_tk, max_gr = self.padding_info(info, padding_idx = padding_idx)
                return info, leng_st, leng_tk, max_gr

        elif channel in self.CONTEXT_IND_CHANNELS: 
            # deal with sub fields
            # if GU.
            #   if TU, LKP: 
            #     for token_str in sentence_str.
            #       if token in TU:
            #         token_str ---(TU)-->     token_idx      --------(LKP)-------> [..., gr_idx, ...]
            #
            #       elif token not in TU:  #(OOV)
            #         token_str --(Func)-> [..., gr_str, ...] --(GU, gr_unk_idx)--> [..., gr_idx, ...]
            
            if not GU: 
                GU = self.getGrainVocab(channel, channel_name = channel_name, Data_Dir =Data_Dir)

            LGU, DGU = GU

            flag1 = False
            if unk_id is None: 
                unk_id = len(LGU)
                flag1 = True
                padding_idx = unk_id + 1
            else:
                padding_idx = 0

            if not (LKP and TU):
                try:
                    LKP, TU = self.getLookUp(channel, channel_name = channel_name, Data_Dir = Data_Dir)
                except:
                    pass

            if (LKP and TU):
                # if we can get LKP and TU
                sentence_tokens = self.sentence.split(' ')
                LTU, DTU = TU
                
                flag2 = False
                if tk_unk_id is None: 
                    tk_unk_id = len(DTU)
                    tk_padding_idx = tk_unk_id + 1
                    flag2 = True

                tk_voc_info = [DTU.get(tk, tk_unk_id) for tk in sentence_tokens]

                info = []
                for idx, tk_voc in enumerate(tk_voc_info):
                    if tk_voc != tk_unk_id:
                        info.append(LKP[tk_voc])
                    elif mask_unk_tk_info == False:
                        # deal with the low freq words
                        token_str = sentence_tokens[idx]
                        # print('[nlptext.Sentence.get_grain_idx]: get grain idx for the OOV token: [', token_str, '] for field', channel) 
                        if flag2:
                            print('[nlptext.Sentence.get_grain_idx]: temporary unk_id for ( token ) is:', tk_unk_id)
                        grains = getChannelGrain4Token(token_str, channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain)
                        gr_list = [DGU.get(gr, unk_id) for gr in grains]
                        if unk_id in gr_list and flag1:
                            print('[nlptext.Sentence.get_grain_idx]: temporary unk_id for (', channel, ') is:', unk_id)
                        info.append(gr_list)
                    elif mask_unk_tk_info == True:
                        token_str = sentence_tokens[idx]
                        print('[nlptext.Sentence.get_grain_idx]: get grain idx for the OOV token: [', token_str, '] for field', channel) 
                        print("[nlptext.Sentence.get_grain_idx]: mask the unk token's grain information")
                        info.append([unk_id])
                    else:
                        raise ValueError("[nlptext.Sentence.get_grain_idx]: Error")

                # assert 
                info, leng_st, leng_tk, max_gr = self.padding_info(info, padding_idx = padding_idx)
                return info, leng_st, leng_tk, max_gr

            else:
                # if we cannot get LKP and TU
                info = self.get_grain_str(channel, channel_name = channel_name)
                info = [[DGU.get(gr, unk_id) for gr in tk] for tk in info]
                
                # check have gr_unk_idx or not.
                for gr_list in info:
                    if unk_id in gr_list:
                        print('[nlptext.Sentence.get_grain_idx]/no TU and LKP: temporary unk_id for (', channel, ') is:', unk_id)
                        break

                info, leng_st, leng_tk, max_gr = self.padding_info(info, padding_idx = padding_idx)
                return info, leng_st, leng_tk, max_gr

        else:
            print('[nlptext.Sentence.get_grain_idx]: Errors') 

    def padding_info(self, info, leng_tk = None, padding_idx = 0):
        if not leng_tk: leng_tk = [len(i) for i in info]
        leng_st = len(info)
        max_gr  = max(leng_tk)
        info_final =np.zeros([leng_st, max_gr], dtype=int)
        # TODO: 0 could be the first items.
        # The GU first item should be </pad>.
        info = [tk + [padding_idx] * (max_gr - len(tk)) for tk in info] 
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
        from .folder import Folder 
        return Folder(self.IdxGroup)

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


