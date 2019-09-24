# import pandas as pd
import os
import pickle
from pprint import pprint
from datetime import datetime
import numpy as np
from collections import defaultdict

from .utils.infrastructure import writeGrainList2File, readPickleFile2GrainUnique, specialTokens, UNK_ID
from .utils.infrastructure import modify_wordBoundary_with_hyperBoundary, trans_charLabels_to_wordLabels
from .utils.infrastructure import getTagDict, trans_bioesTag

from .utils.pyramid import CorpusGroupsReader, FolderTextsReaders, segText2Sents, segSent2Tokens
from .utils.anno import getCITText, getCITSents, getSSET_from_CIT
from .utils.vocab import buildTokens, get_GU_or_LKP, get_num_freq

from .utils.channel import CONTEXT_IND_CHANNELS, CONTEXT_DEP_CHANNELS, ANNO_CHANNELS, CHANNEL_ABBR
from .utils.channel import Channel_Dep_Methods, Channel_Dep_TagSets, getChannelName, get_Channel_Settings


# labels and tags
# labels: ['Person', 'Location']
# tags  : ['O', Person-B', 'Person-I', 'Location-B', 'Location-I']

# token, token_idx, token_locidx
# "New",        10,          123
# DTU['New'] = 10
# token(123) = "New", perhaps there is another token_locidx which is also "New".

def utf8len(s):
    return len(s.encode('utf-8'))


class BasicObject(object):
    # basic path
    Base_Dir = None

    # 五大老五奉行
    CORPUS = {}
    GROUP  = {}
    TEXT   = {}
    SENT   = {}
    TOKEN  = {}

    # token and hyper fields
    TokenVocab = None
    VOCAB = {}
    # deal with transfer hyper fields tokens
    TRANS = {}
    # deal with the sub field information
    LOOKUP = {}
    FREQ   = {}

    CTX_DEP_TMP = {}

    # other items
    specialTokens = specialTokens
    CONTEXT_IND_CHANNELS = CONTEXT_IND_CHANNELS
    CONTEXT_DEP_CHANNELS = CONTEXT_DEP_CHANNELS
    ANNO_CHANNELS        = ANNO_CHANNELS
    CHANNEL_ABBR         = CHANNEL_ABBR

    @classmethod
    def INIT(cls, CORPUSPath, 
             Corpus2GroupMethod, 
             Group2TextMethod, 
             Text2SentMethod, 
             Sent2TokenMethod, TOKENLevel, min_token_freq = 1,
             use_hyper = [], Channel_Dep_Methods = Channel_Dep_Methods, Channel_Dep_TagSets = Channel_Dep_TagSets, 
             anno = False, anno_keywords = {}):
        
        Channel_Dep_Methods = {i:k for i,k in Channel_Dep_Methods.items() if i in use_hyper} 
        Channel_Dep_TagSets = {i:k for i,k in Channel_Dep_TagSets.items() if i in use_hyper} 
        
        ################################################################################################################
        corpus_name = CORPUSPath.split('/')[-2]
        Data_Dir = os.path.join('data', corpus_name, TOKENLevel)
        File_Dir = os.path.join(Data_Dir, 'Pyramid', '_file')
        cls.Data_Dir = Data_Dir
        if not os.path.exists(File_Dir):
            os.makedirs(File_Dir)


        cls.Channel_Hyper_Path = {}
        cls.Channel_Hyper_Path['token'] = os.path.join(File_Dir, 'token.txt')
        if os.path.isfile(cls.Channel_Hyper_Path['token']):
            os.remove(cls.Channel_Hyper_Path['token'])

        ################################################################################################################
        # for hyper field
        Path_Key = os.path.join(Data_Dir, 'Vocab')
        cls.VOCAB = {}
        cls.VOCAB[Path_Key] = {}
        for ch, tagSet in Channel_Dep_TagSets.items():
            LGU = getTagDict(tagSet, tagScheme = 'BIOES')
            DGU = dict(zip(LGU, range(len(LGU))))
            channel_name = ch + '-bioes'
            cls.VOCAB[Path_Key][channel_name] = (LGU, DGU)
            cls.Channel_Hyper_Path[ch] =  os.path.join(File_Dir, channel_name + '.txt')
            if os.path.isfile(cls.Channel_Hyper_Path[ch]):
                os.remove(cls.Channel_Hyper_Path[ch])

        # to enrich annoE vocab inside it.
        if anno: 
            cls.VOCAB[Path_Key]['annoE-bioes'] = (['O'], {'O': 0}); anno_labels = []
            cls.Channel_Hyper_Path['annoE'] =  os.path.join(File_Dir, 'annoE-bioes' + '.txt')
            if os.path.isfile(cls.Channel_Hyper_Path['annoE']):
                os.remove(cls.Channel_Hyper_Path['annoE'])

        ################################################################################################################
        # for annotation
        # assert anno == False or '.' in anno or anno == 'embed'
    
        ################################################################################################################
        # init 五大老五奉行
        # corpus information
        CORPUS = {}; CORPUS['CORPUSPath'] = CORPUSPath; CORPUS['Data_Dir'] = Data_Dir; CORPUS['EndIDXGroups'] = []
        
        # group information
        GROUP = {}; GROUP['Corpus2GroupMethod'] = Corpus2GroupMethod 
        GROUP['GroupType']   = 'File' if Corpus2GroupMethod != 'Dir' else 'Dir'; GROUP['group_names'] = []; GROUP['EndIDXTexts'] = []
        
        # text information
        TEXT = {}; TEXT['Group2TextMethod'] = Group2TextMethod; TEXT['EndIDXSents'] = []
        if Group2TextMethod == 'file': TEXT['ORIGFileName'] = []
        if 'ANNOIden' in anno_keywords: TEXT['ANNOFileName'] = []
            
        # sentence information
        # TODO: change these to array to save space
        SENT = {}; SENT['Text2SentMethod'] = Text2SentMethod; SENT['EndIDXTokens'] = np.zeros(5000000, dtype = int)
        # even more file information will be included here. TODO change this to numpy
        for ch, path in cls.Channel_Hyper_Path.items(): SENT[path] = np.zeros(5000000, dtype = int)
        
        # token information
        TOKEN = {}; TOKEN['Sent2TokenMethod'] = Sent2TokenMethod
        TOKEN['TOKENLevel'] = TOKENLevel; TOKEN['Channel_Hyper_Path'] = cls.Channel_Hyper_Path


        # consider how to deal with the annotation information
        ANNO = {}; ANNO['anno'] = anno; ANNO['anno_keywords'] = anno_keywords

        ################################################################################################################
        CorpusGroups, GroupType = CorpusGroupsReader(CORPUSPath, iden = Corpus2GroupMethod)
        # print(CorpusGroups, GroupType)
        assert GROUP['GroupType'] == GroupType
        pprint(GroupType)

        oldDTU = defaultdict(int)# change this to default dict with int
        oldLTU = []
        oldidx2freq = []
        token_num_in_corpus = 0

        total_sent_num = 0
        for group_idx, group_name in enumerate(CorpusGroups):
            # the following block deals with each group in a corpus
            print(group_name)
            text_names = CorpusGroups[group_name]
            GroupTexts = FolderTextsReaders[Group2TextMethod](group_name, text_names, anno, **anno_keywords)
            textIdx = 0
            for strText_SSET_O_A in GroupTexts:
                # the following block deals with each text in a group
                strText, SSETText, origTextName, annoTextName = strText_SSET_O_A
                
                # print(strText, SSETText, origTextName, annoTextName)
                
                strSents = segText2Sents(strText, method = Text2SentMethod) 

                # skip the sents that are empty and the sents which doesn't have annotations.
                # if len(strSents) == 0 or (anno and len(SSETText) == 0): continue
                if len(strSents) == 0: continue

                if anno: tokenizedSents = []
                # the following to blocks deal with multiple hyperfields (include token and annoE) for each strSent in strSents
                # the main input are strText and strSents, (especially strSents)
                
                # block 1: for token and other hyper fields.
                for idx, strSent in enumerate(strSents):
                    # the following block deals with each strSent in a text.
                    # if strTags is not None, strTokens and strTags share the same length, 
                    # and this should be assert inside segSent2Tokens
                    
                    # should assert [' ' not in strToken for strToken in strTokens]
                    # and len(hyper_info) == len(strTokens)

                    strTokens, hyper_info = segSent2Tokens(strSent, Sent2TokenMethod, TOKENLevel, Channel_Dep_Methods)

                    # LESSION: update strSents, which will be used in annotation.
                    if anno: tokenizedSents.append(strTokens)
                    # deal with tokens
                    for token in strTokens:
                        # the following block deals with each token in a text.
                        if token in oldDTU:
                            # deal with old words
                            token_idx = oldDTU[token]
                            oldidx2freq[token_idx] += 1
                        else:
                            # deal with new words
                            token_idx  = len(oldDTU)
                            oldDTU[token] = token_idx
                            oldidx2freq.append(1)
                            oldLTU.append(token)
                        token_num_in_corpus = token_num_in_corpus + 1

                    with open(cls.Channel_Hyper_Path['token'], 'a') as f:
                        line_sentence = ' '.join(strTokens) + '\n'
                        f.write(line_sentence)
                        bytelenSent = utf8len(line_sentence)
                    
                    # total_sent_num increase all the time
                    # if it is 0, then SENT[cls.Channel_Hyper_Path['token']][total_sent_num - 1] will return 0
                    SENT[cls.Channel_Hyper_Path['token']][total_sent_num] = SENT[cls.Channel_Hyper_Path['token']][total_sent_num - 1] + bytelenSent
                    # try:
                    #     # SENT[cls.Channel_Hyper_Path['token']].append(SENT[cls.Channel_Hyper_Path['token']][-1] + bytelenSent)
                    # except:
                    #     SENT[cls.Channel_Hyper_Path['token']][total_sent_num] = bytelenSent

                    # if use word level, make the sure that the return grains are the same.
                    # this is for generating the hyper field information.
                    # before using the hyper field information, we must know their labels before.
                    for ch, ch_grain_sent in hyper_info.items():
                        ch_grain_sent = [str(cls.VOCAB[Path_Key][channel_name][1].get(i)) for i in ch_grain_sent]
                        assert len(ch_grain_sent) == len(strTokens)
                        # or you can create a file, which is LineSentence type file. each line is a sentence.
                        with open(cls.Channel_Hyper_Path[ch], 'a') as f:
                            line_sentence = ' '.join(ch_grain_sent) + '\n'
                            f.write(line_sentence)
                            bytelenSent = utf8len(line_sentence)

                        SENT[cls.Channel_Hyper_Path[ch]][total_sent_num] = SENT[cls.Channel_Hyper_Path[ch]][total_sent_num - 1] + bytelenSent
                        # try:
                        #     SENT[cls.Channel_Hyper_Path[ch]].append(SENT[cls.Channel_Hyper_Path[ch]][-1] + bytelenSent)
                        # except:
                        #     SENT[cls.Channel_Hyper_Path[ch]].append(bytelenSent)

                    # adding the sentence length, i.e., the token number in this sentence.
                    lenSent = len(strTokens)
                    SENT['EndIDXTokens'][total_sent_num] = SENT['EndIDXTokens'][total_sent_num - 1] + lenSent
                    # try:
                    #     SENT['EndIDXTokens'].append(SENT['EndIDXTokens'][-1] + lenSent)
                    # except:
                    #     SENT['EndIDXTokens'].append(lenSent)

                    # add a num to total_sent_num
                    # to process next sentence
                    total_sent_num = total_sent_num + 1


                    # logging the information and epxand the length if necessary
                    if total_sent_num % 100000 == 0:
                        print(total_sent_num, '--', lenSent, datetime.now(), '\t',SENT['EndIDXTokens'][total_sent_num-1],  '\t', len(oldidx2freq), len(oldDTU), len(oldLTU))
                        print('\t', len(SENT['EndIDXTokens']), 'token')
                        for ch, ch_grain_sent in hyper_info.items():
                            print('\t', len(SENT[cls.Channel_Hyper_Path[ch]]), ch)# .append(bytelenSent)

                        if len(SENT['EndIDXTokens']) - total_sent_num <= 100000:
                            print('\tEnlarging the SENT size...')
                            print('\tcurrent leng is:', len(SENT['EndIDXTokens']))
                            SENT[cls.Channel_Hyper_Path['token']] = np.concatenate([SENT[cls.Channel_Hyper_Path['token']], np.zeros(2000000, dtype = int)])
                            SENT['EndIDXTokens'] = np.concatenate([SENT['EndIDXTokens'], np.zeros(2000000, dtype = int)])
                            for ch, ch_grain_sent in hyper_info.items():
                                SENT[cls.Channel_Hyper_Path[ch]] = np.concatenate([SENT[cls.Channel_Hyper_Path[ch]], np.zeros(2000000, dtype = int)])
                            print('\tupdated leng is:', len(SENT['EndIDXTokens']))


                # block1: for annotation.
                if anno: 
                    # if SSETText == []: print('\nThe SSET of this Text is Empty!!!', '\n', strText, '\n')

                    # produce the VOCAB
                    new_labels = set([sset[-1] for sset in SSETText])
                    for label in new_labels:
                        if label not in anno_labels:
                            anno_labels.append(label)
                            for suffix in ['-B', '-I', '-E', '-S']:
                                cls.VOCAB[Path_Key]['annoE-bioes'][1][label + suffix] = len(cls.VOCAB[Path_Key]['annoE-bioes'][1])
                                cls.VOCAB[Path_Key]['annoE-bioes'][0].append(label + suffix)

                    # it will check strText and SSET inside getCITText
                    CITText  = getCITText(strText, SSETText, TOKENLevel) 
                    
                    # get CITSents
                    CITSents = getCITSents(tokenizedSents, CITText, TOKENLevel)
                    
                    # pay attention: do not change total_sent_num
                    last_total_sent_num = total_sent_num - len(CITSents) 
                    for sentIdx, CITSent in enumerate(CITSents):
                        anno_tags = [CITToken[2] for CITToken in CITSent]
                        anno_tags = [str(cls.VOCAB[Path_Key]['annoE-bioes'][1][i]) for i in anno_tags]

                        # make sure anno seq is equal to token seq
                        assert len(anno_tags) == len(tokenizedSents[sentIdx])
                        
                        with open(cls.Channel_Hyper_Path['annoE'], 'a') as f:
                            line_sentence = ' '.join(anno_tags) + '\n'
                            f.write(line_sentence)
                            bytelenSent = utf8len(line_sentence)
                        
                        sent_loc_id = last_total_sent_num + sentIdx
                        SENT[cls.Channel_Hyper_Path['annoE']][sent_loc_id] = SENT[cls.Channel_Hyper_Path['annoE']][sent_loc_id - 1] + bytelenSent
                        # try:
                        #     SENT[cls.Channel_Hyper_Path['annoE']].append(SENT[cls.Channel_Hyper_Path['annoE']][-1] + bytelenSent)
                        # except:
                        #     SENT[cls.Channel_Hyper_Path['annoE']].append(bytelenSent)
                    assert sent_loc_id + 1 == total_sent_num
                    if annoTextName:
                        TEXT['ANNOFileName'].append(annoTextName)
                    
                # deal with the text and its sentences information
                lenText = len(strSents)
                try:
                    TEXT['EndIDXSents'].append(TEXT['EndIDXSents'][-1] + lenText)
                except:
                    TEXT['EndIDXSents'].append(lenText)
                    
                if origTextName:
                    TEXT['ORIGFileName'].append(origTextName)
                
                textIdx = textIdx + 1
                # if textIdx % 100000 == 0:
                #     print(textIdx, '--', len(strSents), len(strText), datetime.now())
                #     print(len(SENT['EndIDXTokens']))
                #     for ch, ch_grain_sent in hyper_info.items():
                #         print(len(SENT[cls.Channel_Hyper_Path[ch]]), ch)# .append(bytelenSent)

            # Back to Folder
            lenGroup = textIdx
            GROUP['group_names'].append(group_name)
            try:
                GROUP['EndIDXTexts'].append(GROUP['EndIDXTexts'][-1] + lenGroup)
            except:
                GROUP['EndIDXTexts'].append(lenGroup)
        
        # if there is an error, folderIdx will be referenced before assignment
        lenCorpus = group_idx + 1 
        CORPUS['EndIDXGroups'] = [lenCorpus]

        ####################################################################################### 
        # print('reorder the frequency ...')
        del oldDTU

        idx2freq = np.sort(oldidx2freq)[::-1]
        newidx2oldidx = np.argsort(oldidx2freq)[::-1]
        del oldidx2freq
        
        # oldidx2newidx = np.zeros(len(newidx2oldidx), dtype= int)
        # for new_idx, old_idx in enumerate(newidx2oldidx):
        #     oldidx2newidx[old_idx] = new_idx
        # del oldidx2newidx

        LTU = []
        for new_idx in range(len(oldLTU)):
            LTU.append(oldLTU[newidx2oldidx[new_idx]])
        del oldLTU
        del newidx2oldidx

        DTU = defaultdict(int)
        for new_idx, token in enumerate(LTU):
            DTU[token] = new_idx
        
        print('Total Num of All    Tokens', token_num_in_corpus)
        print('Total Num of Unique Tokens', len(LTU))

        # load 五大老五奉行, TODO: this can be changed
        CORPUS['EndIDXGroups'] = np.array(CORPUS['EndIDXGroups'])
        CORPUS['length']       = len(CORPUS['EndIDXGroups'])

        GROUP['EndIDXTexts']   = np.array(GROUP['EndIDXTexts'])
        GROUP['length']        = len(GROUP['EndIDXTexts'])

        TEXT['EndIDXSents']    = np.array(TEXT['EndIDXSents'])
        TEXT['length']         = len(TEXT['EndIDXSents'])

        SENT['EndIDXTokens']   = SENT['EndIDXTokens'][:total_sent_num]
        SENT['length']         = len(SENT['EndIDXTokens'])
        for ch, path in cls.Channel_Hyper_Path.items():
            SENT[path] = SENT[path][:total_sent_num]

        TOKEN['length']        = token_num_in_corpus
        
        cls.CORPUS = CORPUS
        cls.GROUP  = GROUP
        cls.TEXT   = TEXT
        cls.SENT   = SENT
        cls.TOKEN  = TOKEN
        
        cls.idx2freq  = idx2freq
        cls.TokenVocab = (LTU, DTU)
        orig_vocab_length = len(cls.TokenVocab[0])
        cls.original_vocab_token_num = orig_vocab_length
        cls.current_vocab_token_num  = orig_vocab_length
        cls.OBJECT_TO_PICKLE()

        max_vocab_token_num = get_num_freq(cls.idx2freq, min_token_freq = min_token_freq)
        cls.min_token_freq = min_token_freq
        # cls.max_vocab_token_num = max_vocab_token_num
        cls._buildGVforToken(max_vocab_token_num, min_token_freq)

        folder = 'F' + str(min_token_freq) 
        Path_Key = os.path.join(cls.Data_Dir, 'Vocab', folder)
        if not os.path.exists(Path_Key):
            os.makedirs(Path_Key)


    @classmethod
    def OBJECT_TO_PICKLE(cls):
        Data_Dir = cls.Data_Dir
        
        ################################################################################
        Pyramid_Dir = os.path.join(Data_Dir, 'Pyramid')
        if not os.path.exists(Pyramid_Dir):
            os.makedirs(Pyramid_Dir)
        d = {'CORPUS': cls.CORPUS, 
             'GROUP':  cls.GROUP, 
             'TEXT':   cls.TEXT, 
             'SENT':   cls.SENT, 
             'TOKEN':  cls.TOKEN}
        for k, v in d.items():
            pickle_path = os.path.join(Pyramid_Dir, k + '.p')
            with open(pickle_path, 'wb') as handle:
                pickle.dump(v, handle, protocol=4 )
                print(k + '\tit is Dumped into file:', pickle_path)
                print(k + '\tthe length of it is   :', v['length'])
        print('*'*40, '\n')

        ################################################################################
        Path_Key = os.path.join(Data_Dir, 'Vocab')
        if not os.path.exists(Path_Key):
            os.makedirs(Path_Key)

        for k, v in cls.VOCAB[Path_Key].items():
            
            pickle_path = os.path.join(Path_Key, k + '.voc')
            with open(pickle_path, 'wb') as handle:
                # v is (LGU, DGU)
                pickle.dump(v, handle, protocol=4)
                print(k + '\tis Dumped into file:', pickle_path)
                print(k + '\tthe length of it is   :', len(v[0]))
            
            channel_name_path = os.path.join(Path_Key, k + '.tsv')
            writeGrainList2File(channel_name_path, v[0])
            print('\t\tWrite to:', channel_name_path)

        ################################################################################ 
        pickle_path = os.path.join(Path_Key, 'token.voc')
        with open(pickle_path, 'wb') as handle:
            # v is (LGU, DGU)
            pickle.dump(cls.TokenVocab, handle, protocol=4)
            print('token   ' + '\tis Dumped into file:', pickle_path)
            print('token   ' + '\tthe length of it is   :', len(cls.TokenVocab[0]))
        
        channel_name_path = os.path.join(Path_Key, 'token.tsv')
        writeGrainList2File(channel_name_path, cls.TokenVocab[0])
        print('\t\tWrite to:', channel_name_path)

        pickle_path = os.path.join(Path_Key, 'token.freq')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(cls.idx2freq, handle, protocol=4)
        print('*'*40)
        
    @classmethod
    def _load_tokenvocab_from_disk(cls):
        ################################################################################
        Path_Key = os.path.join(cls.Data_Dir, 'Vocab')
        pickle_path = os.path.join(Path_Key, 'token.freq')
        with open(pickle_path, 'rb') as handle:
            cls.idx2freq = pickle.load(handle)

        ################################################################################
        cls.VOCAB[Path_Key] = {} if Path_Key not in cls.VOCAB else cls.VOCAB[Path_Key]

        pickle_path = os.path.join(Path_Key, 'token.voc')
        with open(pickle_path, 'rb') as handle:
            cls.TokenVocab = pickle.load(handle)

        orig_vocab_length = len(cls.TokenVocab[0])
        cls.original_vocab_token_num = orig_vocab_length
        cls.current_vocab_token_num  = orig_vocab_length

    @classmethod
    def _buildGVforToken(cls, max_vocab_token_num = None, min_token_freq = 1):
        # both two inputs are valid
        try:
            if min_token_freq == cls.min_token_freq:
                return cls.TokenVocab 
        except:
            cls.min_token_freq = min_token_freq

        if min_token_freq < cls.min_token_freq:
            # reset all token vocab from disk
            cls._load_tokenvocab_from_disk()

        cls.idx2freq = cls.idx2freq[:max_vocab_token_num]
        LTU, DTU = cls.TokenVocab
        LTU = LTU[:max_vocab_token_num]
        DTU = {}
        for newidx in range(max_vocab_token_num):
            DTU[LTU[newidx]] = newidx
        cls.TokenVocab = LTU, DTU
        # print(max_vocab_token_num, min_token_freq)
        cls.current_vocab_token_num = max_vocab_token_num
        cls.min_token_freq = min_token_freq
        return cls.TokenVocab 

    @classmethod
    def INIT_FROM_PICKLE(cls, Data_Dir, min_token_freq = 1):
        assert os.path.exists(Data_Dir) 
        cls.Data_Dir = Data_Dir
        ################################################################################
        layer_names = ['CORPUS', 'GROUP','TEXT', 'SENT','TOKEN' ]
        for layer_name in layer_names:
            pickle_path = os.path.join(Data_Dir, 'Pyramid', layer_name + '.p')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                setattr(cls, layer_name,  v)
                print(layer_name + '\tread from pickle file :', pickle_path)
                print(layer_name + '\tthe length of it is   :', v['length'])
        print('*'*40, '\n')

        cls.Channel_Hyper_Path = cls.TOKEN['Channel_Hyper_Path']
        # idx2freq and TokenVocab
        cls._load_tokenvocab_from_disk()
        max_vocab_token_num = get_num_freq(cls.idx2freq, min_token_freq = min_token_freq)
        cls._buildGVforToken(max_vocab_token_num, min_token_freq)

        folder = 'F' + str(min_token_freq) 
        Path_Key = os.path.join(cls.Data_Dir, 'Vocab', folder)
        if not os.path.exists(Path_Key):
            os.makedirs(Path_Key)

    @classmethod
    def _getGVfromVocab(cls, Path_Key, channel_name):
        return cls.VOCAB[Path_Key][channel_name]

    @classmethod
    def _getGVfromDisk(cls, Path_Key, channel_name):
        cls.VOCAB[Path_Key] = cls.VOCAB[Path_Key] if Path_Key in cls.VOCAB else {}
        channel_name_pickle = os.path.join(Path_Key, channel_name + '.voc')
        GrainVocab = readPickleFile2GrainUnique(channel_name_pickle)
        cls.VOCAB[Path_Key][channel_name] = GrainVocab
        return GrainVocab 

    @classmethod
    def _buildGVforHyper(cls, Path_Key, channel, channel_name):
        # build GV for hyper field
        print('\t\tBuild GrainUnique for channel:', channel_name)
        ############################################# Generate New LGU for CTX_Dep
        channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
        print(channel, Max_Ngram, end_grain, tagScheme)
        
        ch = 'annoE' if 'annoR' == channel else channel
        # this many fall into loops
        # BIOES_GU = cls.getGrainVocab(ch, tagScheme = 'BIOES') # cautions: must get the corresponding base GU.
        try:
            # generall, we hope this can give us results.
            BIOES_GU = cls._getGVfromVocab(Path_Key, ch + '-bioes')
        except:
            BIOES_GU = cls._getGVfromDisk(Path_Key, ch + '-bioes')

        BIOES_GU_neat = BIOES_GU[0]

        LGU_neat = list(set([trans_bioesTag(channel, i, tagScheme) for i in BIOES_GU_neat]))
        LGU_neat.sort()

        LGU = LGU_neat
        DGU = dict(zip(LGU, range(len(LGU))))

        ############################################# New LGU
        GrainVocab = (LGU, DGU)

        cls.VOCAB[Path_Key] = cls.VOCAB[Path_Key] if Path_Key in cls.VOCAB else {}
        cls.VOCAB[Path_Key][channel_name] = GrainVocab
        
        # how about this one?
        
        channel_name_pickle = os.path.join(Path_Key, channel_name + '.voc')
        with open(channel_name_pickle, 'wb') as handle:
            pickle.dump(GrainVocab, handle)
        print('\t\tWrite to:', channel_name_pickle)

        channel_name_path = os.path.join(Path_Key, channel_name + '.tsv')
        writeGrainList2File(channel_name_path, GrainVocab[0])
        print('\t\tWrite to:', channel_name_path)
        return GrainVocab

    @classmethod
    def _buildGVforSub(cls, Path_Key, channel, channel_name, min_token_freq = 1):

        print('\t\tBuild Grain Uniqe and LookUp Table for channel:', channel_name)

        channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
        
        # build GV and LKP
        GrainVocab, LKP, grainidx2freq = get_GU_or_LKP(cls.TokenVocab, cls.idx2freq, channel, Min_Ngram , Max_Ngram , end_grain , min_grain_freq )
        
        # save to VOCAB
        cls.VOCAB[Path_Key] = cls.VOCAB[Path_Key] if Path_Key in cls.VOCAB else {}
        cls.VOCAB[Path_Key][channel_name] = GrainVocab

        # save GV to disk
        channel_name_pickle = os.path.join(Path_Key, channel_name + '.voc')
        with open(channel_name_pickle, 'wb') as handle:
            pickle.dump(GrainVocab, handle)
        print('\t\tWrite to:', channel_name_pickle)
        
        channel_name_path = os.path.join(Path_Key, channel_name + '.tsv')
        writeGrainList2File(channel_name_path, GrainVocab[0])
        print('\t\tWrite to:', channel_name_path)

        # save LKP to LOOKUP
        cls.LOOKUP[Path_Key] = cls.LOOKUP[Path_Key] if Path_Key in cls.LOOKUP else {}
        cls.LOOKUP[Path_Key][channel_name] = LKP
        # save LKP to disk
        pickle_path = os.path.join(Path_Key,  channel_name + '.lkp')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(LKP, handle)
        # assert len(LookUp) == len(cls.TokenVocab[0])
        print('\t\tWrite to:', pickle_path)

        # save LKP to LOOKUP
        cls.FREQ[Path_Key] = cls.FREQ[Path_Key] if Path_Key in cls.FREQ else {}
        cls.FREQ[Path_Key][channel_name] = grainidx2freq
        # save LKP to disk
        pickle_path = os.path.join(Path_Key,  channel_name + '.freq')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(grainidx2freq, handle)
        print('\t\tWrite to:', pickle_path)
       
        return GrainVocab

    @classmethod
    def getGrainVocab(cls, channel, Min_Ngram = 1, Max_Ngram = 1, end_grain = False, tagScheme = 'BIO', min_grain_freq = 1, channel_name = None,
                      min_token_freq = None, Data_Dir = None,  **kwargs):

        # for token, only return itself token vocab
        if channel == 'token':
            if min_token_freq == cls.min_token_freq or not min_token_freq:
                return cls.TokenVocab
            else:
                raise('Error in getting token vocab, only one method is implemented now.')

        # find the Data_Dir
        Data_Dir = cls.Data_Dir if not Data_Dir else Data_Dir

        # build the path key and the channel name
        if not channel_name:
            channel_name = getChannelName(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, 
                                          min_grain_freq = min_grain_freq, tagScheme = tagScheme)
        else:
            channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel = channel, channel_name = channel_name, style = 'extract')

        # build the Path_Key
        if channel not in cls.CONTEXT_IND_CHANNELS:
            Path_Key = os.path.join(Data_Dir, 'Vocab')
        else:
            min_token_freq = cls.min_token_freq if not min_token_freq else min_token_freq
            # it cannot makes a folder
            folder = 'F' + str(min_token_freq) 
            Path_Key = os.path.join(Data_Dir, 'Vocab', folder)

        # try to return the GV
        try:
            # generall, we hope this can give us results.
            return cls._getGVfromVocab(Path_Key, channel_name)
        except:
            try:
                return cls._getGVfromDisk(Path_Key, channel_name)
            except:
                # if it is the cls.Data_Dir, we can build GV; else we cannot.
                if cls.Data_Dir == Data_Dir:
                    # to check whether the channel is hyper or sub
                    if channel not in cls.CONTEXT_IND_CHANNELS:
                        # also include token
                        return cls._buildGVforHyper(Path_Key, channel, channel_name)
                    else:
                        # TODO incorporate max grain num and min grain freq into channel_name in the future
                        return cls._buildGVforSub(Path_Key, channel, channel_name, min_token_freq = min_token_freq)
                else:
                    channel_name_pickle = os.path.join(Path_Key, channel_name + '.voc')
                    print('In', Data_Dir, 'there is no GrainUnqiue for:', channel_name)
                    print('Error in:', channel_name_pickle)

    @classmethod
    def getLookUp(cls, channel = None, Min_Ngram = 1, Max_Ngram = 1, end_grain = False, min_grain_freq = 1, channel_name = None,
                  Data_Dir = None,  min_token_freq = None, **kwargs):

        # find the Data_Dir
        Data_Dir = cls.Data_Dir if not Data_Dir else Data_Dir

        if not channel_name:
            channel_name = getChannelName(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, 
                                          min_grain_freq = min_grain_freq)
        else:
            channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel = channel, channel_name = channel_name, style = 'extract')

        # build the Path_Key
        min_token_freq = cls.min_token_freq if not min_token_freq else min_token_freq
        # it cannot makes a folder
        folder = 'F' + str(min_token_freq) 
        Path_Key = os.path.join(Data_Dir, 'Vocab', folder)

        try:
            # generall, we hope this can give us results.
            LKP = cls.LOOKUP[Path_Key][channel_name]
            TV = cls.getGrainVocab(channel = 'token', Data_Dir = Data_Dir)
            return LKP, TV
        except:
            try:
                path = os.path.join(Path_Key, channel_name + '.lkp')
                # assert os.path.isfile(path)
                with open(path, 'rb') as handle:
                    LKP = pickle.load(handle)
                
                cls.LOOKUP[Path_Key] = cls.LOOKUP[Path_Key] if Path_Key in cls.LOOKUP else {}
                cls.LOOKUP[Path_Key][channel_name] = LKP

                # read from other pyramid is still no valid
                TV = cls.getGrainVocab(channel = 'token', Data_Dir = Data_Dir)
                print('Get LookUp Table for Channel:', channel_name)
                return LKP, TV
            except:
                print('\tIn', Path_Key, 'there is no LookUp Table for:', channel_name)
                print('\tError in:', path)
                return None

    @classmethod
    def getFreq(cls, channel = None, Min_Ngram = 1, Max_Ngram = 1, end_grain = False, channel_name = None,
                  Data_Dir = None, min_grain_freq = 1, min_token_freq = None, **kwargs):

        # find the Data_Dir
        Data_Dir = cls.Data_Dir if not Data_Dir else Data_Dir
        
        if not channel_name:
            channel_name = getChannelName(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, 
                                          min_grain_freq = min_grain_freq)
        else:
            channel, Min_Ngram, Max_Ngram, end_grain, min_grain_freq, tagScheme = getChannelName(channel = channel, channel_name = channel_name, style = 'extract')

        # build the Path_Key
        min_token_freq = cls.min_token_freq if not min_token_freq else min_token_freq
        # it cannot makes a folder
        folder = 'F' + str(min_token_freq) 
        Path_Key = os.path.join(Data_Dir, 'Vocab', folder)

        try:
            # generall, we hope this can give us results.
            Freq = cls.FREQ[Path_Key][channel_name]
            return Freq
        except:
            try:
                path = os.path.join(Path_Key, channel_name + '.freq')
                # assert os.path.isfile(path)
                with open(path, 'rb') as handle:
                    Freq = pickle.load(handle)
                
                cls.FREQ[Path_Key] = cls.FREQ[Path_Key] if Path_Key in cls.FREQ else {}
                cls.FREQ[Path_Key][channel_name] = Freq
                # read from other pyramid is still no valid
                return Freq
            except:
                print('\tIn', Path_Key, 'there is no LookUp Table for:', channel_name)
                print('\tError in:', path)
                return None


    @classmethod
    def getTrans(cls, channel, tagScheme, Data_Dir = None, GU = None):
        # get Information for TRANS
        channel_name = getChannelName(channel, tagScheme = tagScheme)
        Data_Dir = cls.Data_Dir if not Data_Dir else Data_Dir
        
        if not GU: 
            Path_Key = Data_Dir
            GU = cls.getGrainVocab(channel, tagScheme = tagScheme)
        else:
            Path_Key = Data_Dir + '-tmp'

        try:
            return cls.TRANS[Path_Key][channel_name] # be caution, this may occur an issue
        except:
            ch = 'annoE' if 'anno' in  channel else channel
            BIOES_GU = cls.getGrainVocab(ch, tagScheme = "BIOES")
            new_GU   = GU
            BIOES_LGU= BIOES_GU[0]
            new_DGU  = new_GU[1]
            cls.TRANS[Path_Key] = {} if Path_Key not in cls.TRANS else cls.TRANS[Path_Key]
            cls.TRANS[Path_Key][channel_name] = {str(idx): new_DGU[ trans_bioesTag(channel, bioesTag, tagScheme)] 
                                                 for idx, bioesTag in enumerate(BIOES_LGU)}
            return cls.TRANS[Path_Key][channel_name] 

    @classmethod
    def BUILD_GV_LKP(cls, CHANNEL_SETTINGS_TEMPLATE = None):

        cls.CHANNEL_SETTINGS, cls.channels_folderName = get_Channel_Settings(CHANNEL_SETTINGS_TEMPLATE)

        for channel in cls.CHANNEL_SETTINGS:
            print('Deal with the Channel:', channel)
            channel_setting = cls.CHANNEL_SETTINGS[channel]
            Min_Ngram    = channel_setting.get('Min_Ngram', 1)
            Max_Ngram    = channel_setting.get('Max_Ngram', 1)
            end_grain    = channel_setting.get('end_grain', False)
            tagScheme    = channel_setting.get('tagScheme', 'BIO')
            min_grain_freq = channel_setting.get('min_grain_freq', 1)
            print('Current Channel is       ', '\t', channel)
            print('Current Channel Max_Ngram', '\t', Max_Ngram)
            cls.getGrainVocab(channel, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme, min_grain_freq = min_grain_freq)
    

def convert_Char_2_Word_BasicObject(BasicObject, use_channel = 'pos', MaxTokenUnique = False):
    
    CORPUS = BasicObject.CORPUS.copy()
    SENT = BasicObject.SENT.copy()
    SENT['EndIDXTokens'] = []
    TOKENLevel = 'word'
    TOKEN = {}
    TOKEN['TOKENLevel']    = TOKENLevel
    CORPUSPath = CORPUS['CORPUSPath']
    ch = use_channel
    posTokenIndex = BasicObject.TOKEN[ch + 'TokenIndex']
    posGU = BasicObject.getGrainUnique(ch, tagScheme='BIOES')
    posLGU, posDGU = posGU

    if 'ANNOTokenIndex' in BasicObject.TOKEN:
        annoTokenIndex = BasicObject.TOKEN['ANNOTokenIndex']
        annoGU = BasicObject.getGrainUnique('annoE', tagScheme='BIOES')
        annoLGU, annoDGU = annoGU

    pos_all  = []
    anno_all = []
    word_all = []
    
    from .sentence import Sentence
    for i in range(BasicObject.SENT['length']):
        st = Sentence(i)
        sentence = st.sentence
        s, e = st.IdxTokenStartEnd
        pos_sent_index = posTokenIndex[s:e]
        
        pos_sent  = [posLGU[i] for i in pos_sent_index]

        if 'ANNOTokenIndex' in BasicObject.TOKEN:
            anno_sent_index = annoTokenIndex[s:e]
            anno_sent = [annoLGU[i] for i in anno_sent_index]

        if 'ANNOTokenIndex' in BasicObject.TOKEN:
            pos_sent_BIO = modify_wordBoundary_with_hyperBoundary(pos_sent, anno_sent) # convert to BIO
        else:
            pos_sent_BIO = modify_wordBoundary_with_hyperBoundary(pos_sent, pos_sent) # convert to BIO

        new_pos_seq = getSSET_from_CIT(pos_sent_BIO, pos_sent_BIO, join_char='*')
        pos_seq = [posDGU[i[-1]+'-S'] for i in new_pos_seq] # still pos-es
        pos_all.extend(pos_seq)

        if 'ANNOTokenIndex' in BasicObject.TOKEN:
            new_anno_seq = getSSET_from_CIT(anno_sent, pos_sent_BIO, join_char='*')
            anno_seq = [annoDGU[trans_charLabels_to_wordLabels(i[0], join_char = '*')] for i in new_anno_seq] # still annoE-es
            anno_all.extend(anno_seq)

        new_st_seq = getSSET_from_CIT(sentence, pos_sent_BIO, join_char='')
        st_seq = [i[0] for i in new_st_seq]
        word_all.extend(st_seq)

        assert len(st_seq) == len(pos_seq) 
        if 'ANNOTokenIndex' in BasicObject.TOKEN:
            assert len(st_seq) == len(anno_seq)
        lenSent = len(st_seq)
        try:
            SENT['EndIDXTokens'].append(SENT['EndIDXTokens'][-1] + lenSent)
        except:
            SENT['EndIDXTokens'].append(lenSent)
            

    TOKEN['ORIGToken']     = word_all
    TOKEN[ch+'TokenIndex'] = np.array(pos_all)

    if 'ANNOTokenIndex' in BasicObject.TOKEN:
        TOKEN['ANNOTokenIndex']= np.array(anno_all)

    TOKEN['length']        = len(word_all)

    SENT['EndIDXTokens']    = np.array(SENT['EndIDXTokens'])
    SENT['length']          = len(SENT['EndIDXTokens'])

    assert SENT['length'] == BasicObject.SENT['length']

    ########################################################
    print('Total Num of All    Tokens', len(TOKEN['ORIGToken']))
    # TOKEN['ORIGToken'] = list(TOKEN['ORIGToken'])
    # print(datetime.now())
    TOKEN['ORIGTokenIndex'], LTU, DTU =  buildTokens(TOKEN['ORIGToken'], MaxTokenUnique)
    # print(datetime.now())
    del TOKEN['ORIGToken']

    print('Total Num of Unique Tokens', len(LTU))
    corpus_name = CORPUSPath.split('/')[-2]
    TokenNum_Dir = os.path.join('data', corpus_name, TOKENLevel, 'Token'+str(len(LTU)))
    # print(TokenNum_Dir)
    CORPUS['TokenNum_Dir'] = TokenNum_Dir
    if not os.path.exists(TokenNum_Dir):
        os.makedirs(TokenNum_Dir)

    # token GrainUnique (TokenUnique)
    GRAIN_UNI = {}
    GRAIN_UNI[TokenNum_Dir] = {'token': (LTU, DTU)}
    channel_name_path = os.path.join(TokenNum_Dir, 'token.tsv')
    writeGrainList2File(channel_name_path, LTU)
    print('\t\tWrite to:', channel_name_path)

    ########################################################
    if 'ANNOTokenIndex' in BasicObject.TOKEN:
        channel_name = 'annoE-bioes'
        GRAIN_UNI[TokenNum_Dir][channel_name] = annoGU
        LGU = annoGU[0]
        channel_name_path = os.path.join(TokenNum_Dir, channel_name+ '.tsv')
        writeGrainList2File(channel_name_path, LGU)
        print('\t\tWrite to:', channel_name_path)
        # print(LGU)

    ########################################################
    channel_name = 'pos-bioes'
    GRAIN_UNI[TokenNum_Dir][channel_name] = posGU
    LGU = posGU[0]
    channel_name_path = os.path.join(TokenNum_Dir, channel_name+ '.tsv')
    writeGrainList2File(channel_name_path, LGU)
    print('\t\tWrite to:', channel_name_path, '\n')
    # print(LGU)

    BasicObject.CORPUS= CORPUS
    BasicObject.SENT  = SENT
    BasicObject.TOKEN = TOKEN
    BasicObject.TokenUnique  = (LTU, LGU)
    BasicObject.GRAIN_UNI    = GRAIN_UNI
    BasicObject.TokenNum_Dir = TokenNum_Dir
    
    BasicObject.OBJECT_TO_PICKLE()
    
    return BasicObject