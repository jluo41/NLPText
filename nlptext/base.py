# import pandas as pd
import os
import pickle
from pprint import pprint
from datetime import datetime
import numpy as np

from .utils.infrastructure import writeGrainList2File, readPickleFile2GrainUnique, specialTokens, UNK_ID
from .utils.infrastructure import modify_wordBoundary_with_hyperBoundary, trans_charLabels_to_wordLabels
from .utils.infrastructure import getTagDict, trans_bioesTag

from .utils.pyramid import CorpusGroupsReader, FolderTextsReaders, segText2Sents, segSent2Tokens, getCITText, getCITSents, getSSET_from_CIT

from .utils.vocab import buildTokens, get_GU_or_LKP

from .utils.channel import CONTEXT_IND_CHANNELS, CONTEXT_DEP_CHANNELS, ANNO_CHANNELS, CHANNEL_ABBR
from .utils.channel import Channel_Dep_Methods, Channel_Dep_TagSets, getChannelName, get_Channel_Settings


# TODO: to remove these two items
# MaxTextIdx = False 
# MaxTokenUnique = 3500000

# the default length for the whole corpus
DEFAULT_SENT_LENG = 5000000

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
    # sub fields
    SUB_VOCAB = {}

    # deal with transfer hyper fields tokens
    BIOES_Trans = {}
    CTX_DEP_TMP = {}

    # other items
    specialTokens = specialTokens
    CONTEXT_IND_CHANNELS = CONTEXT_IND_CHANNELS
    CONTEXT_DEP_CHANNELS = CONTEXT_DEP_CHANNELS
    ANNO_CHANNELS        = ANNO_CHANNELS
    CHANNEL_ABBR         = CHANNEL_ABBR

    @classmethod
    def INIT(cls, CORPUSPath, corpusGroupIden, textType, 
             Text2SentMethod, Sent2TokenMethod, TOKENLevel, anno = False, annoKW = {}, 
             use_hyper = False, Channel_Dep_Methods = Channel_Dep_Methods, Channel_Dep_TagSets = Channel_Dep_TagSets, 
              **kwargs):
        
        Channel_Dep_Methods = {} if not use_hyper else Channel_Dep_Methods
        Channel_Dep_TagSets = {} if not use_hyper else Channel_Dep_TagSets
        
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

        cls.VOCAB = {}
        cls.VOCAB[Data_Dir] = {}

        for ch, tagSet in Channel_Dep_TagSets.items():
            LGU = getTagDict(tagSet, tagScheme = 'BIOES')
            DGU = dict(zip(LGU, range(len(LGU))))
            channel_name = ch + '-es'
            cls.VOCAB[Data_Dir][channel_name] = (LGU, DGU)
            
            # don't need to change to pos-es
            cls.Channel_Hyper_Path[ch] =  os.path.join(File_Dir, channel_name + '.txt')
            if os.path.isfile(cls.Channel_Hyper_Path[ch]):
                os.remove(cls.Channel_Hyper_Path[ch])

        ################################################################################################################
        # TODO: set a concrete and correct description for the annotation arguments.
        assert anno == False or '.' in anno or anno == 'embed'
        useSep = Sent2TokenMethod.split('-')[-1] if '-' in Sent2TokenMethod else False
        
        ################################################################################################################
        # init 五大老五奉行
        # corpus information
        CORPUS = {}
        CORPUS['CORPUSPath'] = CORPUSPath
        CORPUS['Data_Dir'] = Data_Dir
        CORPUS['corpusGroupIden'] = corpusGroupIden # None if Dir else
        CORPUS['textType'] = textType
        CORPUS['EndIDXGroups'] = []
        
        # group information
        GROUP = {}
        GROUP['GroupType']   = 'File' if corpusGroupIden else 'Folder'
        GROUP['group_names'] = [] 
        GROUP['EndIDXTexts'] = []
                
        # text information
        TEXT = {}
        TEXT['EndIDXSents'] = []
        TEXT['Text2SentMethod'] = Text2SentMethod
        if textType == 'file': TEXT['ORIGFileName'] = []
        if '.' in str(anno):   TEXT['ANNOFileName'] = []
            
        # sentence information
        SENT = {}
        SENT['Sent2TokenMethod'] = Sent2TokenMethod
        SENT['EndIDXTokens'] = []
        # even more file information will be included here.
        for ch, path in cls.Channel_Hyper_Path.items():
            # change this to numpy 
            SENT[path] = []
        
        # token information
        TOKEN = {}
        TOKEN['TOKENLevel'] = TOKENLevel

        # consider how to deal with the annotation information
        ANNO = {}
        ANNO['anno'] = anno
        ANNO['annoKW'] = annoKW

        ################################################################################################################
        CorpusGroups, GroupType = CorpusGroupsReader(CORPUSPath, iden = corpusGroupIden)
        assert GROUP['GroupType'] == GroupType
        pprint(GroupType)

        ################################################################################################################
        oldDTU = {}# change this to default dict with int
        oldLTU = []
        oldidx2freq = []
        token_num_in_corpus = 0

        for group_idx, group_name in enumerate(CorpusGroups):
            # the following block deals with each group in a corpus
            print(group_name)
            text_names = CorpusGroups[group_name]
            GroupTexts = FolderTextsReaders[textType](group_name, text_names, anno, **annoKW)
            textIdx = 0
            for strText_SSET_O_A in GroupTexts:
                # the following block deals with each text in a group
                strText, SSETText, origTextName, annoTextName = strText_SSET_O_A
                strSents = segText2Sents(strText, method = Text2SentMethod) 

                # skip the sents that are empty and the sents which doesn't have annotations.
                if len(strSents) == 0 or (anno and len(SSETText) == 0): continue
                    
                # the following to blocks deal with multiple hyperfields (include token and annoE) for each strSent in strSents
                # the main input are strText and strSents, (especially strSents)
                
                # block 1: for token and other hyper fields.
                for strSent in strSents:
                    # the following block deals with each strSent in a text.
                    # if strTags is not None, strTokens and strTags share the same length, 
                    # and this should be assert inside segSent2Tokens
                    # strTokens = segSent2Tokens(strSent, method = Sent2TokenMethod)
                    strTokens, hyper_info = segSent2Tokens(strSent, Sent2TokenMethod, TOKENLevel, Channel_Dep_Methods)
                    # deal with tokens
                    for token in strTokens:
                        # the following block deals with each token in a text.
                        if token not in oldDTU:
                            # deal with new words
                            token_idx  = len(oldDTU)
                            oldDTU[token] = token_idx
                            oldidx2freq.append(1)
                            oldLTU.append(token)
                        else:
                            # deal with old words
                            token_idx = oldDTU[token]
                            oldidx2freq[token_idx] += 1
                        token_num_in_corpus = token_num_in_corpus + 1
                    # or you can create a file, which is LineSentence type file. each line is a sentence.

                    with open(cls.Channel_Hyper_Path['token'], 'a') as f:
                        line_sentence = ' '.join(strTokens) + '\n'
                        f.write(line_sentence)
                        bytelenSent = utf8len(line_sentence)
                    try:
                        SENT[cls.Channel_Hyper_Path['token']].append(SENT[cls.Channel_Hyper_Path['token']][-1] + bytelenSent)
                    except:
                        SENT[cls.Channel_Hyper_Path['token']].append(bytelenSent)

                    # if use word level, make the sure that the return grains are the same.
                    # this is for generating the hyper field information.
                    # before using the hyper field information, we must know their labels before.
                    for ch, ch_grain_sent in hyper_info.items():
                        ch_grain_sent = [str(cls.VOCAB[Data_Dir][channel_name][1].get(i)) for i in ch_grain_sent]
                        assert len(ch_grain_sent) == len(strTokens)
                        # or you can create a file, which is LineSentence type file. each line is a sentence.
                        with open(cls.Channel_Hyper_Path[ch], 'a') as f:
                            line_sentence = ' '.join(ch_grain_sent) + '\n'
                            f.write(line_sentence)
                            bytelenSent = utf8len(line_sentence)
                        try:
                            SENT[cls.Channel_Hyper_Path[ch]].append(SENT[cls.Channel_Hyper_Path[ch]][-1] + bytelenSent)
                        except:
                            SENT[cls.Channel_Hyper_Path[ch]].append(bytelenSent)

                    # adding the sentence length, i.e., the token number in this sentence.
                    lenSent = len(strTokens)
                    try:
                        SENT['EndIDXTokens'].append(SENT['EndIDXTokens'][-1] + lenSent)
                    except:
                        SENT['EndIDXTokens'].append(lenSent)

                # block1: for annotation.
                if anno: 
                    if SSETText == []: print('\nThe SSET of this Text is Empty!!!', '\n', strText, '\n')

                    # it will check strText and SSET inside getCITText
                    CITText  = getCITText(strText, SSETText,TOKENLevel) 
                    # get CITSents
                    CITSents = getCITSents(strSents, CITText)
                              
                    for sentIdx, CITSent in enumerate(CITSents):
                        anno_tags = [CITToken[2] for CITToken in CITSent]
                        anno_tags = [cls.VOCAB[Data_Dir]['annoE-es'][1].get(i) for i in anno_tags]
                        with open(cls.Channel_Hyper_Path['annoE'], 'a') as f:
                            line_sentence = ' '.join(anno_tags) + '\n'
                            f.write(line_sentence)
                            bytelenSent = utf8len(line_sentence)
                        try:
                            SENT[cls.Channel_Hyper_Path['annoE']].append(SENT[cls.Channel_Hyper_Path['annoE']][-1] + bytelenSent)
                        except:
                            SENT[cls.Channel_Hyper_Path['annoE']].append(bytelenSent)
                    if annoTextName:
                        TEXT['ANNOFileName'].append(annoTextName)
                    
                lenText = len(strSents)
                try:
                    TEXT['EndIDXSents'].append(TEXT['EndIDXSents'][-1] + lenText)
                except:
                    TEXT['EndIDXSents'].append(lenText)
                    
                if origTextName:
                    TEXT['ORIGFileName'].append(origTextName)
                
                textIdx = textIdx + 1
                if textIdx % 100000 == 0:
                    print(textIdx, '--', len(strSents), len(strText), datetime.now())

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
        idx2freq = np.sort(oldidx2freq)[::-1]
        newidx2oldidx = np.argsort(oldidx2freq)[::-1]
        oldidx2newidx = np.zeros(len(newidx2oldidx), dtype= int)

        for new_idx, old_idx in enumerate(newidx2oldidx):
            oldidx2newidx[old_idx] = new_idx

        LTU = []
        for new_idx in range(len(oldLTU)):
            LTU.append(oldLTU[newidx2oldidx[new_idx]])
        del oldLTU


        DTU = {}
        for new_idx, token in enumerate(LTU):
            DTU[token] = new_idx
        del oldDTU
        
        del oldidx2newidx
        del newidx2oldidx

        # TOKEN['ORIGTokenIndex'] = data
        print('Total Num of All    Tokens', token_num_in_corpus)
        print('Total Num of Unique Tokens', len(LTU))

        # load 五大老五奉行
        CORPUS['EndIDXGroups'] = np.array(CORPUS['EndIDXGroups'], dtype = np.uint32)
        CORPUS['length']       = len(CORPUS['EndIDXGroups'])

        GROUP['EndIDXTexts']   = np.array(GROUP['EndIDXTexts'],  dtype = np.uint32)
        GROUP['length']        = len(GROUP['EndIDXTexts'])

        TEXT['EndIDXSents']    = np.array(TEXT['EndIDXSents'],   dtype = np.uint32)
        TEXT['length']         = len(TEXT['EndIDXSents'])

        SENT['EndIDXTokens']   = np.array(SENT['EndIDXTokens'],  dtype = np.uint32)
        SENT['length']         = len(SENT['EndIDXTokens'])
        for ch, path in cls.Channel_Hyper_Path.items():
            SENT[path] = np.array(SENT[path],  dtype = np.uint32)

        TOKEN['length']        = token_num_in_corpus
        
        cls.CORPUS = CORPUS
        cls.GROUP  = GROUP
        cls.TEXT   = TEXT
        cls.SENT   = SENT
        cls.TOKEN  = TOKEN
        
        cls.idx2freq  = idx2freq
        
        cls.idx2token = LTU
        cls.token2idx = DTU
        cls.TokenVocab = (cls.idx2token, cls.token2idx)
        cls.VOCAB[Data_Dir]['token'] = cls.TokenVocab

        cls.OBJECT_TO_PICKLE()
        

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
        GV_Dir = os.path.join(Data_Dir, 'Vocab')
        if not os.path.exists(GV_Dir):
            os.makedirs(GV_Dir)
        for k, v in cls.VOCAB[Data_Dir].items():
            pickle_path = os.path.join(GV_Dir, k + '.voc')
            with open(pickle_path, 'wb') as handle:
                # v is (LGU, DGU)
                pickle.dump(v, handle, protocol=4)
                print(k + '\tis Dumped into file:', pickle_path)
                print(k + '\tthe length of it is   :', len(v[0]))
            channel_name_path = os.path.join(GV_Dir, k + '.tsv')
            writeGrainList2File(channel_name_path, v[0])
            print('\t\tWrite to:', channel_name_path)

        ################################################################################ 
        pickle_path = os.path.join(GV_Dir, 'token.freq')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(cls.idx2freq, handle, protocol=4)
        print('*'*40)
        

    @classmethod
    def INIT_FROM_PICKLE(cls, Data_Dir, max_vocab_token_num = None, min_token_freq = None):
        assert os.path.exists(Data_Dir) 
        ################################################################################
        layer_names = ['CORPUS', 'GROUP','TEXT', 'SENT','TOKEN' ]
        for layer_name in layer_names:
            pickle_path = os.path.join(Pyramid_Dir, layer_name + '.p')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                setattr(cls, layer_name,  v)
                print(layer_name + '\tread from pickle file :', pickle_path)
                print(layer_name + '\tthe length of it is   :', v['length'])
        print('*'*40, '\n')
        
        ################################################################################
        pickle_path = os.path.join(Data_Dir, 'token.freq')
        with open(pickle_path, 'rb') as handle:
            v = pickle.load(handle)
            cls.idx2freq = v
        
        ################################################################################
        cls.Data_Dir = Data_Dir
        cls.VOCAB[cls.Data_Dir] = {}
        channel_names = ['token'] # only put token in the LGUT
        # Don't need to read all of them.
        for channel_name in channel_names:
            pickle_path = os.path.join(Data_Dir, channel_name + '.voc')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                cls.GRAIN_UNI[cls.TokenNum_Dir][channel_name] = v
                print(channel_name + '\tread from pickle file :', pickle_path)
                print(channel_name + '\tthe length of it is   :', len(v[0]))
        print('*'*40, '\n')
        cls.TokenVocab = cls.VOCAB[Data_Dir]['token'] # (LTU & DTU) 


    @classmethod
    def getGrainVocab(cls, channel, Max_Ngram=1, end_grain = False, tagScheme = 'BIO', channel_name = None,
                      max_vocab_token_num = None, min_token_freq = 1, Data_Dir = None,  **kwargs):

        if not channel_name:
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)


        if Data_Dir:
            # TODO: this is very weird
            assert cls.Data_Dir != Data_Dir 
            # first, identify the channel type, to check whether it is in hyper fields or not.
            if channel not in cls.CONTEXT_IND_CHANNELS:
                try:
                    return cls.VOCAB[Data_Dir][channel_name]
                except:
                    # if not loaded into VOCAB, read it from the disk
                    cls.VOCAB[Data_Dir] = cls.VOCAB[Data_Dir] if Data_Dir in cls.VOCAB else {}
                    channel_name_pickle = os.path.join(Data_Dir, 'Vocab', channel_name + '.voc')
                    try:
                        GrainVocab = readPickleFile2GrainUnique(channel_name_pickle)
                        cls.VOCAB[Data_Dir][channel_name] = GrainVocab
                        return GrainVocab # (LGU, DGU)
                    except:
                        print('In', Data_Dir, 'there is no GrainUnqiue for:', channel_name)
                        print('Error in:', channel_name_pickle)
            else:
                # get correct
                max_vocab_token_num = None, min_token_freq = None
                folder = 'F' + str(min_token_freq) + 'N' + str(max_vocab_token_num)
                SubGrain_Dir = os.path.join(Data_Dir, 'Vocab', folder)
                # TODO: add more modification
                try:
                    return cls.VOCAB[SubGrain_Dir][channel_name]
                except:
                    cls.VOCAB[SubGrain_Dir] = cls.VOCAB[SubGrain_Dir] if SubGrain_Dir in cls.VOCAB else {}
                    channel_name_pickle = os.path.join(SubGrain_Dir, channel_name + '.voc')
                    try:
                        GrainVocab = readPickleFile2GrainUnique(channel_name_pickle)
                        cls.VOCAB[SubGrain_Dir][channel_name] = GrainVocab
                        return GrainUnique # (LGU, DGU)
                    except:
                        print('In', SubGrain_Dir, 'there is no GrainUnqiue for:', channel_name)
                        print('Error in:', channel_name_pickle)
                    
        else:
            # Data_Dir isn't provided.
            Data_Dir = cls.Data_Dir

            if channel not in cls.CONTEXT_IND_CHANNELS:
                
                try:
                    return cls.VOCAB[Data_Dir][channel_name]
                
                except:
                    # if not loaded into VOCAB, read it from the disk
                    cls.VOCAB[Data_Dir] = cls.VOCAB[Data_Dir] if Data_Dir in cls.VOCAB else {}
                    channel_name_pickle = os.path.join(Data_Dir, 'Vocab', channel_name + '.voc')
                    channel_name_path   = os.path.join(Data_Dir, 'Vocab', channel_name + '.tsv')
                    if os.path.isfile(channel_name_pickle):
                        GrainVocab = readPickleFile2GrainUnique(channel_name_pickle)
                        cls.VOCAB[Data_Dir][channel_name] = GrainVocab
                        return GrainVocab # (LGU, DGU)

                    else:
                        # build GV for hyper field
                        print('\t\tBuild GrainUnique for channel:', channel_name)
                        ############################################# Generate New LGU for CTX_Dep
                        channel, Max_Ngram, end_grain, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
                        print(channel, Max_Ngram, end_grain, tagScheme)
                        
                        ch = 'annoE' if 'annoR' == channel else channel
                        BIOES_GU = cls.getGrainVocab(ch, tagScheme = 'BIOES') # cautions: must get the corresponding base GU.
                        BIOES_GU_neat = BIOES_GU[0]

                        LGU_neat = list(set([trans_bioesTag(channel, i, tagScheme) for i in BIOES_GU_neat]))
                        LGU_neat.sort()

                        LGU = LGU_neat
                        DGU = dict(zip(LGU, range(len(LGU))))
                        ############################################# New LGU
                        # DGU = List(LGU)
                        GrainVocab = (LGU, DGU)

                        cls.VOCAB[Data_Dir][channel_name] = GrainVocab
                        # cls.Build_BIOES_Trans(channel, tagScheme, BIOES_GU, GrainUnique)
                        # cls,Build_BIOES_Trans(channel, tagScheme, BIOES_GU, new_GU)

                        # pickle_path = os.path.join(LGU_Dir, channel_name + '.voc')
                        with open(channel_name_pickle, 'wb') as handle:
                            pickle.dump(GrainVocab, handle)
                        print('\t\tWrite to:', channel_name_pickle)

                        writeGrainList2File(channel_name_path, GrainVocab[0])
                        ############################################# New LGU
                        print('\t\tWrite to:', channel_name_path)

                        return GrainVocab

            else:
                max_vocab_token_num = None, min_token_freq = None
                folder = 'F' + str(min_token_freq) + 'N' + str(max_vocab_token_num)
                SubGrain_Dir = os.path.join(Data_Dir, 'Vocab',  folder)

                try:
                    return cls.VOCAB[SubGrain_Dir][channel_name]
                
                except:
                    # if not loaded into VOCAB, read it from the disk
                    cls.VOCAB[SubGrain_Dir] = cls.VOCAB[SubGrain_Dir] if SubGrain_Dir in cls.VOCAB else {}
                    channel_name_pickle = os.path.join(SubGrain_Dir, channel_name + '.voc')
                    channel_name_path   = os.path.join(SubGrain_Dir, channel_name + '.tsv')

                    if os.path.isfile(channel_name_pickle):
                        GrainVocab = readPickleFile2GrainUnique(channel_name_pickle)
                        cls.VOCAB[SubGrain_Dir][channel_name] = GrainVocab
                        return GrainVocab # (LGU, DGU)

                    else:
                        print('\t\tBuild Grain Uniqe and LookUp Table for channel:', channel_name)

                        channel, Max_Ngram, end_grain, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
                        GrainUnique, LookUp  = get_GU_or_LKP(cls.TokenUnique, channel=channel, Max_Ngram = Max_Ngram, end_grain = end_grain,
                                                             max_vocab_token_num = None, min_token_freq = 1,
                                                             max_vocab_grain_num = None, min_grain_freq = 1)
                        ############################################# New LGU
                        # GrainUnique = List(GrainUnique)
                        cls.VOCAB[SubGrain_Dir][channel_name] = GrainVocab

                        with open(channel_name_pickle, 'wb') as handle:
                            pickle.dump(GrainVocab, handle)
                        print('\t\tWrite to:', channel_name_pickle)
                        
                        writeGrainList2File(channel_name_path, GrainVocab[0])
                        print('\t\tWrite to:', channel_name_path)
                        ############################################# New LGU

                        ############################################# New LT
                        cls.LOOKUP[SubGrain_Dir] = cls.LOOKUP[SubGrain_Dir] if SubGrain_Dir in cls.LOOKUP else {}
                        cls.LOOKUP[SubGrain_Dir][channel_name] = LookUp

                        pickle_path = os.path.join(SubGrain_Dir,  channel_name + '.lkp')
                        with open(pickle_path, 'wb') as handle:
                            pickle.dump(LookUp, handle)
                        
                        assert len(LookUp) == len(cls.TokenVocab[0])
                        ############################################# New LT
                        print('\t\tWrite to:', pickle_path)
                       
                        return GrainUnique

    @classmethod
    def getLookUp(cls, channel = None, Max_Ngram = 1, end_grain = False, channel_name = None,
                  Data_Dir = None, max_vocab_token_num = None, min_token_freq = None, **kwargs):
        if not channel_name: 
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain)
        
        if Data_Dir:
            assert cls.Data_Dir != Data_Dir
            try:
                ############################################# ReadFrom LGUDict
                channelLookUp = cls.LOOKUP[TokenNum_Dir][channel_name]
                return channelLookUp, cls.getGrainVocab(channel = 'token', channel_name = 'token', TokenNum_Dir = TokenNum_Dir)
                
            except:
                LOOKUP_Dir = os.path.join(TokenNum_Dir, 'GrainUnique')
                cls.LOOKUP[TokenNum_Dir] = cls.LOOKUP[TokenNum_Dir] if TokenNum_Dir in cls.LOOKUP else {}
                lookup_channel_name_path = os.path.join(LOOKUP_Dir, channel_name + '.lkp')
                assert os.path.isfile(lookup_channel_name_path)

                try:
                    ############################################# ReadFrom TSV or Pickle
                    with open(lookup_channel_name_path, 'rb') as handle:
                        channelLookUp = pickle.load(handle)
                    cls.LOOKUP[TokenNum_Dir][channel_name] = channelLookUp
                    return channelLookUp, cls.getGrainVocab(channel = 'token', channel_name = 'token', TokenNum_Dir = TokenNum_Dir)
                    
                except:
                    print('\tIn', TokenNum_Dir, 'there is no LookUp Table for:', channel_name)
                    print('\tError in:', lookup_channel_name_path)
        else:
            try:
                channelLookUp = cls.LOOKUP[cls.TokenNum_Dir][channel_name]
                return channelLookUp, cls.TokenUnique
                
            except:
                print('Get LookUp Table for Channel:', channel_name)
                TokenNum_Dir = cls.TokenNum_Dir
                cls.LOOKUP[TokenNum_Dir] = cls.LOOKUP[TokenNum_Dir] if TokenNum_Dir in cls.LOOKUP else {}
                LOOKUP_Dir = os.path.join(TokenNum_Dir, 'GrainUnique')
                ############################################# ReadFrom TSV or Pickle
                lookup_channel_name_path = os.path.join(LOOKUP_Dir,  channel_name + '.lkp')
                # assert os.path.isfile(lookup_channel_name_path)
                with open(lookup_channel_name_path, 'rb') as handle:
                    channelLookUp = pickle.load(handle)
                cls.LOOKUP[TokenNum_Dir][channel_name] = channelLookUp
                assert len(channelLookUp) == len(cls.TokenUnique[0])
                return channelLookUp, cls.TokenUnique


             
    @classmethod
    def get_BIOES_Trans(cls, channel, tagScheme, TokenNum_Dir = None, GU = None):
        if GU:
            TokenNum_Dir = 'tmp'
            try:
                return cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] # be caution, this may occur an issue
            except:
                ch = 'annoE' if 'anno' in  channel else channel
                BIOES_GU = cls.getGrainUnique(ch, tagScheme = "BIOES")
                new_GU   = GU
                BIOES_LGU = BIOES_GU[0]
                new_DGU   = new_GU[1]
                cls.BIOES_Trans[TokenNum_Dir] = {} if TokenNum_Dir not in cls.BIOES_Trans else cls.BIOES_Trans[TokenNum_Dir]
                cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] = {idx: new_DGU[trans_bioesTag(channel, bioesTag, tagScheme )] 
                                                                    for idx, bioesTag in enumerate(BIOES_LGU)}
                return cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] 

        if TokenNum_Dir:
            assert TokenNum_Dir != cls.TokenNum_Dir
            try:
                return cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] 
            except:
                ch = 'annoE' if 'anno' in  channel else channel
                BIOES_GU = cls.getGrainUnique(ch, tagScheme = "BIOES")
                new_GU   = cls.getGrainUnique(channel, tagScheme = tagScheme, TokenNum_Dir = TokenNum_Dir)
                BIOES_LGU = BIOES_GU[0]
                new_DGU   = new_GU[1]
                cls.BIOES_Trans[TokenNum_Dir] = {} if TokenNum_Dir not in cls.BIOES_Trans else cls.BIOES_Trans[TokenNum_Dir]
                cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] = {idx: new_DGU[trans_bioesTag(channel, bioesTag, tagScheme )] 
                                                                    for idx, bioesTag in enumerate(BIOES_LGU)}
                return cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] 
        
        else:
            try:
                return cls.BIOES_Trans[cls.TokenNum_Dir][channel+tagScheme] 
            except:
                TokenNum_Dir = cls.TokenNum_Dir
                ch = 'annoE' if 'anno' in  channel else channel
                BIOES_GU = cls.getGrainUnique(ch, tagScheme = "BIOES")
                new_GU   = cls.getGrainUnique(channel, tagScheme = tagScheme)
                # run it any time before ctx_dep channel GU into GRAIN_UNI
                BIOES_LGU = BIOES_GU[0]
                new_DGU   = new_GU[1]

                cls.BIOES_Trans[TokenNum_Dir] = {} if TokenNum_Dir not in cls.BIOES_Trans else cls.BIOES_Trans[TokenNum_Dir]
                cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] = {idx: new_DGU[trans_bioesTag(channel, bioesTag, tagScheme )] 
                                                                    for idx, bioesTag in enumerate(BIOES_LGU)}

                return cls.BIOES_Trans[TokenNum_Dir][channel+tagScheme] 


    @classmethod
    def BUILD_GV_LKP(cls, CHANNEL_SETTINGS_TEMPLATE = None):

        cls.CHANNEL_SETTINGS, cls.channels_folderName = get_Channel_Settings(CHANNEL_SETTINGS_TEMPLATE)

        for channel in cls.CHANNEL_SETTINGS:
            print('Deal with the Channel:', channel)
            channel_setting = cls.CHANNEL_SETTINGS[channel]
            Max_Ngram    = channel_setting.get('Max_Ngram', 1)
            end_grain    = channel_setting.get('end_grain', False)
            tagScheme    = channel_setting.get('tagScheme', 'BIO')
            print('Current Channel is       ', '\t', channel)
            print('Current Channel Max_Ngram', '\t', Max_Ngram)
            cls.getGrainVocab(channel, Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
    

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
        channel_name = 'annoE-es'
        GRAIN_UNI[TokenNum_Dir][channel_name] = annoGU
        LGU = annoGU[0]
        channel_name_path = os.path.join(TokenNum_Dir, channel_name+ '.tsv')
        writeGrainList2File(channel_name_path, LGU)
        print('\t\tWrite to:', channel_name_path)
        # print(LGU)

    ########################################################
    channel_name = 'pos-es'
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