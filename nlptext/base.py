# import pandas as pd
import os
import pickle
from pprint import pprint
from datetime import datetime
import numpy as np

from .utils.infrastructure import writeGrainList2File, readPickleFile2GrainUnique, specialTokens, UNK_ID
from .utils.infrastructure import modify_wordBoundary_with_hyperBoundary, trans_charLabels_to_wordLabels
from .utils.infrastructure import getTagDict, trans_bioesTag

from .utils.pyramid import CorpusFoldersReader, FolderTextsReaders, segText2Sents, segSent2Tokens, getCITText, getCITSents, getSSET_from_CIT

from .utils.vocab import buildTokens, get_GU_or_LKP

from .utils.channel import CONTEXT_IND_CHANNELS, CONTEXT_DEP_CHANNELS, ANNO_CHANNELS, CHANNEL_ABBR
from .utils.channel import Channel_Dep_Methods, Channel_Dep_TagSets, getChannelName, get_Channel_Settings


MaxTextIdx = False # TODO
MaxTokenUnique = 3500000

class BasicObject(object):
    CORPUS = {}
    FOLDER = {}
    TEXT   = {}
    SENT   = {}
    TOKEN  = {}
    LOOKUP = {}
    GRAIN_UNI = {}
    TokenNum_Dir = None
    TokenUnique  = None
    specialTokens = specialTokens
    CONTEXT_IND_CHANNELS = CONTEXT_IND_CHANNELS
    CONTEXT_DEP_CHANNELS = CONTEXT_DEP_CHANNELS
    ANNO_CHANNELS        = ANNO_CHANNELS
    CHANNEL_ABBR         = CHANNEL_ABBR
    BIOES_Trans = {}
    CTX_DEP_TMP = {}
    BATCH_INFO = {}

    @classmethod
    def INIT(cls, CORPUSPath, corpusFileIden, textType,Text2SentMethod, Sent2TokenMethod, TOKENLevel, anno, annoKW,  MaxTextIdx = MaxTextIdx, MaxTokenUnique = MaxTokenUnique):
        assert anno == False or '.' in anno or anno == 'embed'
        ########################################################
        useSep = Sent2TokenMethod.split('-')[-1] if '-' in Sent2TokenMethod else False
        Channel_Dep_GrainUnqiue = {}
        if TOKENLevel == 'char':
            # Thing 1
            Sent2TokenMethod = 'iter'
            # Thing 2
            for ch, tagSet in Channel_Dep_TagSets.items():
                LGU = getTagDict(tagSet, tagScheme = 'BIOES')
                DGU = dict(zip(LGU, range(len(LGU))))
                Channel_Dep_GrainUnqiue[ch] = (LGU, DGU)
        ########################################################

        ########################################################
        ################   Things to Save   ####################
        ########################################################
        CORPUS = {}
        CORPUS['CORPUSPath'] = CORPUSPath
        CORPUS['corpusFileIden'] = corpusFileIden # None if Dir else
        CORPUS['CORPUSType']     = 'File' if corpusFileIden else 'Dir'
        CORPUS['textType'] = textType
        CORPUS['EndIDXFolders'] = []

        FOLDER = {}
        FOLDER['folderPaths'] = [] 
        FOLDER['EndIDXTexts'] = []
                
        TEXT = {}
        TEXT['EndIDXSents'] = []
        TEXT['Text2SentMethod'] = Text2SentMethod
        if textType == 'file':
            TEXT['ORIGFileName'] = []
        # if anno:
        if '.' in str(anno):
            TEXT['ANNOFileName'] = []
            
        SENT = {}
        SENT['EndIDXTokens'] = []
        SENT['Sent2TokenMethod'] = Sent2TokenMethod

        TOKEN = {}
        TOKEN['TOKENLevel'] = TOKENLevel
        TOKEN['ORIGToken'] = []
        ORIGToken = []
        if anno:
            TOKEN['ANNOToken'] = []
            ANNOToken = []

        TOKEN_DEP = {}
        for ch, method in Channel_Dep_TagSets.items():
            TOKEN[ch+'TokenIndex'] = []
            TOKEN_DEP[ch+'TokenIndex'] = []

        ANNO = {}
        ANNO['anno'] = anno
        ANNO['annoKW'] = annoKW
        ########################################################

        #######################################################################################CORPUS
        CorpusFolders, CORPUSType = CorpusFoldersReader(CORPUSPath, iden = corpusFileIden)
        assert CORPUS['CORPUSType'] == CORPUSType
        pprint(CORPUSType)

        for folderIdx, folderPath in enumerate(CorpusFolders):
            ####################################################################FOLDER
            print(folderPath)
            fileNames = CorpusFolders[folderPath]
            FolderTexts = FolderTextsReaders[textType](folderPath, fileNames, anno, **annoKW)
            
            textIdx = 0
            for strText_SSET_O_A in FolderTexts:

                ################################################################TEXT
                strText, SSETText, origTextName, annoTextName = strText_SSET_O_A
                strSents = segText2Sents(strText, method = Text2SentMethod, useSep = useSep) # fixed
                
                if len(strSents) == 0 or (anno and len(SSETText) == 0):
                    # print('Empty Anno strText Or Small Text:', strText)
                    continue

                for strSent in strSents:
                    strTokens = segSent2Tokens(strSent, method = Sent2TokenMethod)
                    ORIGToken.extend(strTokens)

                    ###############################################
                    ############# For Certain Channels ############
                    ###############################################
                    if TOKENLevel == 'char' and not useSep:
                        for ch, method in Channel_Dep_Methods.items():
                            ch_grain_sent = method(strSent, tokenLevel = TOKENLevel, useStartEnd = False, tagScheme = 'BIOES')
                            ch_grain_sent = [i[0] for i in ch_grain_sent]
                            ch_grain_sent = [Channel_Dep_GrainUnqiue[ch][1].get(i, UNK_ID) for i in ch_grain_sent]
                            TOKEN_DEP[ch + 'TokenIndex'].extend(ch_grain_sent)
                            assert len(ch_grain_sent) == len(strTokens)
                    ###############################################
                    ###############################################
                    lenSent = len(strTokens)
                    try:
                        SENT['EndIDXTokens'].append(SENT['EndIDXTokens'][-1] + lenSent)
                    except:
                        SENT['EndIDXTokens'].append(lenSent)

                if anno and TOKENLevel == 'char': # TODO
                    #########################################################Anno
                    for sset in SSETText:
                        try:
                            assert sset[0] == strText[sset[1]: sset[2]]
                        except:
                            print(SSETText, '\n', sset, '\n',origTextName, '\n',annoTextName)
                        
                    if SSETText == []:
                        print('\nThe SSET of this Text is Empty!!!', '\n', strText, '\n')
                            
                    ############### PART One: Get CITText ###########
                    CITText  = getCITText(strText, SSETText)
                    #################################################          
                        
                    ############### PART TWO: Get CITSents ##########
                    CITSents = getCITSents(strSents, CITText)
                    #################################################          
                  
                    ############### PART THREE: Get TOKEN['ANNOToken'] ###########
                    for sentIdx, CITSent in enumerate(CITSents):
                        ANNOToken.extend([CITToken[2] for CITToken in CITSent])
                    if annoTextName:
                        TEXT['ANNOFileName'].append(annoTextName)
                    #########################################################Anno

                lenText = len(strSents)
                try:
                    TEXT['EndIDXSents'].append(TEXT['EndIDXSents'][-1] + lenText)
                except:
                    TEXT['EndIDXSents'].append(lenText)
                    
                if origTextName:
                    TEXT['ORIGFileName'].append(origTextName)
                ################################################################TEXT
                

                ################## Log: textIdx in a Folder ################## log
                if MaxTextIdx:
                    if textIdx == MaxTextIdx:
                        break
                textIdx = textIdx + 1
                if not textIdx % 100000:
                    print(textIdx, '--', len(strSents), len(strText), datetime.now())
                if textIdx % 1000000 == 0:
                    if TOKENLevel == 'char':
                        TOKEN['ORIGToken'].append(np.array(ORIGToken)); ORIGToken = []
                        if anno:
                            TOKEN['ANNOToken'].append(np.array(ANNOToken)); ANNOToken = []
                        if TOKENLevel == 'char' and not useSep:
                            for ch, method in Channel_Dep_Methods.items():
                                TOKEN[ch + 'TokenIndex'].append(np.array(TOKEN_DEP[ch + 'TokenIndex']))
                                TOKEN_DEP[ch + 'TokenIndex'] = []
                    else:
                        TOKEN['ORIGToken'].append(ORIGToken); ORIGToken = []
                ############################################################### log

            # Back to Folder
            # lenFolder = textIdx + 1
            lenFolder = textIdx
            FOLDER['folderPaths'].append(folderPath)
            try:
                FOLDER['EndIDXTexts'].append(FOLDER['EndIDXTexts'][-1] + lenFolder)
            except:
                FOLDER['EndIDXTexts'].append(lenFolder)
            ####################################################################FOLDER
        
    
        lenCorpus = folderIdx + 1 # folderIdx might be referenced before assignment
        CORPUS['EndIDXFolders'] = [lenCorpus]
        #######################################################################################CORPUS


        ########################################################
        if TOKENLevel == 'char':
            TOKEN['ORIGToken'].append(np.array(ORIGToken)); del ORIGToken
            TOKEN['ORIGToken'] = np.concatenate(TOKEN['ORIGToken'])
            if anno and TOKENLevel == 'char': 
                TOKEN['ANNOToken'].append(np.array(ANNOToken)); del ANNOToken 
                TOKEN['ANNOToken'] = np.concatenate(TOKEN['ANNOToken'])

            if TOKENLevel == 'char' and not useSep:
                for ch, method in Channel_Dep_Methods.items():
                    TOKEN[ch + 'TokenIndex'].append(np.array(TOKEN_DEP[ch + 'TokenIndex'])); del TOKEN_DEP[ch + 'TokenIndex'] 
                    TOKEN[ch + 'TokenIndex'] = np.concatenate(TOKEN[ch + 'TokenIndex'])
        else:
            TOKEN['ORIGToken'].append(ORIGToken); del ORIGToken
            TOKEN['ORIGToken'] = sum(TOKEN['ORIGToken'], [])
            # TODO
        ########################################################


        ########################################################
        print('Total Num of All    Tokens', len(TOKEN['ORIGToken']))
        # TOKEN['ORIGToken'] = list(TOKEN['ORIGToken'])
        # print(datetime.now())
        TOKEN['ORIGTokenIndex'], LTU, DTU, DTU_freq =  buildTokens(TOKEN['ORIGToken'], MaxTokenUnique)
        # print(datetime.now())
        del TOKEN['ORIGToken']
        
        print('Total Num of Unique Tokens', len(LTU))
        corpus_name = CORPUSPath.split('/')[-2]
        TokenNum_Dir = os.path.join('data', corpus_name, TOKENLevel, 'Token'+str(len(LTU)))

        CORPUS['TokenNum_Dir'] = TokenNum_Dir
        if not os.path.exists(TokenNum_Dir):
            os.makedirs(TokenNum_Dir)

        # token GrainUnique (TokenUnique)
        GRAIN_UNI = {}
        GRAIN_UNI[TokenNum_Dir] = {'token': (LTU, DTU)}
        # channel_name_path = os.path.join(TokenNum_Dir, 'token.tsv')
        # writeGrainList2File(channel_name_path, LTU)
        # print('\t\tWrite to:', channel_name_path)
        ########################################################

        ########################################################
        if anno and TOKENLevel == 'char':
            annoGrainList = list(set(TOKEN['ANNOToken']))
            channel_name = 'annoE-es'
            tagSet = list(set([i.split('-')[0] for i in annoGrainList]))
            LGU = getTagDict(tagSet, tagScheme = 'BIOES')
            DGU = dict(zip(LGU, range(len(LGU))))
            GU = (LGU, DGU)
            GRAIN_UNI[TokenNum_Dir][channel_name] = GU
            print(LGU)
            L = []
            for i in TOKEN['ANNOToken']:
                split = i.split('-')
                if split[0] == 'O':
                    L.append(DGU["O"])
                else:
                    L.append(DGU['-'.join([split[0], split[-1]])])
            TOKEN['ANNOTokenIndex'] = np.array(L, dtype = np.uint32)
            del TOKEN['ANNOToken']
        ########################################################

        ########################################################
        # only BIOES, keep getGrainUnique part.
        if TOKENLevel == 'char' and not useSep:
            for ch, GrainUnique in Channel_Dep_GrainUnqiue.items():
                channel_name = ch + '-es'
                GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
        ########################################################

        CORPUS['EndIDXFolders'] = np.array(CORPUS['EndIDXFolders'], dtype = np.uint32)
        CORPUS['length']        = len(CORPUS['EndIDXFolders'])

        FOLDER['EndIDXTexts']   = np.array(FOLDER['EndIDXTexts'], dtype = np.uint32)
        FOLDER['length']        = len(FOLDER['EndIDXTexts'])

        TEXT['EndIDXSents']     = np.array(TEXT['EndIDXSents'],   dtype = np.uint32)
        TEXT['length']          = len(TEXT['EndIDXSents'])

        SENT['EndIDXTokens']    = np.array(SENT['EndIDXTokens'],  dtype = np.uint32)
        SENT['length']          = len(SENT['EndIDXTokens'])

        TOKEN['length']         = len(TOKEN['ORIGTokenIndex'])
        
        cls.CORPUS = CORPUS
        cls.FOLDER = FOLDER
        cls.TEXT   = TEXT
        cls.SENT   = SENT
        cls.TOKEN  = TOKEN
        cls.TokenUnique = (LTU, DTU)
        cls.DTU_freq = DTU_freq
        cls.GRAIN_UNI = GRAIN_UNI
        cls.TokenNum_Dir = TokenNum_Dir
        cls.OBJECT_TO_PICKLE()

    @classmethod
    def OBJECT_TO_PICKLE(cls):

        TokenNum_Dir = cls.TokenNum_Dir

        ##########################################################
        Pyramid_Dir = os.path.join(TokenNum_Dir, 'Pyramid')
        if not os.path.exists(Pyramid_Dir):
            os.makedirs(Pyramid_Dir)
        d = {'CORPUS': cls.CORPUS, 
             'FOLDER': cls.FOLDER, 
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
        ##########################################################


        ##########################################################
        GU_Dir = os.path.join(TokenNum_Dir, 'GrainUnique')
        if not os.path.exists(GU_Dir):
            os.makedirs(GU_Dir)
        for k, v in cls.GRAIN_UNI[TokenNum_Dir].items():
            pickle_path = os.path.join(GU_Dir, k + '.voc')
            with open(pickle_path, 'wb') as handle:
                # v is (LGU, DGU)
                pickle.dump(v, handle, protocol=4)
                print(k + '\tis Dumped into file:', pickle_path)
                print(k + '\tthe length of it is   :', len(v[0]))
            channel_name_path = os.path.join(GU_Dir, k + '.tsv')
            writeGrainList2File(channel_name_path, v[0])
            print('\t\tWrite to:', channel_name_path)

        pickle_path = os.path.join(GU_Dir, 'token.freq')
        with open(pickle_path, 'wb') as handle:
            # v is (LGU, DGU)
            pickle.dump(cls.DTU_freq, handle, protocol=4)
            # print(k + '\tis Dumped into file:', pickle_path)
            # print(k + '\tthe length of it is   :', len(v[0]))
        print('*'*40)
        ##########################################################

    @classmethod
    def INIT_FROM_PICKLE(cls, Pyramid_Dir, GrainUnique_Dir):
        # TokenNum_Dir = cls.TokenNum_Dir
        assert os.path.exists(Pyramid_Dir) and os.path.exists(GrainUnique_Dir)

        ##########################################################
        # Pyramid_Dir = Path2Pyramid
        # layer_names = [i.replace('.p', '') for i in os.listdir(Pyramid_Dir) if '.p' in i]
        layer_names = ['CORPUS', 'FOLDER','TEXT', 'SENT','TOKEN' ]
        for layer_name in layer_names:
            pickle_path = os.path.join(Pyramid_Dir, layer_name + '.p')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                setattr(cls, layer_name,  v)
                print(layer_name + '\tread from pickle file :', pickle_path)
                print(layer_name + '\tthe length of it is   :', v['length'])
        print('*'*40, '\n')
        ##########################################################

        # print(cls.CORPUS)
        cls.TokenNum_Dir = cls.CORPUS['TokenNum_Dir'].replace('channel', 'data')
        cls.GRAIN_UNI[cls.TokenNum_Dir] = {}

        ##########################################################
        # GrainUnique_Dir = Path2LGUnique
        # channel_names = [i.replace('.p', '') for i in os.listdir(LGU_Dir) if '.p' in i]
        channel_names = ['token'] # only put token in the LGUT
        # Don't need to read all of them.
        for channel_name in channel_names:
            pickle_path = os.path.join(GrainUnique_Dir, channel_name + '.voc')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                cls.GRAIN_UNI[cls.TokenNum_Dir][channel_name] = v
                print(channel_name + '\tread from pickle file :', pickle_path)
                print(channel_name + '\tthe length of it is   :', len(v[0]))
        print('*'*40, '\n')
        ##########################################################
        # cls.TEXT['Text2SentMethod'] = 're'
        cls.TokenUnique = cls.GRAIN_UNI[cls.TokenNum_Dir]['token'] # (LTU & DTU) 


        pickle_path = os.path.join(GrainUnique_Dir, 'token.freq')
        with open(pickle_path, 'rb') as handle:
            v = pickle.load(handle)
            cls.DTU_freq = v

        ##########################################################
        GUDict = cls.GRAIN_UNI[cls.TokenNum_Dir]
        for ch_name, LGU_DGU in GUDict.items():
            channel_name_path = os.path.join(cls.TokenNum_Dir, ch_name+'.tsv')
            if not os.path.isfile(channel_name_path):
                writeGrainList2File(channel_name_path, LGU_DGU[0])
        # print('*'*40, '\n')
        ##########################################################

    @classmethod
    def BUILD_GRAIN_UNI_AND_LOOKUP(cls, CHANNEL_SETTINGS_TEMPLATE = None):

        cls.CHANNEL_SETTINGS, cls.channels_folderName = get_Channel_Settings(CHANNEL_SETTINGS_TEMPLATE)

        for channel in cls.CHANNEL_SETTINGS:

            print('Deal with the Channel:', channel)
            channel_setting = cls.CHANNEL_SETTINGS[channel]
            Max_Ngram    = channel_setting.get('Max_Ngram', 1)
            end_grain    = channel_setting.get('end_grain', False)
            tagScheme    = channel_setting.get('tagScheme', 'BIO')
            print('Current Channel is       ', '\t', channel)
            print('Current Channel Max_Ngram', '\t', Max_Ngram)
            cls.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
            # cls.getLookUp(channel, Max_Ngram, end_grain = end_grain)

    @classmethod
    def getGrainUnique(cls, channel, Max_Ngram=1, end_grain = False, tagScheme = 'BIO', TokenNum_Dir = None, channel_name = None, **kwargs):
        '''
            At most time,
            getGrainUnique doesn't build LGU, it only finds and returns the LGU
            When it cannot find the LGU,
            It will build LGU (and LookUp Table if possible)
            It will save the LGU to the pickle file
            It will save the LookUp to the pickle file.
        '''
        if not channel_name:
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
        
        if TokenNum_Dir:
            ################################################################## Read From Other TokenNum_Dir
            # channel_name_path = os.path.join(TokenNum_Dir, channel_name + '.p')
            # assert os.path.exists(TokenNum_Dir)
            assert cls.TokenNum_Dir != TokenNum_Dir # TODO: this is very weird here
            try:
                ############################################# ReadFrom LGUDict
                return cls.GRAIN_UNI[TokenNum_Dir][channel_name]        # (LGU, DGU)
                ############################################# ReadFrom LGUDict
            except:
                cls.GRAIN_UNI[TokenNum_Dir] = cls.GRAIN_UNI[TokenNum_Dir] if TokenNum_Dir in cls.GRAIN_UNI else {}
                channel_name_pickle = os.path.join(TokenNum_Dir, 'GrainUnique', channel_name + '.voc')
                try:
                    ############################################# ReadFrom TSV or Pickle
                    GrainUnique = readPickleFile2GrainUnique(channel_name_pickle)
                    cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                    return GrainUnique # (LGU, DGU)
                    ############################################# ReadFrom TSV or Pickle
                except:
                    print('In', TokenNum_Dir, 'there is no GrainUnqiue for:', channel_name)
                    print('Error in:', channel_name_pickle)
                ################################################################## Read From Other TokenNum_Dir

        else:
            ################################################################## Read From Current TokenNum_Dir
            TokenNum_Dir = cls.TokenNum_Dir
            try:
                ############################################# ReadFrom LGUDict
                return cls.GRAIN_UNI[TokenNum_Dir][channel_name]
                ############################################# ReadFrom LGUDict
            
            except:
                # if channel_name in cls.GRAIN_UNI[TokenNum_Dir]:
                channel_name_pickle = os.path.join(TokenNum_Dir, 'GrainUnique', channel_name + '.voc')
                channel_name_path = os.path.join(cls.TokenNum_Dir, 'GrainUnique', channel_name + '.tsv')
                if os.path.isfile(channel_name_pickle):
                    ############################################# ReadFrom TSV or Pickle
                    GrainUnique = readPickleFile2GrainUnique(channel_name_pickle)
                    cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                    return GrainUnique
                    ############################################# ReadFrom TSV or Pickle
                else:
                    ############################################# Generate New LGU
                    # If not, generate the GrainUnqiue in the Raw Way.
                    if channel in cls.CONTEXT_IND_CHANNELS:
                        print('\t\tBuild Grain Uniqe and LookUp Table for channel:', channel_name)
                        ############################################# Generate New LGU and LT for CTX_IND

                        channel, Max_Ngram, end_grain, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
                        GrainUnique, LookUp  = get_GU_or_LKP(cls.TokenUnique, channel=channel, Max_Ngram = Max_Ngram, end_grain = end_grain)
                        ############################################# New LGU
                        # GrainUnique = List(GrainUnique)
                        cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique

                        with open(channel_name_pickle, 'wb') as handle:
                            pickle.dump(GrainUnique, handle)
                        print('\t\tWrite to:', channel_name_pickle)
                        
                        writeGrainList2File(channel_name_path, GrainUnique[0])
                        print('\t\tWrite to:', channel_name_path)
                        ############################################# New LGU

                        ############################################# New LT
                        cls.LOOKUP[TokenNum_Dir] = cls.LOOKUP[TokenNum_Dir] if TokenNum_Dir in cls.LOOKUP else {}
                        cls.LOOKUP[TokenNum_Dir][channel_name] = LookUp

                        LOOKUP_Dir = os.path.join(TokenNum_Dir, 'GrainUnique')
                        if not os.path.exists(LOOKUP_Dir):
                            os.makedirs(LOOKUP_Dir)
                        pickle_path = os.path.join(LOOKUP_Dir,  channel_name + '.lkp')
                        with open(pickle_path, 'wb') as handle:
                            pickle.dump(LookUp, handle)
                        
                        assert len(LookUp) == len(cls.TokenUnique[0])
                        ############################################# New LT
                        print('\t\tWrite to:', pickle_path)
                       
                        return GrainUnique
                        ############################################# Generate New LGU and LT for CTX_IND

                    elif channel in cls.CONTEXT_DEP_CHANNELS + cls.ANNO_CHANNELS:
                        print('\t\tBuild GrainUnique for channel:', channel_name)
                        ############################################# Generate New LGU for CTX_Dep
                        channel, Max_Ngram, end_grain, tagScheme = getChannelName(channel, channel_name = channel_name, style = 'extract')
                        print(channel, Max_Ngram, end_grain, tagScheme)
                        ch = 'annoE' if 'annoR' == channel else channel
                        BIOES_GU = cls.getGrainUnique(ch, tagScheme = 'BIOES') # cautions: must get the corresponding base GU.
                        BIOES_GU_neat = BIOES_GU[0][3:]

                        LGU_neat = list(set([trans_bioesTag(channel, i, tagScheme) for i in BIOES_GU_neat]))
                        LGU_neat.sort()
                        # print(LGU)
                        LGU = specialTokens[:-1] + LGU_neat
                        DGU = dict(zip(LGU, range(len(LGU))))
                        ############################################# New LGU
                        # DGU = List(LGU)
                        GrainUnique = (LGU, DGU)

                        cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                        # cls.Build_BIOES_Trans(channel, tagScheme, BIOES_GU, GrainUnique)
                        # cls,Build_BIOES_Trans(channel, tagScheme, BIOES_GU, new_GU)

                        # pickle_path = os.path.join(LGU_Dir, channel_name + '.voc')
                        with open(channel_name_pickle, 'wb') as handle:
                            pickle.dump(GrainUnique, handle)
                        print('\t\tWrite to:', channel_name_pickle)

                        writeGrainList2File(channel_name_path, GrainUnique[0])
                        ############################################# New LGU
                        print('\t\tWrite to:', channel_name_path)

                        return GrainUnique
                        ############################################# Generate New LGU for CTX_Dep
                    else:
                        print('Error in getGrainUnique, cannot get GU for channel:', channel_name)
                ############################################# Generate New LGU

            ################################################################## Read From Current TokenNum_Dir

    @classmethod
    def getLookUp(cls, channel = None, Max_Ngram = 1, end_grain = False, tagScheme = 'BIO', TokenNum_Dir = None,  channel_name = None, **kwargs):
        '''`getLookUp` doesn't build, only finds and returns the LookUp table. It's a fast way to get the Tensors for the CTX_IND channels.'''
        if not channel_name: 
            channel_name = getChannelName(channel, Max_Ngram = Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)
        
        if TokenNum_Dir:
            assert cls.TokenNum_Dir != TokenNum_Dir
            ################################################################## Read From Other TokenNum_Dir
            try:
                ############################################# ReadFrom LGUDict
                channelLookUp = cls.LOOKUP[TokenNum_Dir][channel_name]
                return channelLookUp, cls.getGrainUnique(channel = 'token', channel_name = 'token', TokenNum_Dir = TokenNum_Dir)
                ############################################# ReadFrom LGUDict
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
                    return channelLookUp, cls.getGrainUnique(channel = 'token', channel_name = 'token', TokenNum_Dir = TokenNum_Dir)
                    ############################################# ReadFrom TSV or Pickle
                except:
                    print('\tIn', TokenNum_Dir, 'there is no LookUp Table for:', channel_name)
                    print('\tError in:', lookup_channel_name_path)
            ################################################################## Read From Other TokenNum_Dir
        
        else:
            ################################################################## Read From Current TokenNum_Dir
            try:
                ############################################# ReadFrom LGUDict
                channelLookUp = cls.LOOKUP[cls.TokenNum_Dir][channel_name]
                return channelLookUp, cls.TokenUnique
                ############################################# ReadFrom LGUDict

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
                ############################################# ReadFrom TSV or Pickle
            ################################################################## Read From Current TokenNum_Dir

    #################
    @classmethod
    def get_BIOES_Trans(cls, channel, tagScheme, TokenNum_Dir = None):
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
    def Calculate_Infos(cls, batch_words):
        Pyramid_Dir = os.path.join(cls.TokenNum_Dir, 'Pyramid')
        pickle_path = os.path.join(Pyramid_Dir,  str(batch_words) + '_Info.p')

        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as handle:
                batch_end_st_idx_list, job_no = pickle.load(handle)

            print('Read info from:', pickle_path)
            return batch_end_st_idx_list, job_no

        else:
            sentences_endidx = cls.SENT['EndIDXTokens']
            tokens_vocidx = cls.TOKEN['ORIGTokenIndex']
            
            total_words = len(tokens_vocidx)           
            total_examples  = len(sentences_endidx)

            batch_end_st_idx_list = []
            job_no = 0 # job_num
            while True:
                job_no = job_no + 1
                batch_token_progress = job_no * batch_words  # 

                if batch_token_progress >= total_words:
                    # if touch the bottom, go to the end and terminate the loop
                    batch_end_st_idx_list.append(total_examples)
                    # # This won't work: print('Current batch token number:', sentences_endidx[total_examples]) 
                    # print("Last sentence's end tk loc:", sentences_endidx[total_examples-1])
                    break

                # if not, find the correct end sentence loc_id for this batch
                batch_end_st_idx = np.argmax(sentences_endidx > batch_token_progress)
                batch_end_st_idx_list.append(batch_end_st_idx)
                
                # print('Current batch token number:', sentences_endidx[batch_end_st_idx])
                # print("Last sentence's end tk loc:", sentences_endidx[batch_end_st_idx-1])
            # print(batch_end_st_idx_list, '\n')

            for idx in range(job_no):

                # start and end are batch's start sentence loc_id and end sentence loc_id
                # as python routines, batch is [start, end), left close right open
                start = batch_end_st_idx_list[idx-1] if idx > 0 else 0
                end   = batch_end_st_idx_list[idx]

                # print(start, end)
                # find the start sentence's start token loc_id, and
                # find the end sentence's start token loc_id. (as the end sentence is exluded)
                token_start = sentences_endidx[start-1] if start > 0 else 0
                token_end   = sentences_endidx[end  -1]

                indexes     = tokens_vocidx[token_start:token_end] # dtype = np.uint32
                sentence_idx = np.array([i-token_start for i in sentences_endidx[start: end]], dtype = np.uint32)
                # print('The start and end sent loc_id:', start, end)
                # print('The token start and end loc idx in each batch:', token_start, token_end)
                # print(sentence_idx[-1], len(indexes), '\n')
                
            # print(end == len(sentences_endidx))
            # print(token_end == len(tokens_vocidx))
            with open(pickle_path, 'wb') as handle:
                pickle.dump([batch_end_st_idx_list, job_no], handle, protocol=4)
                print('Write info to:', pickle_path)
            return batch_end_st_idx_list, job_no
    #################      

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


