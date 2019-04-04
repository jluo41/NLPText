# import pandas as pd
import os
import pickle
from pprint import pprint
from datetime import datetime
import numpy as np

# INIT
# ## pyramid
# from .utils import CorpusFoldersReader, FolderTextsReaders
# from .utils import reCutText2Sent, segText2Sents, segSent2Tokens# (text, method = 'whole')
# from .utils import getCITText, getCITSents
# from .channel import buildTokens, get_LGU_or_LT
# from .utils import getChannelSettingsAndFolderName

from .utils.pyramid import CorpusFoldersReader, FolderTextsReaders
from .utils.pyramid import segText2Sents, segSent2Tokens, getCITText, getCITSents
from .utils.pyramid import buildTokens, get_LGU_or_LT, getChannelSettingsAndFolderName

# ## channel
# from .utils import getChannelName, getTagDict
# from .channel import CONTEXT_IND_CHANNELS, CONTEXT_DEP_CHANNELS, ANNO_CHANNELS, CHANNEL_ABBR
# from .channel import Channel_Dep_Methods, Channel_Dep_TagSets

from .utils.channel import CONTEXT_IND_CHANNELS, CONTEXT_DEP_CHANNELS, ANNO_CHANNELS, CHANNEL_ABBR
from .utils.channel import getChannelName, getTagDict, Channel_Dep_Methods, Channel_Dep_TagSets


# ## infrastructure
# from .utils import writeGrainList2File, readFile2GrainList, readPickleFile2GrainUnique, List
# from .channel import specialTokens, UNK_ID  

from .utils.infrastructure import writeGrainList2File, readPickleFile2GrainUnique, specialTokens, UNK_ID

########################################################
################## Dataset Description #################
########################################################

########### NER ###########

CORPUSPath = 'dataset/ner/'
corpusFileIden = None
textType   = 'file'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = '.Entity'
annoKW = {
    'sep': '\t',
    'notZeroIndex': 1,
}


########### MedPOS ###########

CORPUSPath = 'dataset/medpos/'
textType   = 'file'
corpusFileIden = None
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = '.UMLSTag'
annoKW = {
    'sep': '\t',
    'notZeroIndex': 0,
}

########### Weibo Test ###########
CORPUSPath = 'dataset/weibotest/'
corpusFileIden = None
textType   = 'file'
Text2SentMethod  = 're'
Sent2TokenMethod = 'sep-\t'
TOKENLevel = 'word'
anno = False
annoKW = {}


########### Wiki ###########
CORPUSPath = 'dataset/wiki/'
corpusFileIden = '.txt'

textType   = 'line'

Text2SentMethod  = 're'
Sent2TokenMethod = 'sep- '
TOKENLevel = 'word'

anno = False
annoKW = {}



########### ResumeNER ###########
CORPUSPath = 'dataset/ResumeNER/'
corpusFileIden = '.bmes'
textType   = 'block'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = 'embed' # TODO
annoKW = {}


########### BOSON ###########
CORPUSPath = 'dataset/boson/'
corpusFileIden = '.txt'
textType   = 'line'
Text2SentMethod  = 're'
Sent2TokenMethod = 'iter'
TOKENLevel = 'char'
anno = 'embed'
annoKW = {}

# for f in *_tag.txt; do mv -- "$f" "${f//_tag.txt/.NER}"; done

assert anno == False or '.' in anno or anno == 'embed'
########################################################

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

    CONTEXT_IND_CHANNELS = CONTEXT_IND_CHANNELS
    CONTEXT_DEP_CHANNELS = CONTEXT_DEP_CHANNELS
    ANNO_CHANNELS        = ANNO_CHANNELS
    CHANNEL_ABBR         = CHANNEL_ABBR


    def __init__(self):
        pass

    @classmethod
    def INIT(cls, CORPUSPath, corpusFileIden, textType, 
             Text2SentMethod, Sent2TokenMethod, TOKENLevel, anno, annoKW, 
             MaxTextIdx = MaxTextIdx, MaxTokenUnique = MaxTokenUnique):
        
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

        ########################################################
        ###################     CHAINS      ####################
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
                    CITText = getCITText(strText, SSETText)
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
        TOKEN['ORIGTokenIndex'], LTU, DTU =  buildTokens(TOKEN['ORIGToken'], MaxTokenUnique)
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
        channel_name_path = os.path.join(TokenNum_Dir, 'token.tsv')
        writeGrainList2File(channel_name_path, LTU)
        print('\t\tWrite to:', channel_name_path)
        ########################################################

        ########################################################
        if anno and TOKENLevel == 'char':
            annoGrainList = list(set(TOKEN['ANNOToken']))
            channel_name = 'annoE-bioes'
            tagSet = list(set([i.split('-')[0] for i in annoGrainList]))
            LGU = getTagDict(tagSet, tagScheme = 'BIOES')
            DGU = dict(zip(LGU, range(len(LGU))))
            GRAIN_UNI[TokenNum_Dir][channel_name] = (LGU, DGU)
            channel_name_path = os.path.join(TokenNum_Dir, channel_name+ '.tsv')
            writeGrainList2File(channel_name_path, LGU)
            print('\t\tWrite to:', channel_name_path)

            print(LGU)
            
            L = []
            for i in TOKEN['ANNOToken']:
                split = i.split('-')
                if split[0] == 'O':
                    L.append(DGU["O"])
                else:
                    L.append(DGU['-'.join([split[0], split[-1]])])
            TOKEN['ANNOTokenIndex'] = np.array(L)
            del TOKEN['ANNOToken']
        ########################################################


        ########################################################
        # only BIOES, keep getGrainUnique part.
        if TOKENLevel == 'char' and not useSep:
            for ch, GrainUnique in Channel_Dep_GrainUnqiue.items():
                channel_name = ch + '-bioes'
                # LGU = List(list_grain_unique)
                # DGU = dict(zip(LGU, range(len(LGU))))
                GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                channel_name_path = os.path.join(TokenNum_Dir, channel_name+ '.tsv')
                writeGrainList2File(channel_name_path, GrainUnique[0])
                print('\t\tWrite to:', channel_name_path)
        ########################################################



        CORPUS['EndIDXFolders'] = np.array(CORPUS['EndIDXFolders'])
        CORPUS['length']        = len(CORPUS['EndIDXFolders'])

        FOLDER['EndIDXTexts']   = np.array(FOLDER['EndIDXTexts'])
        FOLDER['length']        = len(FOLDER['EndIDXTexts'])

        TEXT['EndIDXSents']     = np.array(TEXT['EndIDXSents'])
        TEXT['length']          = len(TEXT['EndIDXSents'])

        SENT['EndIDXTokens']    = np.array(SENT['EndIDXTokens'])
        SENT['length']          = len(SENT['EndIDXTokens'])

        TOKEN['length']         = len(TOKEN['ORIGTokenIndex'])
        
        cls.CORPUS = CORPUS
        cls.FOLDER = FOLDER
        cls.TEXT   = TEXT
        cls.SENT   = SENT
        cls.TOKEN  = TOKEN
        cls.TokenUnique = (LTU, DTU)
        cls.GRAIN_UNI = GRAIN_UNI
        cls.TokenNum_Dir = TokenNum_Dir
        cls.OBJECT_TO_PICKLE()

    @classmethod
    def INIT_FROM_PICKLE(cls, Pyramid_Dir, GrainUnique_Dir):
        # TokenNum_Dir = cls.TokenNum_Dir
        assert os.path.exists(Pyramid_Dir) and os.path.exists(GrainUnique_Dir)
        
        # d = {'CORPUS': cls.CORPUS, 
        #      'FOLDER': cls.FOLDER, 
        #      'TEXT': cls.TEXT, 
        #      'SENT': cls.SENT, 
        #      'TOKEN': cls.TOKEN}

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
        cls.TokenNum_Dir = cls.CORPUS['TokenNum_Dir']
        cls.GRAIN_UNI[cls.TokenNum_Dir] = {}

        ##########################################################
        # GrainUnique_Dir = Path2LGUnique
        # channel_names = [i.replace('.p', '') for i in os.listdir(LGU_Dir) if '.p' in i]
        channel_names = ['token'] # only put token in the LGUT
        # Don't need to read all of them.
        for channel_name in channel_names:
            pickle_path = os.path.join(GrainUnique_Dir, channel_name + '.p')
            with open(pickle_path, 'rb') as handle:
                v = pickle.load(handle)
                cls.GRAIN_UNI[cls.TokenNum_Dir][channel_name] = v
                print(channel_name + '\tread from pickle file :', pickle_path)
                print(channel_name + '\tthe length of it is   :', len(v[0]))
        print('*'*40, '\n')
        ##########################################################
        # cls.TEXT['Text2SentMethod'] = 're'
        cls.TokenUnique = cls.GRAIN_UNI[cls.TokenNum_Dir]['token'] # (LTU & DTU) 

        ##########################################################
        GUDict = cls.GRAIN_UNI[cls.TokenNum_Dir]
        for ch_name, LGU_DGU in GUDict.items():
            channel_name_path = os.path.join(cls.TokenNum_Dir, ch_name+'.tsv')
            if not os.path.isfile(channel_name_path):
                writeGrainList2File(channel_name_path, LGU_DGU[0])
        # print('*'*40, '\n')
        ##########################################################

    @classmethod
    def OBJECT_TO_PICKLE(cls):

        TokenNum_Dir = cls.TokenNum_Dir

        ##########################################################
        Pyramid_Dir = os.path.join(TokenNum_Dir, 'Pyramid')
        if not os.path.exists(Pyramid_Dir):
            os.makedirs(Pyramid_Dir)
        d = {'CORPUS': cls.CORPUS, 
             'FOLDER': cls.FOLDER, 
             'TEXT': cls.TEXT, 
             'SENT': cls.SENT, 
             'TOKEN': cls.TOKEN}
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
            pickle_path = os.path.join(GU_Dir, k + '.p')
            with open(pickle_path, 'wb') as handle:
                # v is (LGU, DGU)
                pickle.dump(v, handle, protocol=4)
                print(k + '\tis Dumped into file:', pickle_path)
                print(k + '\tthe length of it is   :', len(v[0]))
        print('*'*40)
        ##########################################################

    @classmethod
    def BUILD_GRAIN_UNI_AND_LOOKUP(cls, CHANNEL_SETTINGS_TEMPLATE = None):
        ########################################################################## GET CHANNEL_SETTINGS
        # if template_path: 
        #     assert os.path.isfile(template_path)
        #     with open(template_path, 'r') as fp:
        #         CHANNEL_SETTINGS_TEMPLATE = json.load(fp)
        # try:
        #     cls.CHANNEL_SETTINGS = {channel: CHANNEL_SETTINGS_TEMPLATE[channel] for channel in CHANNEL_SETTINGS_TEMPLATE 
        #                             if CHANNEL_SETTINGS_TEMPLATE[channel].pop('use') == True}
        # except:
        #     cls.CHANNEL_SETTINGS = CHANNEL_SETTINGS_TEMPLATE
        ########################################################################## GET CHANNEL_SETTINGS
        cls.CHANNEL_SETTINGS, cls.channels_folderName = getChannelSettingsAndFolderName(CHANNEL_SETTINGS_TEMPLATE)

        for channel in cls.CHANNEL_SETTINGS:

            print('Deal with the Channel:', channel)
            channel_setting = cls.CHANNEL_SETTINGS[channel]
            Max_Ngram    = channel_setting.get('Max_Ngram', 1)
            end_grain    = channel_setting.get('end_grain', False)
            tagScheme    = channel_setting.get('tagScheme', 'BIO')
            print('Current Channel is       ', '\t', channel)
            print('Current Channel Max_Ngram', '\t', Max_Ngram)
            cls.getGrainUnique(channel, Max_Ngram, end_grain = end_grain, tagScheme = tagScheme)

    @classmethod
    def getGrainUnique(cls, channel, Max_Ngram=1, end_grain = False, tagScheme = 'BIO', TokenNum_Dir = None):
        '''
            At most time,
            getGrainUnique doesn't build LGU, it only finds and returns the LGU
            When it cannot find the LGU,
            It will build LGU (and LookUp Table if possible)
            It will save the LGU to the pickle file
            It will save the LookUp to the pickle file.
        '''
        channel_name = getChannelName(channel, Max_Ngram, tagScheme, end_grain)
        
        if TokenNum_Dir:
            ################################################################## Read From Other TokenNum_Dir
            channel_name_pickle = os.path.join(TokenNum_Dir, 'GrainUnique', channel_name + '.p')
            # channel_name_path = os.path.join(TokenNum_Dir, channel_name + '.p')

            assert os.path.exists(TokenNum_Dir) and cls.TokenNum_Dir != TokenNum_Dir
            cls.GRAIN_UNI[TokenNum_Dir] = cls.GRAIN_UNI[TokenNum_Dir] if TokenNum_Dir in cls.GRAIN_UNI else {}
            
            if channel_name in cls.GRAIN_UNI[TokenNum_Dir]:
                ############################################# ReadFrom LGUDict
                GrainUnique = cls.GRAIN_UNI[TokenNum_Dir][channel_name]
                return GrainUnique # (LGU, DGU)
                ############################################# ReadFrom LGUDict

            elif os.path.isfile(channel_name_pickle):
                ############################################# ReadFrom TSV or Pickle
                GrainUnique = readPickleFile2GrainUnique(channel_name_pickle)
                # read from pickle
                cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                return GrainUnique # (LGU, DGU)
                ############################################# ReadFrom TSV or Pickle
            else:
                print('In', TokenNum_Dir, 'there is no', channel_name)
            ################################################################## Read From Other TokenNum_Dir

        else:
            ################################################################## Read From Current TokenNum_Dir
            TokenNum_Dir = cls.TokenNum_Dir
            channel_name_pickle = os.path.join(TokenNum_Dir, 'GrainUnique', channel_name + '.p')
            channel_name_path = os.path.join(cls.TokenNum_Dir, channel_name + '.tsv')
        
            if channel_name in cls.GRAIN_UNI[TokenNum_Dir]:
                ############################################# ReadFrom LGUDict
                return cls.GRAIN_UNI[TokenNum_Dir][channel_name]
                ############################################# ReadFrom LGUDict

            elif os.path.isfile(channel_name_pickle):
                ############################################# ReadFrom TSV or Pickle
                GrainUnique = readPickleFile2GrainUnique(channel_name_pickle)
                # read from pickle
                cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique
                return GrainUnique
                ############################################# ReadFrom TSV or Pickle

            else:
                ############################################# Generate New LGU
                # If not, generate the GrainUnqiue in the Raw Way.
                if channel in cls.CONTEXT_IND_CHANNELS:
                    print('\t\tBuild LGU and LT for channel:', channel_name)

                    ############################################# Generate New LGU and LT for CTX_IND
                    GrainUnique, LookUp  = get_LGU_or_LT(cls.TokenUnique, channel=channel,  
                                                         Max_Ngram = Max_Ngram, end_grain = end_grain)
                    
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

                    LOOKUP_Dir = os.path.join(TokenNum_Dir, 'LookUp')
                    if not os.path.exists(LOOKUP_Dir):
                        os.makedirs(LOOKUP_Dir)
                    pickle_path = os.path.join(LOOKUP_Dir,  channel_name + '.p')
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(LookUp, handle)
                    
                    assert len(LookUp) == len(cls.TokenUnique[0])
                    ############################################# New LT
                    print('\t\tWrite to:', pickle_path)
                   
                    return GrainUnique
                    ############################################# Generate New LGU and LT for CTX_IND

                elif channel in cls.CONTEXT_DEP_CHANNELS + cls.ANNO_CHANNELS:
                    print('\t\tBuild LGU for channel:', channel_name)
                    ############################################# Generate New LGU for CTX_Dep
                    
                    ch = 'annoE' if 'annoR' == channel else channel
                    BIOES_LGU = cls.getGrainUnique(ch, tagScheme = 'BIOES')[0][3:]
                    print(BIOES_LGU)
                    # BIOES_LGU  = cls.GRAIN_UNI[TokenNum_Dir][ch +'-bioes'][3:]
                    if 'S' not in tagScheme and 'E' not in tagScheme:
                        LGU = [i.replace('-S', '-B').replace('-E', '-I')  for i in BIOES_LGU]
                    elif 'S' not in tagScheme:
                        LGU = [i.replace('-S', '-B')  for i in BIOES_LGU]
                    elif 'E' not in tagScheme:
                        LGU = [i.replace('-E', '-I')  for i in BIOES_LGU]
                    else:
                        LGU = BIOES_LGU
                    if channel == 'annoR':
                        LGU = [i.split('-')[-1]  for i in LGU]


                    LGU = list(set(LGU)); LGU.sort()
                    # print(LGU)
                    LGU = specialTokens[:-1] + LGU
                    DGU = dict(zip(LGU, range(len(LGU))))
                    ############################################# New LGU
                    # DGU = List(LGU)
                    GrainUnique = (LGU, DGU)

                    cls.GRAIN_UNI[TokenNum_Dir][channel_name] = GrainUnique

                    # pickle_path = os.path.join(LGU_Dir, channel_name + '.p')
                    with open(channel_name_pickle, 'wb') as handle:
                        pickle.dump(GrainUnique, handle)
                    print('\t\tWrite to:', channel_name_pickle)

                    writeGrainList2File(channel_name_path, GrainUnique[0])
                    ############################################# New LGU
                    print('\t\tWrite to:', channel_name_path)

                    return GrainUnique
                    ############################################# Generate New LGU for CTX_Dep

                ############################################# Generate New LGU

            ################################################################## Read From Current TokenNum_Dir

    @classmethod
    def getLookUp(cls, channel, Max_Ngram, end_grain = False, tagScheme = 'BIO', TokenNum_Dir = None):
        '''
            `getLookUp` doesn't build Lookup Table, it only finds and returns the LookUp table
            `getLookUp` is a faster way to get the Tensors for the CTX_IND channels.
        '''
        ############################################################################### QUALIFY CHANNELS
        if channel not in cls.CONTEXT_IND_CHANNELS or channel == 'token':
            print('The channel "', channel, '" is not a CTX_INP channel. \t NO LOOKUP TABLE!' );return 
        ############################################################################### QUALIFY CHANNELS

        
        channel_name = getChannelName(channel, Max_Ngram, tagScheme, end_grain)
        if TokenNum_Dir:
            ################################################################## Read From Other TokenNum_Dir
            LOOKUP_Dir = os.path.join(TokenNum_Dir, 'LookUp')
            assert os.path.exists(LOOKUP_Dir) and cls.TokenNum_Dir != TokenNum_Dir

            cls.LOOKUP[TokenNum_Dir] = cls.LOOKUP[TokenNum_Dir] if TokenNum_Dir in cls.LOOKUP else {}
            
            
            if 'token' not in cls.GRAIN_UNI[TokenNum_Dir]:
                channel_name_pickle = os.path.join(TokenNum_Dir, 'GrainUnique', 'token.p')
                GrainUnique = readPickleFile2GrainUnique(channel_name_pickle)
                print('\tRead other TokenUnique from:', channel_name_pickle)
                cls.GRAIN_UNI[TokenNum_Dir]['token'] = GrainUnique

            if channel_name in cls.LOOKUP[TokenNum_Dir]:
                ############################################# ReadFrom LGUDict
                channelLookUp = cls.LOOKUP[TokenNum_Dir][channel_name]
                return channelLookUp, cls.GRAIN_UNI[TokenNum_Dir]['token']
                ############################################# ReadFrom LGUDict

            else:
                ############################################# ReadFrom TSV or Pickle
                lookup_channel_name_path = os.path.join(LOOKUP_Dir, channel_name + '.p')
                channelLookUp = []
                assert os.path.isfile(lookup_channel_name_path)
                with open(lookup_channel_name_path, 'rb') as handle:
                    channelLookUp = pickle.load(handle)
                cls.LOOKUP[TokenNum_Dir][channel_name] = channelLookUp
                return channelLookUp, cls.GRAIN_UNI[TokenNum_Dir]['token']
                ############################################# ReadFrom TSV or Pickle

            ################################################################## Read From Other TokenNum_Dir
        
        else:
            ################################################################## Read From Current TokenNum_Dir
            TokenNum_Dir = cls.TokenNum_Dir
            cls.LOOKUP[TokenNum_Dir] = cls.LOOKUP[TokenNum_Dir] if TokenNum_Dir in cls.LOOKUP else {}
            LOOKUP_Dir = os.path.join(TokenNum_Dir, 'LookUp')

            if channel_name in cls.LOOKUP[TokenNum_Dir]:
                ############################################# ReadFrom LGUDict
                channelLookUp = cls.LOOKUP[TokenNum_Dir][channel_name]
                return channelLookUp, cls.GRAIN_UNI[TokenNum_Dir]['token']
                ############################################# ReadFrom LGUDict

            else:
                ############################################# ReadFrom TSV or Pickle
                lookup_channel_name_path = os.path.join(LOOKUP_Dir,  channel_name + '.p')
                # assert os.path.isfile(lookup_channel_name_path)
                with open(lookup_channel_name_path, 'rb') as handle:
                    channelLookUp = pickle.load(handle)
                cls.LOOKUP[TokenNum_Dir][channel_name] = channelLookUp
                assert len(channelLookUp) == len(cls.TokenUnique[0])
                return channelLookUp, cls.GRAIN_UNI[TokenNum_Dir]['token']
                ############################################# ReadFrom TSV or Pickle

            
    