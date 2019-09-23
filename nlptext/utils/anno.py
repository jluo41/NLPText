

##################################################################################################TEXT-ANNO
def getCITText(strText, SSETText, TOKENLevel='char'):
    # assert len(SSETText) > 0 
    if TOKENLevel == 'char':
        for sset in SSETText:
            try:
                assert strText[sset[1]: sset[2]] == sset[0]
            except:
                print('strText:', strText[sset[1] : sset[2]])
                print('SSETText:', sset[0])
        CITAnnoText = []
        for sset in SSETText:
            # BIOES
            strAnno, s, e, tag = sset
            CIT = [[c, s + idx, tag+ '-I']  for idx, c in enumerate(strAnno)]
            CIT[-1][2] = tag + '-E'
            CIT[ 0][2] = tag + '-B'
            if len(CIT) == 1:
                CIT[0][2] = tag + '-S' 
            CITAnnoText.extend(CIT)

        # print(strAnnoText)
        CITText = [[char, idx, 'O'] for idx, char in enumerate(strText)]
        for citAnno in CITAnnoText:
            c, idx, t = citAnno
            assert CITText[idx][0] == c
            CITText[idx] = citAnno

    elif TOKENLevel == 'word':
        strText = strText.split(' ')
        for sset in SSETText:
            try:
                assert ' '.join(strText[sset[1]: sset[2]]) == sset[0]
            except:
                print('strText:', strText[sset[1] : sset[2]])
                print('SSETText:', sset[0])

        CITAnnoText = []
        for sset in SSETText:
            # BIOES
            strAnno, s, e, tag = sset
            # this is important
            strAnno = strAnno.split(' ')
            CIT = [[c, s + idx, tag+ '-I']  for idx, c in enumerate(strAnno)]
            CIT[-1][2] = tag + '-E'
            CIT[ 0][2] = tag + '-B'
            if len(CIT) == 1:
                CIT[0][2] = tag + '-S' 
            CITAnnoText.extend(CIT)

        # print('\nCITAnnoText\n')
        # print(CITAnnoText)
        # print('\n')
        # print(strText)
        # print('\n')
        CITText = [[char, idx, 'O'] for idx, char in enumerate(strText)]
        for citAnno in CITAnnoText:
            c, idx, t = citAnno
            assert CITText[idx][0] == c
            CITText[idx] = citAnno
        # print('                 English CITText')
        # print(CITText)
    return CITText

def getCITSents(tokenizedSents, CITText, TOKENLevel='char'):
    # if TOKENLevel == 'char':

    # if TOKENLevel == 'word':
    #     strSents = [tokenizedSent.split(' ') for tokenizedSent in tokenizedSents]
    # print('\nin strSents\n')
    # print(strSents)
    # print(CITText)
    lenLastSent = 0
    collapse    = 0 # don't need to move 
    CITSents = []
    for tokenizedSent in tokenizedSents:

        CITSent = []
        for sentTokenIdx, c in enumerate(tokenizedSent):
            # sentTokenIdx = txtTokenIdx - lenLastSent - collapse
            txtTokenIdx = sentTokenIdx + lenLastSent + collapse
            cT, _, tT = CITText[txtTokenIdx]
            while c != cT and c != ' ':
                # print(c, cT)
                collapse = collapse + 1
                txtTokenIdx = sentTokenIdx + lenLastSent + collapse
                if txtTokenIdx >= len(CITText):
                    raise ValueError('You cannot find a good token!')
                cT, _, tT = CITText[txtTokenIdx]
            CITSent.append([c, sentTokenIdx, tT])
        lenLastSent = lenLastSent + len(tokenizedSent)
        CITSents.append(CITSent)
    # CITSents
    # Here we get CITSents  
    return CITSents
       
def getSSET_from_CIT(orig_seq, tag_seq, tag_seq_tagScheme = 'BIO', join_char = ''):
    # orig_seq is sentence without start or end
    # tag_seq may have start or end
    if tag_seq[0] == '</start>':
        tag_seq = tag_seq[1:-1]
        
    tagScheme = tag_seq_tagScheme
    if tagScheme == 'BIOES':
        tag_seq = [i.replace('-S', '-B').replace('-E', '-I') for i in tag_seq]
    elif tagScheme == 'BIOE':
        tag_seq = [i.replace('-E', '-I') for i in tag_seq]
    elif tagScheme == 'BIOS':
        tag_seq = [i.replace('-S', '-B') for i in tag_seq]
    elif tagScheme == 'BIO':
        pass
    else:
        print('The tagScheme', tagScheme, 'is not supported yet...')
    
    # use BIO tagScheme
    CIT = list(zip(orig_seq, range(len(orig_seq)), tag_seq))
    taggedCIT = [cit for cit in CIT if cit[2]!= 'O']
    
    startIdx = [idx for idx in range(len(taggedCIT)) if taggedCIT[idx][2][-2:] == '-B']
    startIdx.append(len(taggedCIT))

    entitiesList = []
    for i in range(len(startIdx)-1):
        entityAtom = taggedCIT[startIdx[i]: startIdx[i+1]]
        string = join_char.join([cit[0] for cit in entityAtom])
        start, end = entityAtom[0][1], entityAtom[-1][1] + 1
        tag = entityAtom[0][2].split('-')[0]
        entitiesList.append((string, start, end, tag))

    # if join_char == '*':
    #     a = [set([t.split('-')[0] for t in i[0].split('*')]) for i in entitiesList]
    #     for i in a:
    #         if len(i) > 1:
    #             pprint(list(zip(orig_seq, tag_seq)))
                
    return entitiesList