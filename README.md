```
.
├── __init__.py
├── base.py
├── corpus.py
├── folder.py
├── text.py
├── sentence.py
├── token.py
└── utils
    ├── channel.py
    ├── grain.py
    ├── infrastructure.py
    └── pyramid.py

```


# 1. Pyramid

From Corpus to Groups to Texts to Sentences to Tokens

1. Save COPRUS, GROUP, TEXT, SENT and TOKEN

2. Save token to file/token.txt.

3. Get Token Vocabulary (idx2token, token2index), and token frequency (idx2freq).

# 2. Query

1. Construct a connection across corpus, group, text, sentence and token.

2. Given a sentence Idx (the location index), get the sentence content

3. Given a text Idx (the location index), get the text content.

4. Given a token Idx (the location index), get the token content.

# 3. Annotation

1. By whatever means, from group to texts, for each texts, we need to prepare [strText:str, SSET:[content:str, start:int, end:int, label: str]]. Besides, we need to verify that SSET is valid for strText.

2. Then strText is processed to strSents: [strSent:str].

3. Together with strText and strSents, we use SSET to get the tag sequence for each strSent. The tag is in the BIOES tag scheme. However, instead of saving the tag string, we save the tag idx this time.

4. Save the token to file/annoE_idx.txt. Each line is the tag seqeuence correponding to the line of the original sentence in file/token.txt

5. We also need to query the annotation tag sequence from that file.

6. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['annoE']['BIO']`.


# 4. Hyper Field of Token Granularity (i.e. part-of-speech when word level)

Hyper field of token granularity means the hyper field entity corresponds to exactly one token entity.

For example, a part-of-speech entity corresponds to exactly one word. When the token level is word, part-of-speech is the hyper field of token granularity.

1. This is on the process of sentence to tokens. Input is strSent: str, and output is tokens[token:str] and tags[tag:str], where tag is of BIOES tagging scheme. Similar to annotation, we also only save the tag idx and save all of them into file/pos_idx.txt.

2. We also need to query the annotation tag sequence from that file.

3. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['pos']['BIO']`.


# 5. Hyper Field of Larger Granularity (i.e. named entitiy)

Hyper field whose granularity larger than token means the hyper field entity corresponds to more than (including equal to) one token entity.

1. This is on the process of sentence to tokens. Input is strSent: str, and output is tags[tag:str], where tag is of BIOES tagging scheme. Similar to annotation, we also only save the tag idx and save all of them into file/pos_idx.txt.

2. We also need to query the annotation tag sequence from that file.

3. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['pos']['BIO']`.
