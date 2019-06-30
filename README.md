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

1. Save `COPRUS`, `GROUP`, `TEXT`, `SENT` and `TOKEN`

2. Save token to `file/token.txt`.

3. Get Token Vocabulary (`idx2token`, `token2index`), and token frequency (`idx2freq`).

# 2. Query

1. Construct a connection across corpus, group, text, sentence and token.

2. Given a sentence Idx (the location index), get the sentence content

3. Given a text Idx (the location index), get the text content.

4. Given a token Idx (the location index), get the token content.

# 3. Annotation

1. By whatever means, from group to texts, for each texts, we need to prepare `[strText:str, SSET:[content:str, start:int, end:int, label: str]]`. Besides, we need to verify that SSET is valid for strText.

2. Then strText is processed to `strSents: [strSent:str]`.

3. Together with strText and strSents, we use SSET to get the tag sequence for each strSent. The tag is in the BIOES tag scheme. However, instead of saving the tag string, we save the tag idx this time.

4. Save the token to `file/annoE_idx.txt`. Each line is the tag seqeuence correponding to the line of the original sentence in `file/token.txt`.

5. We also need to query the annotation tag sequence from that file.

6. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['annoE']['BIO']`.


# 4. Hyper Field of Token Granularity (i.e. part-of-speech, word level)

Hyper field of token granularity means the hyper field entity corresponds to exactly one token entity.

For example, a part-of-speech entity corresponds to exactly one word. When the token level is word, part-of-speech is the hyper field of token granularity.

1. We need to develop a function to get hyper field grains for a given sentence.

2. This is on the process of sentence to tokens. Input is strSent: str, and output is `tokens: [token:str]` and `tags: [tag:str]`, where tag is of BIOES tagging scheme. Similar to annotation, we also only save the tag idx and save all of them into file/pos_idx.txt.

3. We also need to query the annotation tag sequence from that file.

4. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['pos']['BIO']`.


# 5. Hyper Field of Larger Granularity (i.e. named entitiy)

Hyper field whose granularity larger than token means the hyper field entity corresponds to more than (including equal to) one token entity.

1. We need to develop a function to get hyper field grains for a given sentence.

2. This is on the process of sentence to tokens. Input is strSent: str, and output is tags[tag:str], where tag is of BIOES tagging scheme. Similar to annotation, we also only save the tag idx and save all of them into file/pos_idx.txt.

3. We also need to query the annotation tag sequence from that file.

4. Build the `BIOES_trans`. This is a dictioanry. The formal content is bioesidx2bioidx. Its path should be `BIOES_trans['ner']['BIO']`.


# 6. Sub Field Information

Sub field information refers to the grains that only determined exclusively by a token, such as the letter sequence of an English word.

1. We need to develop a function to get sub field grains for a given sentence.

2. Get Sub field Grain Vocabulary based on the token vocabulary, and the n-gram we choose.

3. Get the frequency tables of the sub field grains based on the whole corpus and on the whole token vocabulary.

4. Get the Look Up Table to connect token vocabulary and sub field vocabulary.

# 7. Query Field Information

For sentences and tokens, whether they are from our corpus pyramid or are newly created, given a field, get their field grain sequence.

1. `get_grain_str`

2. `get_grain_idx`

They are APIs for further tasks.

# 8. Chunk of Texts or Sentences

Sometimes we need a chunk of texts or sentences where the number of tokens numbers in this chunk is close to and smaller than a constant, say, 10000. One scenario is for word embedding training.

Use text as an instance, the data we want is: chunkidx2endtxtidx.

For each chunk, get all text idxs in it. From startidx to endidx.

The following still uses text as the instance. The concrete operation:

```
chunkidx_2_endbyteidx = [] # the text that is not included inside this chunk.
chunkidx_2_cumlengoftexts = []

current_chunk_lengoftexts = []
current_chunk_length_count = 0
for text in texts:
	text_token_num = text.get_tokens_num()
	
	if current_chunk_length_count + text_token_num > 10000:
		
		# get and save the chunkendbyteidx
		# this is correct, notice that this text is not include in this chunk
		
		endbyteidx = text.get_endbyteidx() 
		chunkidx_2_endbyteidx.append(endbyteidx)

		# save the txtidx2endtokenidx, derived from lengoftexts
		chunkidx_2_cumlengoftexts.append(np.cumsum(current_chunk_lengoftexts))

		# get the new the current chunk information
		current_chunk_length_count = text_token_num
		current_chunk_lengoftexts = [text_token_num]

	else:
		# update current_chunk_length_count when it is <= 10000
		current_chunk_length_count = current_chunk_length_count + text_token_num

		# append the object's length in the current_chunk_lengoftexts
		current_chunk_lengoftexts.append(text_token_num)

# when the loop is over, we still need the to append 
# the last small chunk into the total chunks.
# here, the object is the smallest object.
endbyteidx = object.get_endbyteidx()
chunkidx_2_endbyteidx.append(endbyteidx)

# save the txtidx2endtokenidx, derived from lengoftexts
chunkidx_2_cumlengoftexts.append(np.cumsum(current_chunk_lengoftexts))
```

Eventually, we get chunkidx_2_endbyteidx and chunkidx_2_cumlengoftexts.

Then we should think out how to produce jobs. Each chunk is for one job.

```
# the file we need to read is of path: path_token
job_paras = []
for chunkidx in range(len(chunkidx_2_endbyteidx)):
	startbyteidx = 0 if chunkidx == 0 else chunkidx_2_endbyteidx[chunkidx-1]
	endbyteidx = chunkidx_2_endbyteidx[chunkidx-1]
	chunk_token_string = read_str_from_file(path_token, startbyteidx, endbyteidx)
	# for the text in the chunk.
	txtidx2endtokenidx = chunkidx_2_lengoftexts[chunk_string]
	job_para = (chunk_token_string, txtidx2endtokenidx)
	job_paras.append(job_para)
```

Inside the fieldembed_core.pyx, the method is:

```
# for the input: chunk_tokens (i.e. chunk_token_string), txtidx2endtokenidx.
# we first prepare chunk_token_seq.

chunk_tokens = chunk_tokens.split() # i.e. chunk_token_seq

# then loop the text sequence in chunk_tokens

token_indexes = []
text_endidx   = []

new_chunk_idx = 0


for txtidx, endtokenidx in enumerate(txtidx2endtokenidx):
	starttokenidx = 0 if txtidx == 0 else txtidx2endtokenidx[endtokenidx - 1]
	for tokenidx in range(starttokenidx, endtokenidx):
		tokenstr = get_token_str(tokenidx)
		tokenvocidx = get_token_vocidx(tokenstr)
		
		if tokenvocidx not meets conditions:
			continue

		token_indexes.append(tokenvocidx)
		new_chunk_idx = new_chunk_idx + 1

	text_endidx.append(new_chunk_idx)

# at the end, we get token_indexes, text_endidx, new_chunk_idx.

```


