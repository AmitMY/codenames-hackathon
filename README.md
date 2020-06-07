# codenames-hackathon

To use our agent:

1. Install env from the requirements.txt file.
2. Download the embedding, the one you choose. We note that the board words are lower cased, we suggest that you use lower case embeddings. 
* if you choose the glove embedding make sure it is gensim compatible by:

    * python -m gensim.scripts.glove2word2vec --input  glove.42B.300d.txt --output glove.42B.300d.w2vformat.txt
where the input file: "glove.42B.300d.txt" is the unzipped glove file, and the output file is what you pass for the agent

