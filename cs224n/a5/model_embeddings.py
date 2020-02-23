#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size=word_embed_size
        self.vocab=vocab
        self.echar=50
        self.char=nn.Embedding(len(self.vocab.char2id),self.echar,padding_idx=vocab.char_pad)
        self.cnn=CNN(self.echar,word_embed_size)
        self.highway=Highway(self.word_embed_size)
        self.dropout=nn.Dropout(p=0.3)


        ### YOUR CODE HERE for part 1h

        ### END YOUR CODE

    def forward(self, sents_var):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        #look up character embedding to shape tensor[sentence_length, batch_size, char_embedding_dim, max_word_length]
        sents_embedded=self.char(sents_var).permute(0,1,3,2)
        sents_conv_out=self.cnn(sents_embedded)
        sents_highway=self.highway(sents_conv_out)
        output=self.dropout(sents_highway)
        return output
        ### END YOUR CODE

