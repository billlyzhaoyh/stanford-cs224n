#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self,char_embedding_dim,word_embedding_dim):
    	super(CNN,self).__init__()
    	self.con1d=torch.nn.Conv1d(char_embedding_dim, word_embedding_dim, 5, padding=1)
    	self.maxpool=torch.nn.AdaptiveMaxPool1d(1)


    def forward(self,x_reshaped):
    	"""
    	@params x_reshaped (tensor[sentence_length, batch_size, char_embedding_dim, max_word_length])
    	@params x_convout (tensor[sentence_length, batch_size, word_embedding_dim])
	
    	"""
    	sentence_length, batch_size, char_embedding_dim, max_word_length=x_reshaped.shape
    	x_conv=self.con1d(x_reshaped.view(-1,char_embedding_dim,max_word_length))
    	x_conv_out=self.maxpool(x_conv).squeeze()
    	return x_conv_out.view(sentence_length,batch_size,-1)



    ### END YOUR CODE

