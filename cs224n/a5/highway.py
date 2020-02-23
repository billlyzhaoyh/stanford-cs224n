#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__ (self,embeb_dim):
    	super(Highway,self).__init__()
    	#initiliase layers for projection and gate
    	self.projection=nn.Linear(embeb_dim,embeb_dim)
    	self.gate=nn.Linear(embeb_dim,embeb_dim)

    	#initialise b_gate to be positive to encourage gate opening


    def forward(self,x_conv_out):
    	"""
    	@param x_conv_out (tensor[sentence_length, batch_size, word_embedding_dim]): input character embeddings after 1d convolution and max pooling
        @param x_highway (tensor[sentence_length, batch_size, word_embedding_dim]): output after the highway module, should have the same dimension as the input
        """
    	x_proj=F.relu(self.projection(x_conv_out))
    	# print(x_proj.size())
    	x_gate=torch.sigmoid(self.gate(x_conv_out))
    	# print(x_gate.size())
    	x_highway=torch.mul(x_proj, x_gate)+torch.mul((1-x_gate),x_conv_out)

    	return x_highway






    ### END YOUR CODE

class Highway_Shut(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__ (self,embeb_dim):
    	super(Highway_Shut,self).__init__()
    	#initiliase layers for projection and gate
    	self.projection=nn.Linear(embeb_dim,embeb_dim)
    	self.gate=nn.Linear(embeb_dim,embeb_dim)

    	K=torch.Tensor([-float("Inf")])
    	#initialise b_gate to be extremely negative to shut off the gate
    	self.gate.bias.data = self.gate.bias.data + K


    def forward(self,x_conv_out):
    	"""
    	@param x_conv_out (tensor[batch_size,char_embedding_dim]): input character embeddings after 1d convolution and max pooling
        @param x_highway (tensor[batch_size,char_embedding_dim]): output after the highway module, should have the same dimension as the input
        """
    	x_proj=F.relu(self.projection(x_conv_out))
    	# print(x_proj.size())
    	x_gate=torch.sigmoid(self.gate(x_conv_out))
    	# print(x_gate.size())
    	x_highway=torch.mul(x_proj, x_gate)+torch.mul((1-x_gate),x_conv_out)

    	return x_highway