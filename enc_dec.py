# coding: utf-8
# ---------------------------------------------------------------------
'''
Neural Machine Translation - Encoder Decoder model
    Chainer implementation of an encoder-decoder sequence to sequence
    model using bi-directional LSTM encoder
'''
# ---------------------------------------------------------------------


import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.array import concat
import chainer.functions.math.sum as M

# Import configuration file
from nmt_config import *


class EncoderDecoder(Chain):
    '''
    Constructor to initialize model
    Params:
        vsize_enc   - vocabulary size for source language (fed into encoder)
        vsize_dec   - vocabulary size for target language (fed into decoder)
        n_units     - size of the LSTMs
        attn        - specifies whether to use attention
    '''

    def __init__(self, vsize_enc, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        super(EncoderDecoder, self).__init__()
        # --------------------------------------------------------------------
        # add encoder layers
        # --------------------------------------------------------------------

        # add embedding layer
        self.add_link("embed_enc", L.EmbedID(vsize_enc, n_units))  # 建立embedding layer embed_layer

        """
        ___ANSWER-1.pdf-A___
        Those code is used to build hidden layers in encoder, using deep LSTMs of RNN.
        'nlayers_enc' represents the number of LSTM layers for encoder.
        'self.lstm_enc' is a list to store names of all hidden lyaers in encoder.
        The method add_link() registers a link to this chain, this means build a layer to this neural network,
        and gives this layer a name. We can access those Links by their names.
        We use bidirectional recurrent neural network, which both includes left-to-right and 
        right-to-left context. That means we should build two LSTM layers in each hidden layer.
        That is why are there two loops.
        """
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]  #
        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))

        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_rev_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))
        '''
        '''

        # --------------------------------------------------------------------
        # add decoder layers
        # --------------------------------------------------------------------

        """
        - L.EmbedID(vsize_dec, 2*n_units): embeds predicted word. 
                input size: vocabulary size for target language
                output size: 2 * the size of LSTMs in encoder
        - L.LSTM(2*n_units, 2*n_units): the hidden state in decoder
                input suze:
                output size:
        - L.Linear(2*n_units, vsize_dec): output layer, predicts the output word  

        In encoder, we combine the left-to-right and right-to-left context, so in 
        decoder, we need doulbe the size of LSTM.
        
        """

        self.add_link("embed_dec", L.EmbedID(vsize_dec, 2 * n_units))

        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        for lstm_name in self.lstm_dec:
            self.add_link(lstm_name, L.LSTM(2 * n_units, 2 * n_units))

        if attn > 0:
            # __QUESTION Add attention
            self.add_link('attn_out', L.Linear(4 * n_units, 2 * n_units))

        # Save the attention preference
        self.attn = attn

        # add output layer
        self.add_link("out", L.Linear(2 * n_units, vsize_dec))
        '''
        '''

        # Store GPU id
        self.gpuid = gpuid
        self.n_units = n_units

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0
    """
    initializes decoder LSTM to final encoder state.
    h_state: hidden state-- an output of this cell, carries the information about what an RNN cell has seen over the
                        time and supply it to the present time such that a loss function is
                        not just dependent upon the data it is seeing in this time instant,
                        but also, data it has seen historically.
    c_state: cell state--Cell state is a memory of LSTM cell
    Cell state is a memory of LSTM cell, hidden state is an output of this cell
    
    
    """

    def set_decoder_state(self):
        xp = cuda.cupy if self.gpuid >= 0 else np  # Convert the input to an array
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        self[self.lstm_dec[0]].set_state(c_state, h_state)


    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of names of lstm layers to use
    '''

    def feed_lstm(self, word, embed_layer, lstm_layer_list, train):
        # get embedding for word
        embed_id = F.dropout(embed_layer(word), ratio=dropout_ratio)
        # feed into first LSTM layer
        hs = self[lstm_layer_list[0]](F.dropout(embed_id, ratio=dropout_ratio))
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            hs = self[lstm_layer](F.dropout(hs, ratio=dropout_ratio))

    # Function to encode an source sentence word
    def encode(self, word, lstm_layer_list, train):
        self.feed_lstm(word, self.embed_enc, lstm_layer_list, train)

    # Function to decode a target sentence word
    def decode(self, word, train):
        self.feed_lstm(word, self.embed_dec, self.lstm_dec, train)

    # Attention layer
    def attn_layer(self, ht, Hs):
        score = F.matmul(ht, Hs, transb=True)
        a_t = F.softmax(score)
        c_t = F.reshape(M.sum(F.scale(Hs, F.transpose(a_t), axis=0), axis=0), (1, ht.shape[1]))
        return c_t, a_t

    def encode_list(self, in_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        if train:
            var_en = (Variable(xp.asarray(in_word_list, dtype=np.int32).reshape((-1, 1))))

            var_rev_en = (Variable(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1, 1))))
        else:
            with chainer.no_backprop_mode():
                var_en = (Variable(xp.asarray(in_word_list, dtype=np.int32).reshape((-1, 1))))

                var_rev_en = (Variable(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1, 1))))

        first_entry = True

        # encode tokens
        for f_word, r_word in zip(var_en, var_rev_en):
           
            """
            In each hiddde layer of encoder, we build two kinds of LSTM, left-to-right and 
            right-to-left LSTM, so we should encode the words both in forward and reverse order 
            """
            self.encode(f_word, self.lstm_enc, train)
            self.encode(r_word, self.lstm_rev_enc, train)

            # this can be used for implementing attention
            if first_entry == False:
                forward_states = F.concat((forward_states, self[self.lstm_enc[-1]].h), axis=0)
                backward_states = F.concat((self[self.lstm_rev_enc[-1]].h, backward_states), axis=0)
            else:
                forward_states = self[self.lstm_enc[-1]].h
                backward_states = self[self.lstm_rev_enc[-1]].h
                first_entry = False

        enc_states = F.concat((forward_states, backward_states), axis=1)

        return enc_states

    # Select a word from a probability distribution
    # should return a chainer variable
    def select_word(self, prob, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if not sample:
            indx = xp.argmax(prob.data[0])
            if not train:
                with chainer.no_backprop_mode():
                    pred_word = Variable(xp.asarray([indx], dtype=np.int32))
            else:
                pred_word = Variable(xp.asarray([indx], dtype=np.int32))

        else:
            pass
        return pred_word

    def encode_decode_train(self, in_word_list, out_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        enc_states = self.encode_list(in_word_list, train=train)
        # print(enc_states.shape[0])
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        if not train:
            with chainer.no_backprop_mode():
                # convert list of tokens into chainer variable list
                var_dec = (Variable(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1, 1))))
                # Initialise first decoded word to GOID
                pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32))
        else:
            # convert list of tokens into chainer variable list
            var_dec = (Variable(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1, 1))))
            # Initialise first decoded word to GOID
            pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32))

        # compute loss
        self.loss = 0
        # decode tokens
        for next_word_var in var_dec[1:]:
            self.decode(pred_word, train=train)
            if self.attn == NO_ATTN:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            else:
                h_t = self[self.lstm_dec[-1]].h

                c_t = self.attn_layer(h_t, enc_states)[0]
                # print(c_t.shape)
                predicted_out = self.out(self.attn_out(F.concat((c_t, h_t))))

            # compute loss
            prob = F.softmax(predicted_out)

            pred_word = self.select_word(prob, train=train, sample=False)  
            
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
            
        report({"loss": self.loss}, self)

        return self.loss

    def decoder_predict(self, start_word, enc_states, max_predict_len=MAX_PREDICT_LEN, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        alpha_arr = xp.empty((0, enc_states.shape[0]), dtype=xp.float32)  # 0 , 17
        # print(enc_states.shape)
        # print('*'*80)
        # return list of predicted words
        predicted_sent = []
        # load start symbol
        with chainer.no_backprop_mode():
            pred_word = Variable(xp.asarray([start_word], dtype=np.int32))
        pred_count = 0

        # start prediction loop
        while pred_count < max_predict_len and (int(pred_word.data) != (EOS_ID)):
            self.decode(pred_word, train=False)
            if self.attn == NO_ATTN:
                prob = F.softmax(self.out(self[self.lstm_dec[-1]].h))
            else:
                h_t = self[self.lstm_dec[-1]].h
                c_t = self.attn_layer(h_t, enc_states)[0]
                a_t = self.attn_layer(h_t, enc_states)[1]

                alpha_arr = xp.concatenate((alpha_arr, a_t.data))
                # print(a_t.shape)
                predicted_out = self.out(self.attn_out(F.concat((c_t, h_t))))

                prob = F.softmax(predicted_out)
                # print(alpha_arr.shape)
                # print(a_v.shape)

                # print(a_t.data)

            pred_word = self.select_word(prob, train=False, sample=sample)
            # add integer id of predicted word to output list
            predicted_sent.append(int(pred_word.data))
            pred_count += 1
        
        # alpha_arr = xp.append(alpha_arr, a_v.T, axis=1.pdf)
        # print(alpha_arr.shape)


        return predicted_sent, alpha_arr

    def encode_decode_predict(self, in_word_list, max_predict_len=20, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # encode list of words/tokens
        in_word_list_no_padding = [w for w in in_word_list if w != PAD_ID]
        enc_states = self.encode_list(in_word_list, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, enc_states,
                                                         max_predict_len, sample=sample)
        return predicted_sent, alpha_arr
