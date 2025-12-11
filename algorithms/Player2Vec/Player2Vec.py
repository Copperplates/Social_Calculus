"""
This code is attributed to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

Paper: 'Key Player Identification in Underground Forums
over Attributed Heterogeneous Information Network Embedding Framework'
Link: http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf
"""

import argparse
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import AttentionLayer, GraphConvolution
from utils.metrics import masked_softmax_cross_entropy, masked_accuracy


class Player2Vec(keras.Model):
    """
    The Player2Vec model
    """

    def __init__(self, input_dim: int, nhid: int, output_dim: int,
                 args: argparse.ArgumentParser().parse_args()) -> None:
        """
        :param input_dim: the input feature dimension
        :param nhid: the output embedding dimension of the first GCN layer
        :param output_dim: the output embedding dimension of the last GCN layer
        (number of classes)
        :param args: additional parameters
        """
        super().__init__()

        self.input_dim = input_dim
        self.nodes = args.nodes
        self.nhid = nhid
        self.class_size = args.class_size
        self.train_size = args.train_size
        self.output_dim = output_dim
        self.num_meta = args.num_meta
        self.weight_decay = args.weight_decay
        self.num_features_nonzero = args.num_features_nonzero

        self.GCN_layers = []
        self.GCN_layers.append(GraphConvolution(input_dim=self.input_dim,
                                             output_dim=self.nhid,
                                             num_features_nonzero=self.
                                             num_features_nonzero,
                                             activation=tf.nn.relu,
                                             dropout=args.dropout,
                                             is_sparse_inputs=True,
                                             norm=True))

        self.GCN_layers.append(GraphConvolution(input_dim=self.nhid,
                                             output_dim=self.output_dim,
                                             num_features_nonzero=self.
                                             num_features_nonzero,
                                             activation=lambda x: x,
                                             dropout=args.dropout,
                                             norm=False))

        self.att_layer = AttentionLayer(input_dim=output_dim,
                                        num_nodes=self.nodes,
                                        attention_size=self.num_meta,
                                        v_type='tanh')


    def call(self, inputs: list, training: bool = True, return_embeddings: bool = False) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward propagation
        :param inputs: the information passed to next layers
        :param training: whether in the training mode
        :param return_embeddings: whether to return embeddings and probabilities
        """
        supports, x, label, mask = inputs
        outputs = []
        hidden_embeddings = [] # To store output of first GCN layer

        # forward propagation
        for i in range(len(supports)):
            output = [x]
            for idx, layer in enumerate(self.GCN_layers):
                hidden = layer((output[-1], [supports[i]]), training=training)
                output.append(hidden)
                if idx == 0: # First GCN layer
                    hidden_embeddings.append(hidden)
            output = output[-1]
            outputs.append(output)
            
        outputs_reshaped = tf.reshape(outputs,
                             [len(supports), self.nodes * self.output_dim])

        # Get attention weights and final output
        final_output, alpha = self.att_layer(inputs=outputs_reshaped, return_weights=True)
        final_output = tf.reshape(final_output, [self.nodes, self.output_dim])
        
        # Calculate aggregated embedding (weighted sum of hidden_embeddings using alpha)
        # alpha shape: [1, num_meta]
        # hidden_embeddings: list of [Nodes, nhid]
        if return_embeddings:
            alpha_weights = tf.reshape(alpha, [len(supports), 1, 1]) # [num_meta, 1, 1]
            stacked_embeddings = tf.stack(hidden_embeddings) # [num_meta, Nodes, nhid]
            # Weighted sum
            final_embedding = tf.reduce_sum(stacked_embeddings * alpha_weights, axis=0) # [Nodes, nhid]

        # Weight decay loss
        loss = tf.zeros([])
        for layer in self.GCN_layers:
            for var in layer.trainable_variables:
                loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.att_layer.trainable_variables:
            loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy loss
        loss += masked_softmax_cross_entropy(final_output, label, mask)

        # Prediction results
        acc = masked_accuracy(final_output, label, mask)

        if return_embeddings:
            return loss, acc, final_output, final_embedding
        else:
            return loss, acc
