#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv

import pdb_loader


class ProteinGraphModel(torch.nn.Module):
    """
    >>> seed = torch.manual_seed(42)
    >>> dataset = pdb_loader.PDBdataset(homologs_file="data/homologs.txt.gz")
    >>> g_anchor, g_positive = dataset.__getitem__(1000)
    >>> g_anchor
    Data(edge_index=[2, 558], node_id=[154], num_nodes=154, x=[154, 20])
    >>> gcn = ProteinGraphModel()
    >>> z = gcn(g_anchor)
    >>> z.shape
    torch.Size([1, 512])
    >>> z, out = gcn(g_anchor, get_conv=True)
    >>> z.shape
    torch.Size([1, 512])
    >>> out.shape
    torch.Size([154, 512])
    """

    def __init__(self,
                 in_channels=20,
                 num_relations=6,
                 hidden_dims=(256, 256),
                 latent_dim=512,
                 normalized_latent_space=True):
        super().__init__()
        self.normalized_latent_space = normalized_latent_space
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations

        all_dims = in_channels, *hidden_dims, latent_dim
        self.convs = torch.nn.ModuleList()
        for prev, next in zip(all_dims, all_dims[1:]):
            self.convs.append(RGCNConv(prev, next, num_relations=num_relations, num_bases=num_relations // 2))

    def forward(self, data):
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index=data.edge_index, edge_type=data.edge_type)
            if not i == len(self.convs) - 1:
                x = F.relu(x)
        x = torch.tanh(x)
        z = torch.max(x, dim=-2).values
        return z[None, ...]


class ProteinCNNModel(torch.nn.Module):
    """
    This encodes the geometry around each residue.
    """

    def __init__(self,
                 hidden_dims=(64, 64, 64, 64),
                 ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = hidden_dims[-1]
        all_dims = 1, *hidden_dims

        self.convs = torch.nn.ModuleList()
        for prev, next in zip(all_dims, all_dims[1:]):
            self.convs.append(nn.Conv2d(prev, next, kernel_size=3, padding='same'))

    def forward(self, distmat):
        with torch.no_grad():
            x = 1 - torch.sigmoid(distmat - 8)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if not i == len(self.convs) - 1:
                x = F.relu(x)
        # The output is (n_channel, num_nodes, num_nodes) take the diag and transpose to get (num_nodes, n_channel)
        # Use the diagonal term
        x = torch.diagonal(x, dim1=-2, dim2=-1)
        x = torch.tanh(x)
        return x.T


class WrapperModel(torch.nn.Module):
    def __init__(self, cnn_model, graph_model):
        super().__init__()
        self.cnn_model = cnn_model
        self.graph_model = graph_model


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = 'logs/encoder.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    seed = torch.manual_seed(42)
    dataset = pdb_loader.PDBdataset(homologs_file="data/homologs.txt.gz")
    g_anchor, g_positive = dataset.__getitem__(1000)
    print(g_anchor)
    model = ProteinGraphModel()
    z = model(g_anchor)
    print(z.shape)

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
