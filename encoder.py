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
import os
import logging
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
import BLASTloader


class ProteinGraphModel(torch.nn.Module):
    """
    >>> seed = torch.manual_seed(42)
    >>> dataset = BLASTloader.PDBdataset(homologs_file="data/homologs.txt.gz")
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

    def __init__(self, in_channels=20, num_relations=6, hidden_dims=(256, 256), latent_dim=512,
                 normalized_latent_space=True):
        super().__init__()
        self.normalized_latent_space = normalized_latent_space
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations

        all_dims = in_channels, *hidden_dims, latent_dim
        self.convs = torch.nn.ModuleList()
        for prev, next in zip(all_dims, all_dims[1:]):
            self.convs.append(RGCNConv(prev, next, num_relations=num_relations))
            # self.convs.append(GCNConv(prev, next))

    def forward(self, data, get_conv=False):
        x = data.x
        for conv in self.convs:
            x = conv(x, edge_index = data.edge_index, edge_type=data.edge_type)
            # x = conv(x, data.edge_index)
            x = F.relu(x)
        x = torch.tanh(x)
        z = torch.max(x, dim=0).values
        if self.normalized_latent_space:
            z = z / torch.linalg.norm(z)
        if get_conv:
            return z[None, ...], x
        else:
            return z[None, ...]


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    seed = torch.manual_seed(42)
    dataset = BLASTloader.PDBdataset(homologs_file="data/homologs.txt.gz")
    g_anchor, g_positive = dataset.__getitem__(1000)
    print(g_anchor)
    model = ProteinGraphModel()
    z = model(g_anchor)
    print(z.shape)

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
