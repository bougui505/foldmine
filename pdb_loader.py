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
import gzip
import logging
import numpy as np
import torch
import torch_geometric.data

from utils import log

import protein

logfilename = 'logs/loader.log'
logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
logging.info(f"################ Starting {__file__} ################")


class PDBdataset(torch.utils.data.Dataset):
    """
    >>> dataset = PDBdataset()
    >>> dataset.__getitem__(0)
    (Data(), Data())
    >>> dataset.__getitem__(1000)
    154
    ...
    (Data(edge_index=[2, 728], node_id=[154], num_nodes=154, x=[154, 20]), ...)
    """

    def __init__(self, homologs_file='data/homologs_foldseek.txt.gz', max_pos=1):
        self.hf = homologs_file
        self.max_pos = max_pos
        self.mapping = self.get_mapping()
        self.graph_builder = protein.GraphBuilder()

    def __len__(self):
        return len(self.mapping)

    def get_mapping(self):
        with gzip.open(self.hf, 'r') as f:
            mapping = dict()
            offset = 0
            for i, line in enumerate(f):
                mapping[i] = offset
                offset += len(line)
        return mapping

    def __getitem__(self, index):
        # Get a Scope40 chain and its homologs
        offset = self.mapping[index]
        with gzip.open(self.hf, 'r') as f:
            f.seek(offset)
            line = f.readline().decode()
        pdbs = line.split()

        # Create a shortlist of the anchor chain and some homologs
        anchor = [pdbs[0]]
        max_pos = min(self.max_pos, len(pdbs) - 1)
        positives = list(np.random.choice(pdbs[1:], size=max_pos, replace=False))
        to_embed = anchor + positives

        # Get the graph and distmat representation for this shortlist
        # TODO make a smarter error catching
        graphs, distmats = list(), list()
        try:
            for chain in to_embed:
                graph, distmat = self.graph_builder.build_graph(pdbcode=chain[:4], chain=chain[5:])
                graphs.append(graph)
                distmats.append(distmat)
        except Exception as e:  # (ValueError, KeyError, FileNotFoundError):
            # Often a modified residue
            log(e)
            log(f"Graphein error for {anchor}, {positives}")
            graphs = [torch_geometric.data.Data(x=None, edge_index=None)] * (max_pos + 1)
            distmats = None
        return graphs, distmats


def get_batch_test():
    """
    >>> batch = get_batch_test()
    >>> len(batch)
    3
    >>> batch
    [(Data(), Data()), (Data(), Data()), (Data(edge_index=[2, 717], node_id=[154], num_nodes=154, x=[154, 20]), Data(edge_index=[2, ...], node_id=[...], num_nodes=..., x=[..., 20]))]
    """
    dataset = PDBdataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4,
                                             collate_fn=lambda x: x)
    for batch in dataloader:
        break
    return batch


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    # dataset = PDBdataset('data/homologs_decoy.txt.gz')
    # dataset = torch.utils.data.DataLoader(dataset,
    #                                       batch_size=1,
    #                                       shuffle=False,
    #                                       num_workers=0,
    #                                       collate_fn=lambda x: x)
    # for i, _ in enumerate(dataset):
    #     pass
    #     if i > 10:
    #         break

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()