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
import gzip
import logging
import numpy as np
import torch
import torch_geometric.data

import protein

logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
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
    def __init__(self, homologs_file='data/homologs_foldseek.txt.gz'):
        self.hf = homologs_file
        self.mapping = self.get_mapping()

    def __len__(self):
        with gzip.open(self.hf, 'r') as f:
            nx = sum(1 for line in f)
        return nx

    def get_mapping(self):
        with gzip.open(self.hf, 'r') as f:
            mapping = dict()
            offset = 0
            for i, line in enumerate(f):
                mapping[i] = offset
                offset += len(line)
        return mapping

    def __getitem__(self, index):
        offset = self.mapping[index]
        with gzip.open(self.hf, 'r') as f:
            f.seek(offset)
            line = f.readline().decode()
        pdbs = line.split()
        anchor = pdbs[0]
        positive = np.random.choice(pdbs[1:])
        try:
            g_anchor = protein.graph(pdbcode=anchor[:4], chain=anchor[5:])
            g_positive = protein.graph(pdbcode=positive[:4], chain=positive[5:])
        except Exception as e:  #  (ValueError, KeyError, FileNotFoundError):
            log(e)
            log(f"Graphein error for {anchor}, {positive}")
            # Not a protein chain
            g_anchor = torch_geometric.data.Data(x=None, edge_index=None)
            g_positive = torch_geometric.data.Data(x=None, edge_index=None)
        return g_anchor, g_positive


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def get_script_dir():
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

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
