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
import torch
import numpy as np
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
import graphein.protein as gp
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
import logging
import torch_geometric
import networkx as nx

logger = logging.getLogger('graphein.protein')
logger.setLevel(level="ERROR")

format_convertor = GraphFormatConvertor('nx',
                                        'pyg',
                                        verbose=None,
                                        columns=['edge_index', "amino_acid_one_hot", 'bindist'])


def graph(pdbcode, chain='all', doplot=False, distbins=[4, 6, 8, 12]):
    """
    >>> g = graph('1ycr', chain='A', doplot=False)
    >>> g
    Data(edge_index=[2, 880], node_id=[85], bindist=[880], num_nodes=85, x=[85, 20])
    """
    # Edges function
    # gp.add_peptide_bonds, gp.add_hydrogen_bond_interactions, gp.add_disulfide_interactions,
    # gp.add_ionic_interactions, gp.add_aromatic_interactions, gp.add_aromatic_sulphur_interactions,
    # gp.add_cation_pi_interactions,
    new_funcs = {
        "granularity":
        'CA',
        "keep_hets": [False],
        "edge_construction_functions": [
            lambda g: gp.add_distance_threshold(g, long_interaction_threshold=2, threshold=max(distbins)),
            gp.add_distance_to_edges
        ],
        "node_metadata_functions": [amino_acid_one_hot]
    }
    config = ProteinGraphConfig(**new_funcs)
    g = construct_graph(config=config, pdb_path=f"data/pdb/{pdbcode[1:3]}/pdb{pdbcode}.ent.gz", chain_selection=chain)
    if doplot:
        p = plotly_protein_structure_graph(g,
                                           colour_edges_by="kind",
                                           colour_nodes_by="degree",
                                           label_node_ids=False,
                                           plot_title="Peptide backbone graph. Nodes coloured by degree.",
                                           node_size_multiplier=1)
        p.show()
    # One hot encoding of distances using distbins argument
    for e in g.edges(data=True):
        dist = e[2]['distance']
        binid = np.digitize(dist, distbins)
        onehot = np.zeros((1, len(distbins)))
        onehot[0, binid] = 1
        e[2]['bindist'] = onehot
    g = format_convertor(g)
    g.x = torch.asarray(np.array(g.amino_acid_one_hot)).type(torch.FloatTensor)
    del g.amino_acid_one_hot
    return g


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
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb', help='Compute and plot the graph of the given PDB file')
    parser.add_argument('--chain', help='Chain to compute the graph on')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.pdb is not None:
        if args.chain is None:
            print("Please, give a chain id using --chain option")
        else:
            g = graph(args.pdb, chain=args.chain, doplot=False)
            print(g)
