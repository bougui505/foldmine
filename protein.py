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

logger = logging.getLogger("graphein.protein")
logger.setLevel(level="ERROR")

CHOICES = {
    'peptide_bond': gp.add_peptide_bonds,
    'hbond': gp.add_hydrogen_bond_interactions,
    'aromatic': gp.add_aromatic_interactions,
    'aromatic_sulphur': gp.add_aromatic_sulphur_interactions,
    'ionic': gp.add_ionic_interactions,
    'pi': gp.add_cation_pi_interactions,
    'disulfide': gp.add_disulfide_interactions
}


class GraphBuilder:
    """
    Just a small class to encapsulate the graph encoding choices and to be able to json dump them.
    """

    def __init__(self, choices=('peptide_bond', 'distance', 'hbond'), distbins=(6, 8, 10, 12)):
        self.distbins = distbins

        # Iterate through CHOICES to keep a consistent ordering and check keys
        self.edge_creating_funcs = {}
        for name, func in CHOICES.items():
            if name in choices:
                self.edge_creating_funcs[name] = func
        self.num_edge_types = len(self.edge_creating_funcs)
        self.edge_type_mapping = {edge_type: i for i, edge_type in enumerate(self.edge_creating_funcs)}
        self.num_edge_types_no_dist = self.num_edge_types

        if 'distance' in choices:
            self.distbins = distbins
            self.num_edge_types += len(distbins)
            self.edge_creating_funcs['distance'] = lambda g: gp.add_distance_threshold(
                g, long_interaction_threshold=2, threshold=max(distbins) - 0.01)

        edge_construction_funcs = list(self.edge_creating_funcs.values())
        new_funcs = {
            "granularity": 'CA',
            "keep_hets": [False],
            "node_metadata_functions": [amino_acid_one_hot],
            "edge_construction_functions": edge_construction_funcs,
        }
        self.config = ProteinGraphConfig(**new_funcs)

        self.format_convertor = GraphFormatConvertor("nx",
                                                     "pyg",
                                                     columns=["amino_acid_one_hot", "edge_index", "edge_type"])

    def graphein_to_simple_nx(self, graphein_graph):
        """
        graphein returns edges with 'kind' encoding edge type and 'distance'.
        One need to make this an 'edge_type' with just an int.

        The 'kind' key is a set of all interactions between residues.
        We want to keep only the first in edge_type_mapping and encode it with id.
        """
        # One hot encoding of distances using distbins argument
        for e in graphein_graph.edges(data=True):
            found_interaction = False
            kind = e[2]['kind']
            edge_type = 'failed'

            # We now need to iterate in the right order, since sets are not.
            for possible_interactions in self.edge_type_mapping.keys():
                if possible_interactions in kind:
                    edge_type = self.edge_type_mapping[possible_interactions]
                    found_interaction = True
                    continue
            if not found_interaction:
                assert kind == {"distance_threshold"}
                dist = e[2]['distance']
                binid = np.digitize(dist, self.distbins)
                edge_type = self.num_edge_types_no_dist + binid
            assert edge_type != 'failed'
            e[2]['edge_type'] = [edge_type]
        return graphein_graph

    def graphein_to_pyg(self, graphein_graph):
        """
        Graphein to pyg keeps the node ordering consistent.
        """
        nx_graph = self.graphein_to_simple_nx(graphein_graph)
        pyg_graph = self.format_convertor(nx_graph)

        # Now g contains all attributes as general graph ids. One needs to make it node features and edge types.
        # Data(edge_index=[2, 1867], node_id=[154], amino_acid_one_hot=[154], kind=[1867], bindist=[1867], num_nodes=154)
        pyg_graph.x = torch.asarray(np.array(pyg_graph.amino_acid_one_hot)).type(torch.FloatTensor)
        del pyg_graph.amino_acid_one_hot
        pyg_graph.edge_type = torch.asarray(np.array(pyg_graph.edge_type), dtype=torch.int32)
        return pyg_graph

    def build_graph(self, pdbcode, chain='all', doplot=False):
        """
        >>> g = graph('1ycr', chain='A', doplot=False)
        >>> g
        Data(node_id=[85], edge_type=[963], num_nodes=85, x=[85, 20])
        """
        script_dir = os.path.dirname(os.path.realpath(__file__))
        pdb_path = os.path.join(script_dir, "data", "pdb_chainsplit", pdbcode[1:3], f"{pdbcode}_{chain}.pdb.gz")
        graphein_graph = construct_graph(config=self.config, pdb_path=pdb_path)

        # When constructing the graph, graphein creates a 'graph' dictionnary field in the nx object.
        # It computes a distmat when building the edges that is stored as a field of this graph
        # This distmat is computed at a chosen granularity from G.graph["pdb_df"], in which we get the
        # coordinates of the residues, ordered by sequence.
        distmat = graphein_graph.graph["dist_mat"].values[None, ...]
        distmat = torch.from_numpy(distmat).float()

        if doplot:
            p = plotly_protein_structure_graph(graphein_graph,
                                               colour_edges_by="kind",
                                               colour_nodes_by="degree",
                                               label_node_ids=False,
                                               plot_title="Peptide backbone graph. Nodes coloured by degree.",
                                               node_size_multiplier=1)
            p.show()

        pyg_graph = self.graphein_to_pyg(graphein_graph)
        return pyg_graph, distmat


if __name__ == '__main__':
    import sys
    import argparse

    import doctest
    import logging

    logfilename = 'logs/protein.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--pdb', help='Compute and plot the graph of the given PDB file')
    parser.add_argument('--chain', help='Chain to compute the graph on')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    graph_builder = GraphBuilder()
    # graph_builder.build_graph("1ycr", chain="A", doplot=False)
    # graph_builder.build_graph("2fyw", chain="A", doplot=False)
    graph = graph_builder.build_graph("1aut", chain="L", doplot=False)
    print(graph)

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.pdb is not None:
        if args.chain is None:
            print("Please, give a chain id using --chain option")
        else:
            graph_builder = GraphBuilder()
            g = graph_builder.build_graph(args.pdb, chain=args.chain, doplot=False)
            print(g)
