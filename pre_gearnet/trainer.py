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
import datetime
import torch
import time

from pre_gearnet import encoder, pdb_loader
import utils

# See: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
torch.multiprocessing.set_sharing_strategy('file_system')


def get_norm(nested_out):
    """
    # >>> batch = pdb_loader.get_batch_test()
    # >>> model = encoder.ProteinGraphModel(latent_dim=512)
    # >>> out = forward_batch_nested(batch, model)
    # >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in nested_out]
    # [(torch.Size([1, 512]), torch.Size([1, 512]))]
    # >>> get_norm(nested_out)
    # tensor(..., grad_fn=<MeanBackward0>)
    """
    import itertools
    flattened_list = list(itertools.chain.from_iterable(nested_out))
    z = torch.cat(flattened_list, dim=0)
    norms = torch.linalg.norm(z, dim=1)
    return norms.mean()


def forward_batch_nested(batch, model):
    """
    The goal is to make a separate inference on elements on which one can call model(x),
    for a list of list of inputs.
    # >>> batch = get_batch_test()
    # >>> model = encoder.ProteinGraphModel()
    # >>> out = forward_batch_graph(batch, model)
    # >>> [(z_anchor.shape, z_positive.shape) for z_anchor, z_positive in nested_out]
    # [(torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    nested_out = []
    for distmat_list in batch:
        homologs_encodings = list()
        for distmat in distmat_list:
            homologs_encodings.append(model(distmat))
        nested_out.append(homologs_encodings)
    return nested_out


def get_contrastive_loss(nested_out, tau=1., normalize=True):
    """
    For simplicity, we flatten everything and keep an index of where in the data each tensor was,
    this creates the i_j_dict and the flattened_tensor of shape (elt_in_batch, embedding_dimension)

    We optionnally normalize these vectors.
    Then we precompute the pairwise distances along with their sum.

    We iterate through each element i of the nested batch.
    Then we find the other elements of the same class (homologs) named p_indexes for positives.

    >>> n = 3

    # >>> nested_out = [[torch.tensor([512))+i for j in range(i+3)] for i in range(n)]
    # >>> print([[ten.shape for ten in lista] for lista in nested_out])

    >>> nested_out = [[torch.tensor([[0, 1]]), torch.tensor([[0, -1]])],
    ... [torch.tensor([[0, 1]]), torch.tensor([[0, -1]])]]
    >>> nested_out = [[ten.float() for ten in lista] for lista in nested_out]
    >>> loss = get_contrastive_loss(nested_out, normalize=True)
    >>> loss
    tensor(9.9951)
    >>> nested_out = [[torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
    ... [torch.tensor([[1, 0]]), torch.tensor([[1, 0]])]]
    >>> nested_out = [[ten.float() for ten in lista] for lista in nested_out]
    >>> loss = get_contrastive_loss(nested_out, normalize=True)
    >>> loss
    tensor(2.8272)
    >>> nested_out = [[torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
    ... [torch.tensor([[0,-3]]), torch.tensor([[0,-3]])]]
    >>> nested_out = [[ten.float() for ten in lista] for lista in nested_out]
    >>> loss = get_contrastive_loss(nested_out, normalize=True)
    >>> loss
    tensor(1.9963)
    >>> nested_out = [[torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
    ... [torch.tensor([[0,-3]]), torch.tensor([[0,-3]]), torch.tensor([[0,-3]])]]
    >>> nested_out = [[ten.float() for ten in lista] for lista in nested_out]
    >>> loss = get_contrastive_loss(nested_out, normalize=True)
    >>> loss
    tensor(3.1417)

    """

    i_j_dict = {}
    row_index = {}
    idx = 0
    for i in range(len(nested_out)):
        row_index[i] = list()
        for j in range(len(nested_out[i])):
            i_j_dict[(i, j)] = idx
            row_index[i].append(idx)
            idx += 1
    flattened_list = [nested_out[i][j] for i, j in i_j_dict.keys()]
    flattened_tensor = torch.vstack(flattened_list)
    all_idx = set(i_j_dict.values())

    if normalize:
        flattened_tensor = flattened_tensor / (torch.linalg.norm(flattened_tensor, dim=1) + 1e-4)[:, None]

    distance_mat = torch.einsum('ij,kj->ik', flattened_tensor, flattened_tensor)
    distance_mat = torch.exp(distance_mat / tau)

    loss = 0
    for (i, j), flat_idx in i_j_dict.items():
        p_indexes = set(row_index[i]) - {flat_idx}
        elt_coef = -1 / len(p_indexes)
        elt_sum = 0
        inner_den = 1e-4

        # The denominator does not depend on which positive we are looking at.
        # Another looping option is :
        # for a in all_idx-set(row_index[i]):
        for a in all_idx - {flat_idx}:
            inner_den += distance_mat[a, i]

        for p in p_indexes:
            inner_num = distance_mat[p, flat_idx]
            elt_sum += torch.log(inner_num / inner_den)
        loss += elt_coef * elt_sum
    return loss


def train(
        batch_size=4,
        n_epochs=20,
        latent_dim=128,
        save_each_epoch=True,
        print_each=100,
        homologs_file='data/homologs_scope40_clean.txt.gz',
        num_workers=os.cpu_count(),
        save_each=30,  # in minutes
        modelfilename='models/sscl.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'):
    dataset = pdb_loader.PDBdataset(homologs_file=homologs_file)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=lambda x: x)

    # if modelfilename is not None and os.path.exists(modelfilename):
    #     log(f'# Loading model from {modelfilename}')
    #     model = load_model(filename=modelfilename, latent_dim=latent_dim)
    #     model.train()
    # else:
    #     log('GCN model')
    #     model = encoder.ProteinGraphModel(latent_dim=latent_dim, normalized_latent_space=False)

    cnn_model = encoder.ProteinCNNModel()
    graph_model = encoder.ProteinGraphModel(in_channels=20 + cnn_model.latent_dim,
                                            num_relations=dataset.graph_builder.num_edge_types,
                                            latent_dim=latent_dim,
                                            normalized_latent_space=False)
    wrapper_model = encoder.WrapperModel(cnn_model, graph_model)
    wrapper_model = wrapper_model.to(device)
    opt = torch.optim.Adam(wrapper_model.parameters())
    utils.save_model(wrapper_model, modelfilename)

    t_0 = time.time()
    step = 0
    total_steps = n_epochs * len(dataloader)
    eta = utils.ETA(total_steps=total_steps)
    loss = None
    for epoch in range(n_epochs):
        for batch in dataloader:
            step += 1

            # Batch is a nested structures : a list of three lists of lists :
            # Batch[0] is name, graph_list, distmat_list
            # Filter the None, get distmat and encode residue local neighborhood with CNN
            chains_batch = [item[0] for item in batch]
            filtered_batch = [item for item in batch if item[1] is not None]
            bs = len(filtered_batch)
            if bs < 2:
                print(f"Not enough data for {chains_batch}")
                continue

            try:
                distmat_batch = [[distmat.to(device) for distmat in distmat_list] for (_, _, distmat_list) in
                                 filtered_batch]
                nested_out_cnn = forward_batch_nested(distmat_batch, cnn_model)

                # Now get graphs and populate residues with embeddings from the CNN.
                graph_batch = list()
                for i, (_, graph_list, _) in enumerate(filtered_batch):
                    homolog_graph_list = list()
                    for j, graph in enumerate(graph_list):
                        graph.to(device)
                        graph.x = torch.cat((graph.x, nested_out_cnn[i][j]), dim=1)
                        homolog_graph_list.append(graph)
                    graph_batch.append(homolog_graph_list)

                nested_out_graph = forward_batch_nested(graph_batch, graph_model)
                norm = get_norm(nested_out_graph)
                norm_loss = 0.01 * (norm - 1) ** 2
                contrastive_loss = get_contrastive_loss(nested_out_graph)
                loss = norm_loss + contrastive_loss
                loss.backward()
                opt.step()
            except RuntimeError as e:
                print(e)
            opt.zero_grad()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                utils.save_model(graph_model, modelfilename)

            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                print_msg = f"epoch: {epoch + 1}|step: {step}|loss: {loss:.4f}|norm: {norm:.4f}|bs: {bs}|last_saved: {last_saved}| eta: {eta_val}"
                print(print_msg)
                utils.log(print_msg)
        if save_each_epoch:
            t_0 = time.time()
            utils.save_model(graph_model, modelfilename)


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    # See: https://stackoverflow.com/a/13839732/1679629
    import logging

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logfilename = 'logs/trainer.log'
    fileh = logging.FileHandler(logfilename, 'a')
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileh.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(fileh)  # set the new handler
    logging.info(f"################ Starting {__file__} ################")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', help='Train the SSCL', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', help='Model to load or for saving', metavar='model.pt', default='models/default.pt')
    parser.add_argument('--print_each', type=int, default=100)
    parser.add_argument('--latent_dim', default=512, type=int)
    parser.add_argument('--save_every',
                        help='Save the model every given number of minutes (default: 30 min)',
                        type=int,
                        default=30)
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    # train(homologs_file='data/homologs_decoy.txt.gz', num_workers=1)

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    if args.train:
        train(n_epochs=args.epochs,
              modelfilename=args.model,
              print_each=args.print_each,
              latent_dim=args.latent_dim,
              save_each=args.save_every,
              batch_size=args.bs)
