import os
import sys

import numpy as np
import pickle
import time
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from ..utils import read_embeddings, pdbchain_to_hdf5path


def create_batches(tensors, max_size=10000):
    """

    @param tensors: list of np tensors of variable size (nres,dim)
    @param max_size: the max size of a batch
    @return:
    """
    all_batches = []
    all_sections = []
    current_size = 0
    current_sections = []
    current_batch = []
    for tensor in tensors:
        size = len(tensor)
        if current_size < max_size:
            current_batch.append(tensor)
            current_sections.append(size)
            current_size += size
        else:
            stack = np.concatenate(current_batch, axis=0)
            all_batches.append(stack)
            all_sections.append(current_sections)
            current_batch = [tensor]
            current_sections = [size]
            current_size = size
    stack = np.concatenate(current_batch, axis=0)
    all_batches.append(stack)
    all_sections.append(current_sections)
    return all_batches, all_sections


def get_pairwise_dist(all_embeddings, return_level='residue'):
    """
    For the residue level, the large number of vectors does not fit in memory.
    Moreover, one needs to perform a complex block aggregation to go (N_res, N_res), to (N_graphs, N_graphs)

    To speed things up, the computations happen in the gpu. Moreover, the bottleneck being the distance computation,
    we group embeddings in larger batches to make cdist over several graphs and then brek these down in small blocks
    @param all_embeddings:
    @param return_level:
    @return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_0 = time.time()
    if return_level == 'residue':
        n_dists = len(all_embeddings)
        # batch embeddings by grouping them together and keeping track of lengths. Push to torch and gpu
        batches, sections = create_batches(all_embeddings)
        torch_batches = [torch.from_numpy(emb1)[None, ...].to(device) for emb1 in batches]

        # Populate small blocks that correspond to distances between batches.
        block_dists = []
        for i, (torch_batch_row, sections_row) in enumerate(
                tqdm(zip(torch_batches, sections), total=len(torch_batches))):
            row_dists = []
            for j, (torch_batch_col, sections_col) in enumerate(zip(torch_batches, sections)):
                if j < i:
                    continue
                # Compute pairwise dists and split into blocks corresponding to individual graphs
                local_dists = torch.cdist(torch_batch_row, torch_batch_col)
                chunks_row = torch.split(local_dists, sections_row, dim=1)
                all_chunks = [torch.split(chunk_row, sections_col, dim=2) for chunk_row in chunks_row]

                # Turn each small block into one scalar value and store in a small_dist block of the return matrix
                small_dist = torch.zeros(size=(len(sections_row), len(sections_col)))
                for k, small_row in enumerate(all_chunks):
                    for l, small_block in enumerate(small_row):
                        mindist = torch.min(small_block)
                        small_dist[k, l] = mindist
                row_dists.append(small_dist)
            block_dists.append(row_dists)

        # Concatenate the block results using padding to take the j<i condition
        row_dists = []
        for i in range(len(block_dists)):
            row = block_dists[i]
            stacked_row = torch.cat(row, dim=1)
            p2d = (n_dists - stacked_row.shape[1], 0)
            out = pad(stacked_row, p2d, "constant")
            row_dists.append(out)
        all_dists = torch.cat(row_dists, dim=0)

        # Filter values beneath the diagonal (just the blocks) and complete with transpose
        all_dists = torch.triu(all_dists, diagonal=1)
        all_dists = all_dists + all_dists.T
    else:
        # Graph-level computation
        all_embeddings = np.stack(all_embeddings)[None, ...]
        all_embeddings = torch.from_numpy(all_embeddings).to(device)
        all_dists = torch.cdist(all_embeddings, all_embeddings)[0]
    all_dists = all_dists.cpu().numpy()
    print('computed cdist in ', time.time() - t_0)
    return all_dists


def process_hdf5(infilename='data/hdf5/embeddings_scope.hdf5',
                 return_level='graph',
                 name_suffix='512',
                 out_dir='analysis/data/pickles'):
    all_pickles = os.path.join(out_dir, f'scope_results_{name_suffix}_{return_level}.p')
    dict_pickle = os.path.join(out_dir, f'scope_dict_result_{name_suffix}_{return_level}.p')

    all_systems, all_embeddings = read_embeddings(infilename=infilename, return_level=return_level)

    # Compute the pairwise distances
    # all_systems = all_systems[:10]
    # all_embeddings = all_embeddings[:10]
    dists = get_pairwise_dist(all_embeddings, return_level=return_level)

    res_dict = {}
    for i, system in enumerate(tqdm(all_systems)):
        sorter = np.argsort(dists[i])
        res_dict[system] = [all_systems[j] for j in sorter]

    # Save the results for sharing results.
    pickle.dump((all_systems, all_embeddings, dists, res_dict), open(all_pickles, 'wb'))
    pickle.dump(res_dict, open(dict_pickle, 'wb'))


if __name__ == '__main__':
    pass
    # setup variables
    # infilename = 'data/hdf5/embeddings_scope_3072.hdf5'
    infilename, name_suffix = 'data/hdf5/embeddings_scope.hdf5', '512'
    # infilename, name_suffix = 'data/hdf5/embeddings_scope_32.hdf5', '32'
    # return_level = 'graph'
    return_level = 'residue'
    out_dir = 'analysis/data/pickles'
    process_hdf5(infilename=infilename, return_level=return_level, out_dir=out_dir, name_suffix=name_suffix)
