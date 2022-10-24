import os
import sys

import h5py
import numpy as np
import pickle
import time
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from makeindex import len_hdf5


# Read the embeddings to retrieve all the graph level embeddings
def read_embeddings(infilename='data/hdf5/embeddings_scope.hdf5', return_level='residue', early_stop=None):
    """
    Read an embeddings hdf5 file and return the list of systems along with their embeddings
    at the res or the graph level
    @param infilename:
    @param return_level:
    @param early_stop:
    @return:
    """
    with h5py.File(infilename, 'r') as f:
        n = len_hdf5(f)
        pbar = tqdm(total=n)
        all_systems = []
        all_embeddings = []
        i = 0
        # all_embeddings.append(f['1d']['d1dlwa_']['res_embs'][()])
        # all_embeddings.append(f['2g']['d2gkma_']['res_embs'][()])
        # all_embeddings.append(f['1j']['d1j6wa_']['res_embs'][()])
        # all_embeddings.append(f['1h']['d1hywa_']['res_embs'][()])
        for key in f.keys():
            for system in f[key].keys():  # iterate pdb systems
                embs_to_get = 'res_embs' if return_level == 'residue' else 'graph_embs'
                v = f[key][system][embs_to_get][()]
                all_systems.append(system)
                all_embeddings.append(v)
                pbar.update(1)
                i += 1
            if early_stop is not None and i > early_stop:
                break
        pbar.close()
    return all_systems, all_embeddings


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


def train_compressor(vectors, out_dim, samples_to_use=100000, type='pca'):
    if type == 'umap':
        import umap
        reducer = umap.UMAP(n_neighbors=10, n_components=out_dim, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=out_dim)
    samples_to_use = min(len(vectors), samples_to_use)
    X_train = vectors[np.random.choice(len(vectors), size=samples_to_use, replace=False)]
    reducer.fit(X_train)
    return reducer


def pdbfile_to_chain(pdb_file):
    # # toto/tata/1ycr_A.pdb => 1ycr_A
    # pdb_name = os.path.basename(pdb_file).split('.')[0]

    # Careful with dots in name !
    # toto/tata/1ycr_A.pdb.gz => 1ycr_A.pd
    pdb_name = os.path.basename(pdb_file)[:-4]
    return pdb_name


def pdbchain_to_hdf5path(pdb_chain):
    # 1ycr_A => yc/1ycr_A
    pdb_dir = f"{pdb_chain[1:3]}/{pdb_chain}"
    return pdb_dir


def pdbfile_to_hdf5path(pdb_file):
    # toto/tata/1ycr_A.pdb.gz => yc/1ycr_A
    return pdbchain_to_hdf5path(pdbfile_to_chain(pdb_file))


def compress_hdf5(in_hdf5, out_hdf5, out_dim=32):
    all_systems, all_embeddings_graphs = read_embeddings(infilename=in_hdf5, return_level='graph')
    all_embeddings_graphs = np.stack(all_embeddings_graphs)
    graph_reducer = train_compressor(all_embeddings_graphs, out_dim=out_dim)
    all_embeddings_graphs_compressed = graph_reducer.transform(all_embeddings_graphs)
    print('compressed graphs')

    all_systems, all_embeddings_residues = read_embeddings(infilename=in_hdf5, return_level='residue')
    stacked_embeddings_residues = np.vstack(all_embeddings_residues)
    residue_reducer = train_compressor(stacked_embeddings_residues, out_dim=out_dim)
    with h5py.File(out_hdf5, 'a') as f:
        for i, (system, graph_emb, res_embs) in enumerate(tqdm(zip(all_systems,
                                                                   all_embeddings_graphs_compressed,
                                                                   all_embeddings_residues),
                                                               total=len(all_systems))):

            pdb_dir = pdbchain_to_hdf5path(system)

            pdbgrp = f.require_group(pdb_dir)
            compressed_res_embs = residue_reducer.transform(res_embs)
            datasets = {'graph_embs': graph_emb, 'res_embs': compressed_res_embs}
            for name, value in datasets.items():
                pdbgrp.create_dataset(name=name, data=value)


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

    # # large_hdf5 = 'data/hdf5/embeddings_scope.hdf5'
    large_hdf5 = 'data/hdf5/embeddings_scope_3072.hdf5'
    small_hdf5 = 'data/hdf5/embeddings_scope_32.hdf5'
    # compress_hdf5(in_hdf5=large_hdf5, out_hdf5=small_hdf5, out_dim=32)
