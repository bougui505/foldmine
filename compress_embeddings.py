import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from utils import pdbchain_to_hdf5path, read_embeddings


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
    large_hdf5 = 'data/hdf5/embeddings_scope_3072.hdf5'
    small_hdf5 = 'data/hdf5/embeddings_scope_32.hdf5'
    compress_hdf5(in_hdf5=large_hdf5, out_hdf5=small_hdf5, out_dim=32)
