import os

import glob
import gzip
import h5py

import torch
from torchdrug import layers, transforms, data

from utils import pdbfile_to_hdf5path, pdbfile_to_chain


class PDBdataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/pdb_chainsplit', chain_list=None, out_path=None):
        if chain_list is None:
            self.chainlist = glob.glob(os.path.join(data_path, "**", "*.pdb*"))
        else:
            self.chainlist = [os.path.join(data_path, chain) for chain in chain_list]

        # Optionally filter chainlist to remove existing entries in the hdf5
        if out_path is not None:
            with h5py.File(out_path, 'a') as f:
                self.chainlist = [chain for chain in self.chainlist if pdbfile_to_hdf5path(chain) not in f]

        self.data_path = data_path
        self.protein_view_transform = transforms.ProteinView(view='residue')

    def __len__(self):
        return len(self.chainlist)

    def __getitem__(self, index):
        pdb = self.chainlist[index]
        pdb_name = pdbfile_to_chain(pdb)
        td_graph = None

        # Torchdrug can only process pdb files. One needs to temporary save extracted pdb.gz
        compressed = pdb.endswith('.gz')
        if compressed:
            pdb_name = os.path.basename(pdb).split('.')[0]
            temp_pdb_name = f'/dev/shm/{pdb_name}'
            with gzip.open(pdb, 'r') as f:
                with open(temp_pdb_name, 'wb') as new_f:
                    lines = f.readlines()
                    new_f.writelines(lines)

        # Build the graph
        try:
            # Build an initial residue graph from the pdb
            # Weird castings, but seems to be necessary to use the transform.
            # Seems to be a pain to get the graph from the pdb without using the transform...
            td_protein = data.Protein.from_pdb(pdb_file=pdb if not compressed else temp_pdb_name)
            item = {"graph": td_protein}
            td_graph = self.protein_view_transform(item)['graph']
        except Exception as e:
            print(e)
            print(pdb)
        finally:
            if compressed:
                os.remove(temp_pdb_name)
        return pdb_name, td_graph


class Collater:
    def __init__(self):
        self.graph_construction_model = layers.GraphConstruction(node_layers=[layers.geometry.AlphaCarbonNode()],
                                                                 edge_layers=[
                                                                     layers.geometry.SpatialEdge(radius=10.0,
                                                                                                 min_distance=5),
                                                                     layers.geometry.KNNEdge(k=10, min_distance=5),
                                                                     layers.geometry.SequentialEdge(max_distance=2)],
                                                                 edge_feature="gearnet")

    def collate_block(self, samples):
        # Filter failed preprocessing and pack graphs.
        samples = [sample for sample in samples if sample[1] is not None]
        if len(samples) == 0:
            return None, None
        try:
            names = [sample[0] for sample in samples]
            td_prots = [sample[1] for sample in samples]
            batched_proteins = data.Protein.pack(td_prots)
            batched_graphs = self.graph_construction_model(batched_proteins)
        except:
            return None, None
        return names, batched_graphs
