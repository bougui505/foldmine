import gzip
import os
import sys

import argparse
import easydict
import glob
import h5py
import jinja2
import numpy as np
from jinja2 import meta
from tqdm import tqdm
import yaml

import torch
from torchdrug import models, layers, transforms, data, utils
from torchdrug.layers import geometry


def parse_torchdrug_yaml(yml_file='config.yml'):
    """
    This code makes the best of gearnet repo to make it work.
    It is all boilerplate code to load a yml into a config dict like file that is used by torchdrug.
    @param yml_file:
    @return:
    """

    def detect_variables(cfg_file):
        with open(cfg_file, "r") as fin:
            raw = fin.read()
        env = jinja2.Environment()
        ast = env.parse(raw)
        vars = meta.find_undeclared_variables(ast)
        return vars

    def load_config(cfg_file, context=None):
        with open(cfg_file, "r") as fin:
            raw = fin.read()
        template = jinja2.Template(raw)
        instance = template.render(context)
        cfg = yaml.safe_load(instance)
        cfg = easydict.EasyDict(cfg)
        return cfg

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="yaml configuration file", default='config.yml')
        parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
        args, unparsed = parser.parse_known_args()
        # get dynamic arguments defined in the config file
        vars = detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var, default="null")
        vars = parser.parse_known_args(unparsed)[0]
        vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

        return args, vars

    # Useful to parse the config file type
    args, vars = parse_args()
    cfg = load_config(yml_file, context=vars)
    return cfg


def load_model(cfg):
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    # Model Loading
    model_dict = cfg.task.model.model
    model_dict.pop('class')
    model = models.GeometryAwareRelationalGraphNeuralNetwork(**model_dict)
    state_dict = torch.load('data/model/latest_mc_gearnet_edge.pth', map_location='cpu')
    simple_state_dict = {}
    for k, v in state_dict['model'].items():
        corrected_key = remove_prefix(k, 'model.model.')
        if 'batch_norm_layers' in corrected_key:
            corrected_key = corrected_key.replace('batch_norm_layers', 'batch_norms')
        if 'mlp' in corrected_key or 'spatial_line' in corrected_key:
            continue
        simple_state_dict[corrected_key] = v
    model.load_state_dict(simple_state_dict)
    return model


class PDBdataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/pdb_chainsplit', chain_list=None):
        if chain_list is None:
            self.chainlist = glob.glob(os.path.join(data_path, "**", "*.pdb*"))
        else:
            self.chainlist = [os.path.join(data_path, chain) for chain in chain_list]
        self.data_path = data_path
        self.protein_view_transform = transforms.ProteinView(view='residue')

    def __len__(self):
        return len(self.chainlist)

    def __getitem__(self, index):
        pdb = self.chainlist[index]
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
            # print(e)
            print(pdb)
        finally:
            if compressed:
                os.remove(temp_pdb_name)
        return pdb, td_graph


class Collater:
    def __init__(self):
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[
                                                                     geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                                 edge_feature="gearnet")

    def collate_block(self, samples):
        # Filter failed preprocessing and pack graphs.
        samples = [sample for sample in samples if sample[1] is not None]
        if len(samples) == 0:
            return None, None
        names = [sample[0] for sample in samples]
        td_prots = [sample[1] for sample in samples]
        batched_proteins = data.Protein.pack(td_prots)
        batched_graphs = self.graph_construction_model(batched_proteins)
        return names, batched_graphs


def split_results(tensor_to_split, succesive_lengths):
    succesive_lengths = [0] + list(np.cumsum(to_numpy(succesive_lengths)))
    slices = list()
    for low, high in zip(succesive_lengths, succesive_lengths[1:]):
        slices.append(tensor_to_split[low:high])
    return slices


def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--pdb_path', help="Path to pdb database", default='data/pdb_chainsplit')
    parser.add_argument('-o', "--out_path", help="Path to the out hdf5", default='test.hdf5')
    args, unparsed = parser.parse_known_args()

    cfg = parse_torchdrug_yaml()
    model = load_model(cfg=cfg)

    dataset = PDBdataset(data_path=args.pdb_path, chain_list=['f3/1f3k_A.pdb'])
    collater = Collater()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=collater.collate_block)

    with h5py.File(args.out_path, 'a') as f:
        for i, (names, batched_graphs) in enumerate(tqdm(dataloader)):
            # Sometimes all preprocessing failed
            if names is None:
                continue

            # Now make inference and collect residues and graph embeddings.
            with torch.no_grad():
                out = model(graph=batched_graphs, input=batched_graphs.residue_feature.float())
            graph_feat = out['graph_feature']
            node_feat = out['node_feature']
            res_ids = batched_graphs.residue_number

            # Split the result per batch.
            successive_lengths = batched_graphs.num_residues
            split_ids = split_results(res_ids, successive_lengths)
            split_node_feat = split_results(node_feat, successive_lengths)

            # Append each system to the hdf5
            for pdb, graph_embs, res_ids, res_embs in zip(names, graph_feat, split_ids, split_node_feat):
                pdb = os.path.basename(pdb)
                pdb_dir = f"{pdb[1:3]}/{pdb}/"
                pdbgrp = f.require_group(pdb_dir)
                grp_keys = list(pdbgrp.keys())
                datasets = {'graph_embs': graph_embs, 'res_ids': res_ids, 'res_embs': res_embs}
                for name, value in datasets.items():
                    if not name in grp_keys:
                        print(value.shape)
                        pdbgrp.create_dataset(name=name, data=value)

    def test():
        with h5py.File('test.hdf5', 'a') as f:
            embs = f['f3/1f3k_A.pdb/res_embs']
            graph_embs = f['f3/1f3k_A.pdb/graph_embs']
            res_ids = f['f3/1f3k_A.pdb/res_ids']
            print(embs.shape)
            print(graph_embs.shape)
            print(res_ids.shape)
    # test()