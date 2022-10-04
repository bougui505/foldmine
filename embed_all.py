import os
import sys

import argparse
import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torchdrug import models, layers, transforms, data, utils
from torchdrug.layers import geometry
import h5py


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
    """
    >>> dataset = PDBdataset()
    >>> dataset.__getitem__(0)
    (Data(), Data())
    >>> dataset.__getitem__(1000)
    154
    ...
    (Data(edge_index=[2, 728], node_id=[154], num_nodes=154, x=[154, 20]), ...)
    """

    def __init__(self, chainlist, data_path='data/pdb_chainsplit'):
        self.chainlist = chainlist
        self.data_path = data_path
        self.protein_view_transform = transforms.ProteinView(view='residue')

    def __len__(self):
        return len(self.chainlist)

    def __getitem__(self, index):
        pdb = self.chainlist[index]

        # Build an initial residue graph from the pdb
        # Weird castings, but seems to be necessary to use the transform.
        # Seems to be a pain to get the graph from the pdb without using the transform...
        td_protein = data.Protein.from_pdb(pdb_file=pdb)
        item = {"graph": td_protein}
        td_graph = self.protein_view_transform(item)['graph']
        return pdb, td_graph

        # That's quite easy, you can pack the proteins together and make inference over the packed graphs...


class Collater:
    def __init__(self):
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[
                                                                     geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                                 edge_feature="gearnet")

    def collate_block(self, samples):
        names = [sample[0] for sample in samples]
        td_prots = [sample[1] for sample in samples]
        batched_proteins = data.Protein.pack(td_prots)
        batched_graphs = self.graph_construction_model(batched_proteins)
        return names, batched_graphs


def split_res(tensor_to_split, succesive_lengths):
    succesive_lengths = [0] + list(succesive_lengths)
    slices = list()
    for low, high in zip(succesive_lengths, succesive_lengths[1:]):
        slices.append(tensor_to_split[low:high])
    return slices


if __name__ == '__main__':
    cfg = parse_torchdrug_yaml()
    model = load_model(cfg=cfg)

    chainlist = [chain_name.strip() for chain_name in open('chain_list.txt', 'r').readlines() if len(chain_name) > 1]

    dataset = PDBdataset(chainlist=chainlist)
    collater = Collater()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=2,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=collater.collate_block)

    for i, (names, batched_graphs) in enumerate(dataloader):
        # Now make inference and collect residues and graph embeddings.
        with torch.no_grad():
            out = model(graph=batched_graphs, input=batched_graphs.residue_feature.float())
        graph_feat = out['graph_feature']
        node_feat = out['node_feature']
        successive_lengths = batched_graphs.num_residues
        res_ids = batched_graphs.residue_number
        split_ids = split_res(res_ids, successive_lengths)
        split_node_feat = split_res(node_feat, successive_lengths)
        for pdb, res_ids, embeddings in zip(names, split_ids, split_node_feat):
             print(pdb)
             print(res_ids.shape)
             print(embeddings.shape)
