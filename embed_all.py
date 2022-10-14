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


def parse_torchdrug_yaml(yml_file='data/model/config.yml'):
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
        vars = detect_variables(yml_file)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var, default="null")
        vars = parser.parse_known_args(unparsed)[0]
        vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

        return vars

    # Useful to parse the config file type
    vars = parse_args()
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
    # print(type(state_dict['model']))
    # print(state_dict['model']['mlp.layers.0.weight'].keys())
    # sys.exit()
    model.load_state_dict(simple_state_dict)
    return model


def pdbfile_to_chain(pdb_file):
    # toto/tata/1ycr_A.pdb.gz => 1ycr_A
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    return pdb_name


def pdbchain_to_hdf5path(pdb_chain):
    # 1ycr_A => yc/1ycr_A
    pdb_dir = f"{pdb_chain[1:3]}/{pdb_chain}"
    return pdb_dir


def pdbfile_to_hdf5path(pdb_file):
    # toto/tata/1ycr_A.pdb.gz => yc/1ycr_A
    return pdbchain_to_hdf5path(pdbfile_to_chain(pdb_file))


class PDBdataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/pdb_chainsplit', chain_list=None, out_path=None):
        if chain_list is None:
            self.chainlist = glob.glob(os.path.join(data_path, "**", "*.pdb*"))
        else:
            self.chainlist = [os.path.join(data_path, chain) for chain in chain_list]

        # Optionally filter chainlist to remove existing entries in the hdf5
        if out_path is not None:
            with h5py.File(args.out_path, 'a') as f:
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


def split_results(tensor_to_split, succesive_lengths):
    succesive_lengths = [0] + list(np.cumsum(to_numpy(succesive_lengths)))
    slices = list()
    for low, high in zip(succesive_lengths, succesive_lengths[1:]):
        slices.append(tensor_to_split[low:high])
    return slices


def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()


def test():
    with h5py.File('test.hdf5', 'a') as f:
        embs = f['f3/1f3k_A.pdb/res_embs']
        graph_embs = f['f3/1f3k_A.pdb/graph_embs']
        res_ids = f['f3/1f3k_A.pdb/res_ids']
        print(embs.shape)
        print(graph_embs.shape)
        print(res_ids.shape)


def rename(data_path='data/scope_pdb/pdbstyle-2.01'):
    """
    Add pdb extension that is missing when downloading from scope40.
    @param data_path:
    @return:
    """
    chain_list = glob.glob(os.path.join(data_path, "**", "*.ent"))
    for old_path in chain_list:
        dirname = os.path.dirname(old_path)
        basename = os.path.basename(old_path)
        new_path = os.path.join(dirname, basename.split('.ent')[0] + '.pdb')
        # new_path = os.path.join(basename.split('.')[0] + '.pdb')
        os.rename(old_path, new_path)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--pdb_path', help="Path to pdb database", default='data/pdb_chainsplit')
    parser.add_argument('-o', "--out_path", help="Path to the out hdf5", default='test.hdf5')
    args, unparsed = parser.parse_known_args()

    # rename()

    # Load model
    cfg = parse_torchdrug_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(cfg=cfg)
    model = model.to(device)

    # Prepare the data
    # chain_list = ['d1q35a_.pdb']
    # print(chain_list)
    chain_list = None
    dataset = PDBdataset(data_path=args.pdb_path,
                         out_path=args.out_path,
                         chain_list=chain_list
                         )
    collater = Collater()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=os.cpu_count(),
                                             collate_fn=collater.collate_block)

    with h5py.File(args.out_path, 'a') as f:
        for i, (names, batched_graphs) in enumerate(tqdm(dataloader)):
            # Sometimes all preprocessing failed
            if names is None:
                continue
            try:
                batched_graphs = batched_graphs.to(device)
                # Now make inference and collect residues and graph embeddings.
                with torch.no_grad():
                    out = model(graph=batched_graphs, input=batched_graphs.residue_feature.float())
                graph_feat = out['graph_feature'][:, -512:].to('cpu')
                node_feat = out['node_feature'][:, -512:].to('cpu')
                res_ids = batched_graphs.residue_number.to('cpu')

                # Split the result per batch.
                successive_lengths = batched_graphs.num_residues
                split_ids = split_results(res_ids, successive_lengths)
                split_node_feat = split_results(node_feat, successive_lengths)
            except Exception as e:
                print(e)
                continue
            # Append each system to the hdf5
            for pdb, graph_embs, res_ids, res_embs in zip(names, graph_feat, split_ids, split_node_feat):
                pdb_dir = pdbchain_to_hdf5path(pdb)
                pdbgrp = f.require_group(pdb_dir)
                grp_keys = list(pdbgrp.keys())
                datasets = {'graph_embs': graph_embs, 'res_ids': res_ids, 'res_embs': res_embs}
                for name, value in datasets.items():
                    if not name in grp_keys:
                        pdbgrp.create_dataset(name=name, data=value)
