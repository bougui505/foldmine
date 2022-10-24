import os
import sys

import argparse
import easydict
import glob
import h5py
import jinja2
from jinja2 import meta
from tqdm import tqdm
import yaml

import torch
from torchdrug import models, utils

from loader import PDBdataset, Collater
from utils import pdbchain_to_hdf5path


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


def load_gearnet_model(cfg, weights_path='data/model/latest_mc_gearnet_edge.pth'):
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    # Model Loading
    model_dict = cfg.task.model.model
    model_dict.pop('class')
    model = models.GeometryAwareRelationalGraphNeuralNetwork(**model_dict)
    state_dict = torch.load(weights_path, map_location='cpu')
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


def compute_embeddings(model, dataloader, out_path, device='cpu'):
    with h5py.File(out_path, 'a') as f:
        for i, (names, batched_graphs) in enumerate(tqdm(dataloader)):
            # Sometimes all preprocessing failed
            if names is None:
                continue
            try:
                batched_graphs = batched_graphs.to(device)
                # Now make inference and collect residues and graph embeddings.
                with torch.no_grad():
                    out = model(graph=batched_graphs, input=batched_graphs.residue_feature.float())
                # graph_feat = out['graph_feature'][:, -512:].to('cpu')
                # node_feat = out['node_feature'][:, -512:].to('cpu')
                graph_feat = out['graph_feature'].to('cpu')
                node_feat = out['node_feature'].to('cpu')
                res_ids = batched_graphs.residue_number.to('cpu')

                # Split the result per batch.
                successive_lengths = batched_graphs.num_residues
                split_ids = torch.split(res_ids, successive_lengths)
                split_node_feat = torch.split(node_feat, successive_lengths)
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


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--pdb_path', help="Path to pdb database", default='data/pdb_chainsplit')
    parser.add_argument('-o', "--out_path", help="Path to the out hdf5", default='test.hdf5')
    args, unparsed = parser.parse_known_args()

    # Load model
    yml_file = 'data/model/config.yml'
    weights_path = 'data/model/latest_mc_gearnet_edge.pth'
    cfg = parse_torchdrug_yaml(yml_file=yml_file)
    model = load_gearnet_model(cfg=cfg, weights_path=weights_path)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # Finally, get the embeddings and store them in a hdf5
    compute_embeddings(model=model, dataloader=dataloader, out_path=args.out_path, device=device)
