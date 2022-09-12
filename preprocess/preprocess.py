#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import glob
import gzip
import json
import os
from pymol import cmd
import requests
import torch
import tqdm
import wget
import sys

sys.path.append('..')
import protein


# GET HOMOLOGS

def parse_blast(path_to_blast='blast_results.txt.gz',
                parsed='../data/homologs.txt.gz',
                remove_self=True):
    """
    The blast output format 6 starts with the two systems that got compared and include additional information.
    The first one is contiguous (all 1ycr queries)...
    We just go through the 2 million lines and when the first one changes,
        we dump all its homologs as a line of 'parsed'

    Should take a few minutes.

    @param path_to_blast:
    @param remove_self:
    @return:
    """
    with gzip.open(path_to_blast, 'r') as homologs:
        with gzip.open(parsed, 'wt') as outfile:
            # Parse each line of the blast and group by homologs
            previous = None
            homologs_list = []
            for i, line in enumerate(homologs):
                if not i % 1000000:
                    print(f"{i}/208646142")

                line = line.decode()

                # Remove header
                if line.startswith("#"):
                    continue
                splitline = line.split()
                pdbchain = splitline[0]
                homolog = splitline[1]
                # print(previous, pdbchain, homolog)
                # If we encounter a new system and the previous had homologs, write a line and clear homologs
                if pdbchain != previous:
                    if homologs_list:
                        outstring = ' '.join([previous] + homologs_list) + '\n'
                        outfile.write(outstring)
                    homologs_list = []

                # For this new line, populate homolog if it's not a copy of self.
                if homolog != pdbchain or not remove_self:
                    homologs_list.append(homolog)
                previous = pdbchain
            # After the for loop, dump the file.
            if homologs_list:
                outstring = ' '.join([previous] + homologs_list) + '\n'
                outfile.write(outstring)


# GET PROTEIN CHAINS

def get_entitites_from_pdb():
    """
    This is simply a PDB Search request to obtain all PDB containing a protein chain.
    @return:
    """
    url = 'https://search.rcsb.org/rcsbsearch/v2/query?json={"query": {"type": "terminal","label": "text", ' \
          '"service": "text","parameters": {"attribute": "entity_poly.rcsb_entity_polymer_type","operator": ' \
          '"exact_match","negation": false,"value": "Protein"}},"request_options": {"return_all_hits": true},' \
          '"return_type": "polymer_entity"}'
    r = requests.get(url)
    json_result = r.json()
    outfile = 'protein_list.json'
    with open(outfile, 'w') as file:
        json.dump(json_result, file)


def chunks(L, n):
    return [L[x:x + n] for x in range(0, len(L), n)]


def get_chain_names(jsonfile_query='protein_list.json', nbatch=100):
    """
    Because of the two-step downloading process of the PDB, one needs to transform entity ids into their chain names.
    We batch those requests to avoid having too long a URL while avoiding to get blacklisted from the PDB
    @param jsonfile_query:
    @param nbatch:
    @return:
    """
    jsonfile = open(jsonfile_query, 'r')
    data = json.loads(jsonfile.read())
    idlist = [i['identifier'] for i in data['result_set']]
    batches = chunks(idlist, nbatch)

    with open('protein_chains.txt', 'w') as outfile:
        for batch in batches:
            query = '{polymer_entities(entity_ids: ["%s"]) {rcsb_id entry {rcsb_entry_container_identifiers {entry_id}}    polymer_entity_instances {rcsb_polymer_entity_instance_container_identifiers {auth_asym_id}}}}' % '","'.join(
                batch)
            url = f'https://data.rcsb.org/graphql?query={query}'
            r = requests.get(url)
            for e in r.json()['data']['polymer_entities']:
                pdb = e['entry']['rcsb_entry_container_identifiers']['entry_id']
                chain = e['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers'][
                    'auth_asym_id']
                outfile.write(f'{pdb.lower()}_{chain}\n')


def download_mmtf(protein_chains='protein_chains.txt', base_dir='../data/pdb_mmtf', redownload=False):
    """
    Download the PDB containing protein chains in the MMTF format.
    The request has to be done one at a time, otherwise one needs to use an Apache server to decode the payload.
    @param protein_chains:
    @param base_dir:
    @param redownload:
    @return:
    """
    with open(protein_chains, 'r') as protein_chains:
        for i, protchain in enumerate(protein_chains):
            if i > 3:
                break
            pdb, chain = protchain[:4].lower(), protchain[6:]
            outdir = os.path.join(base_dir, pdb[1:3])
            outfile_path = os.path.join(outdir, f'{pdb}.mmtf')
            os.makedirs(outdir, exist_ok=True)
            if not os.path.exists(outfile_path) or redownload:
                wget.download(f'https://mmtf.rcsb.org/v1.0/full/{pdb}', outfile_path)


class Splitter(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB
    """

    def __init__(self, outdir='pdb_chainsplit', pdb_mmtf_path='../data/pdb_mmtf'):
        self.list_IDs = glob.glob(f'{pdb_mmtf_path}/**/*.mmtf')
        self.outdir = outdir
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        pdbcode = os.path.splitext(os.path.basename(pdbfile))[0]
        dirname = f'{self.outdir}/{pdbcode[1:3]}'
        os.makedirs(dirname, exist_ok=True)

        # If we ran that already, just skip.
        # This is a bit risky as we could have had partial results.
        ls = os.listdir(dirname)
        for name in ls:
            if name.startswith(pdbcode):
                return

        # Load and iterate to get all chains
        pymolname = index
        cmd.load(filename=pdbfile, object=pymolname)
        chains = cmd.get_chains(f'{pymolname} and polymer.protein')
        for chain in chains:
            fn = f'{dirname}/{pdbcode}_{chain}.pdb.gz'
            cmd.save(fn, selection=f'{pymolname} and polymer.protein and chain {chain}')
        cmd.delete(pymolname)


def get_split_chains(outdir='../data/pdb_chainsplit', indir='../data/pdb_mmtf'):
    """
    Iterate over indir (whole mmtf pdb) and dump smaller pdbs split by chain
    Careful, some PDB files are too big and will still be buggy.
    However, we are stuck with the PDB format because of BioPandas.
    @param outdir:
    @param indir:
    @return:
    """
    splitter = Splitter(outdir=outdir, pdb_mmtf_path=indir)
    splitter = torch.utils.data.DataLoader(splitter, num_workers=os.cpu_count(), collate_fn=lambda x: x)
    for _ in splitter:
        pass


# FILTER HOMOLOGS.TXT FILE BASED ON SCOPE40

def get_mapping_scope40(index_file='../data/dir.des.scope.2.08-stable.txt'):
    """
    Get a mapping from filename to the chain they represent
    @param index_file:
    @return:
    """
    mapping = {}
    with open(index_file, 'r') as index_file:
        for line in index_file:
            splitted_line = line.split()
            if len(splitted_line) > 3 and not splitted_line[0] == "#" and not splitted_line[3] == '-':
                filename = splitted_line[3]
                pdb, chain = splitted_line[4], splitted_line[5].split(':')[0]
                mapping[filename] = (pdb, chain)
    return mapping


def filter_homologs_subset(index_file='../data/dir.des.scope.2.08-stable.txt',
                           original_homologs='../data/homologs.txt.gz',
                           filtered_homologs='../data/homologs_foldseek.txt.gz'):
    """
    Subset a homolog file to keep only lines that have an anchor in ids_to_keep
    @param ids_to_keep:
    @param original_homologs:
    @param filtered_homologs:
    @return:
    """
    scope_mapping = get_mapping_scope40(index_file)
    protset = set(['_'.join([pdb, chain]) for pdb, chain in scope_mapping.values()])

    with gzip.open(original_homologs, 'r') as homologs:
        with gzip.open(filtered_homologs, 'wt') as outfile:
            lines = homologs.readlines()
            for line in tqdm.tqdm(lines):
                line = line.decode()
                pdbchain = line.split()[0]
                if pdbchain in protset:
                    outfile.write(line)


# REMOVE BUGGY OR NONSENSICAL FILES OR HUGE SYSTEMS

class FilterDataset(torch.utils.data.Dataset):
    """
    Just a dataset to go through a list of chains
    """

    def __init__(self, pdb_chain_list):
        self.path_list = pdb_chain_list
        self.graph_builder = protein.GraphBuilder()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # Get a Scope40 chain and its homologs
        pdb_chain = self.path_list[index]
        pdb, chain = pdb_chain.split('_')
        try:
            graph, distmat = self.graph_builder.build_graph(pdbcode=pdb, chain=chain)
        except Exception as e:  # (ValueError, KeyError, FileNotFoundError):
            # Often a modified residue
            print(e)
            return 0, pdb_chain, None
        return 1, pdb_chain, distmat


def ensure_loadable_and_small(homologs_file='../data/homologs_foldseek.txt.gz',
                              homologs_file_clean='../data/homologs_foldseek_clean.txt.gz',
                              max_res=600):
    """
    Filter out lines of the homolog file that will raise errors.

    - Iterate through the homolog file to get a set of used protein chains
    - Loop over these chains and try to load them. Put the successful and not too big ones in a set
    - Loop over the homolog file once again, removing buggy anchors lines and lines without enough admissible homologs.
    @param homologs_file:
    @param homologs_file_clean:
    @param max_res:
    @return:
    """
    unique_chains = set()
    with gzip.open(homologs_file, 'r') as homologs_file:
        for line in homologs_file:
            all_pdbs_line = line.decode().strip().split()
            unique_chains.update(all_pdbs_line)
    dataset = FilterDataset(list(unique_chains))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=os.cpu_count(),
                                             collate_fn=lambda x: x)
    pbar = tqdm.tqdm(total=len(dataloader))
    admissible = set()
    for i, out in enumerate(dataloader):
        failed, chain_name, distmat = out[0]
        if not failed:
            print(distmat.shape)
            nres = len(distmat)
            if nres < max_res:
                admissible.add(chain_name)
        pbar.update(1)
        # debug
        if i > 10:
            break
    print(admissible)
    with gzip.open(homologs_file, 'r') as homologs_file:
        with gzip.open(homologs_file_clean, 'wt') as clean_homologs_file:
            for line in homologs_file:
                all_pdbs_line = line.decode().strip().split()

                # The reference should be loadable
                anchor = all_pdbs_line[0]
                if not anchor in admissible:
                    continue

                # Then the homologs are filtered to be admissible
                # If sufficiently many remain, we write that in the clean file
                admissible_homologs = [pdb_chain for pdb_chain in all_pdbs_line[1:] if pdb_chain in admissible]
                if len(admissible_homologs) > 0:
                    clean_line = ' '.join([anchor] + admissible_homologs)
                    clean_homologs_file.write(clean_line)


if __name__ == '__main__':
    pass
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--parse_blast', action='store_true')
    parser.add_argument('--query_pdb', action='store_true')
    parser.add_argument('--download_mmtf', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    # ensure_loadable_and_small()
    # Required files to run this preprocessing :
    # path to blast output : 'blast_results.txt.gz'
    # path to scope specifications : '../data/dir.des.scope.2.08-stable.txt'
    # path to scope ids : '../data/foldseek_list.txt'

    # Get all homologs
    if args.all or args.parse_blast:
        parse_blast()

    # Get all protein chains names in a .txt format
    if args.all or args.query_pdb:
        get_entitites_from_pdb()
        get_chain_names()

    # Get the mmtf structures of all proteins containing these chains.
    if args.all or args.download_mmtf:
        download_mmtf()

    # Split all these structures into pdb containing only one chain.
    if args.all or args.split:
        get_split_chains()

    # Filter to keep only the proteins present in SCOPe40
    if args.all or args.filter:
        filter_homologs_subset()
