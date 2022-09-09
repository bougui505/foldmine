#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import glob
import gzip
import json
import os
from pymol import cmd
import requests
import torch
import wget


# GET HOMOLOGS

def parse_blast(path_to_blast='blast_results.txt.gz',
                parsed='../data/homologs.txt.gz',
                remove_self=True):
    """
    Should take a few minutes

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
    splitter = Splitter(outdir=outdir, pdb_mmtf_path=indir)
    splitter = torch.utils.data.DataLoader(splitter, num_workers=os.cpu_count(), collate_fn=lambda x: x)
    for _ in splitter:
        pass


def filter_homologs_subset(ids_to_keep='../data/foldseek_list.txt',
                           original_homologs='../data/homologs.txt.gz',
                           filtered_homologs='../data/homologs_foldseek.txt.gz'):
    with open(ids_to_keep) as protfile:
        protset = set()
        for e in protfile.readlines():
            pdb = e.strip()[5:9]
            chain = e.strip()[9]
            if chain == ".":
                continue
            protset.add(f'{pdb}_{chain}')

    with gzip.open(original_homologs, 'r') as homologs:
        with gzip.open(filtered_homologs, 'wt') as outfile:
            for line in homologs.readlines():
                line = line.decode()
                pdbchain = line.split()[0]
                if pdbchain in protset:
                    outfile.write(line)


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

    # Required files to run this preprocessing :
    # path to blast output : 'blast_results.txt.gz'
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
