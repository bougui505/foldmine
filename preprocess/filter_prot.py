#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import gzip

# with open('../data/protein_chains.txt') as protfile:
#     protset = set([e.strip() for e in protfile.readlines()])

with open('../data/foldseek_list.txt') as protfile:
    protset = set()
    for e in protfile.readlines():
        pdb = e.strip()[5:9].upper()
        chain = e.strip()[9].upper()
        if chain == ".":
            continue
        protset.add(f'{pdb}_{chain}')

with gzip.open('../data/homologs.txt.gz', 'r') as homologs:
    # with open('test.txt') as homologs:
    with gzip.open('../data/homologs_foldseek.txt.gz', 'wt') as outfile:
        for line in homologs.readlines():
            line = line.decode()
            pdbchain = line.split()[0].upper()
            if pdbchain in protset:
                outfile.write(line)
