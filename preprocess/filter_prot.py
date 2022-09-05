#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import gzip

with open('../data/protein_chains.txt') as protfile:
    protset = set([e.strip() for e in protfile.readlines()])

with gzip.open('../data/homologs.txt.gz', 'r') as homologs:
    # with open('test.txt') as homologs:
    with gzip.open('../data/homologs_filtered.txt.gz', 'wt') as outfile:
        for line in homologs.readlines():
            line = line.decode()
            pdbchain = line.split()[0].upper()
            if pdbchain in protset:
                outfile.write(line)
