#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import torch
import glob
from pymol import cmd
from misc import randomgen
from misc.pytorch import torchify
import tqdm


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> # Generate random datapoints for testing
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb')
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     print(len(batch))
    ...     print([coords.shape for coords in batch])
    ...     if i == 0:
    ...         break
    4
    [(2209, 3), (9848, 3), (9360, 3), (1118, 3)]

    # with a batch size of 1:
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = dataiter.next()
    >>> len(batch)
    1
    >>> coords = batch[0]
    >>> coords.shape
    (2209, 3)

    # with return_name=True
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', return_name=True)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = dataiter.next()
    >>> name, coords = batch[0]
    >>> coords.shape
    (2209, 3)
    >>> name
    '/media/bougui/scratch/pdb/a9/pdb5a96.ent.gz'
    """
    def __init__(self, pdbpath='pdb_chainsplit', selection='all', return_name=False):
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.pdb.gz')
        self.return_name = return_name
        self.selection = selection
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        # coords = cmd.get_coords(selection=f'{pymolname} and {self.selection}')
        nres = cmd.select(f'{pymolname} and polymer.protein and name CA')
        cmd.delete(pymolname)
        return pdbfile, nres


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.tetmod()
        sys.exit()

    dataset = PDBdataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    pbar = tqdm.tqdm(total=len(dataloader))
    with open('pdb_chainsplit_nres.txt', 'w') as outfile:
        for i, out in enumerate(dataloader):
            pdbfile, nres = out[0]
            outfile.write(f'{pdbfile} {nres}\n')
            outfile.flush()
            pbar.update(1)
