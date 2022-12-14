#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
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

from annoy import AnnoyIndex
import h5py
import os
import time
import tqdm

from utils import len_hdf5


class Mapping(object):
    """
    # >>> with Mapping('test.h5') as mapping:
    # ...     mapping.add(0, 'toto')

    >>> mapping = Mapping('test.h5')
    >>> mapping.add(0, 'toto')

    >>> mapping.number_to_name(0)
    'toto'
    >>> mapping.name_to_number('toto')
    0

    >>> mapping = Mapping('test2.h5', hash_func_number=hash_func_number, hash_func_name=hash_func_name)
    >>> mapping.add(0, 'toto')
    >>> mapping.number_to_name(0)
    'toto'

    >>> mapping.name_to_number('toto')
    0
    """

    def __init__(self, h5fname, hash_func_number=lambda x: x, hash_func_name=lambda x: x, verbose=False):
        self.verbose = verbose
        self.hash_func_number = hash_func_number
        self.hash_func_name = hash_func_name
        self.h5fname = h5fname
        self.h5f = h5py.File(self.h5fname, 'a')
        self.h5f.require_group('name_to_number')
        self.h5f.require_group('number_to_name')

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tra):
        if typ is None:
            self.__del__()

    def __del__(self):
        self.h5f.close()
        if self.verbose:
            print(f'{self.h5fname} closed')

    def add(self, number, name):
        """
        """
        number_hash = self.hash_func_number(number)
        group = self.h5f['number_to_name']
        leaf = group.require_group(str(number_hash))
        leaf.attrs[str(number)] = name

        name_hash = self.hash_func_name(name)
        group = self.h5f['name_to_number']
        leaf = group.require_group(name_hash)
        leaf.attrs[name] = number

    def number_to_name(self, number):
        number_hash = self.hash_func_number(number)
        group = self.h5f['number_to_name'][str(number_hash)]
        return group.attrs[str(number)]

    def name_to_number(self, name):
        name_hash = self.hash_func_name(name)
        return self.h5f['name_to_number'][name_hash].attrs[str(name)]


def hash_func_number(number):
    """
    >>> number = 123456789
    >>> hash_func_number(number)
    '123/456/789'
    >>> number = 123456
    >>> hash_func_number(number)
    '0/123/456'
    >>> number = 56
    >>> hash_func_number(number)
    '0/0/56'
    >>> number = 0
    >>> hash_func_number(number)
    '0/0/0'
    """
    number = int(number)
    return f'{number // 10 ** 6}/{(number // 10 ** 3) % 10 ** 3}/{number % 10 ** 3}'


def hash_func_name(name):
    """
    >>> hash_func_name('1ycr_A')
    'yc/1ycr_A'
    >>> hash_func_name('1ycr_A_34')
    'yc/1ycr_A'
    """
    pdb, chain = name.split('_')[:2]
    return f'{pdb[1:3]}/{pdb}_{chain}'


class NNindex(object):
    """
    >>> index = NNindex('index_test')
    >>> index.build('data/small.hdf5')
    >>> nnames, dists = index.query('7f14_B', k=3)
    >>> nnames
    ['7f14_B', '7f13_B', '3f0g_E']

    >>> index = NNindex('index_res_test')
    >>> index.build('data/small.hdf5', residue_level=True)
    """

    def __init__(self, index_dirname, dim=512):
        self.dim = dim
        self.index = None
        self.index_dirname = index_dirname
        if not os.path.isdir(index_dirname):
            os.makedirs(index_dirname)
        self.annoyfilename = f'{index_dirname}/annoy.ann'
        self.mappingfilename = f'{index_dirname}/mapping.h5'
        self.metric = 'euclidean'

    def build(self, infilename, n_trees=10, residue_level=False):
        self.index = AnnoyIndex(self.dim, self.metric)
        self.index.on_disk_build(self.annoyfilename)
        i = 0
        annoy_mapping = Mapping(self.mappingfilename, hash_func_number=hash_func_number, hash_func_name=hash_func_name)
        with h5py.File(infilename, 'r') as f:
            n = len_hdf5(f)
            pbar = tqdm.tqdm(total=n)
            for key in f.keys():
                for system in f[key].keys():  # iterate pdb systems
                    if residue_level:
                        res_embs = f[key][system]['res_embs'][()]
                        res_ids = f[key][system]['res_ids'][()]
                        for resid, resemb in zip(res_ids, res_embs):
                            self.index.add_item(i, resemb)
                            annoy_mapping.add(i, f'{system}_{resid}')
                            i += 1
                    else:
                        v = f[key][system]['graph_embs'][()]
                        self.index.add_item(i, v)
                        annoy_mapping.add(i, system)
                        i += 1
                    pbar.update(1)
            pbar.close()
        t_0 = time.time()
        self.index.build(n_trees)
        print(f'Index built in {time.time() - t_0:.3f}s')

    def query(self, name, k=1):
        """
        """
        if self.index is None:
            self.index = AnnoyIndex(self.dim, self.metric)
            self.index.load(self.annoyfilename)
        with Mapping(self.mappingfilename, hash_func_number=hash_func_number, hash_func_name=hash_func_name) as mapping:
            ind = mapping.name_to_number(name)
            knn, dists = self.index.get_nns_by_item(ind, n=k, include_distances=True)
            nnames = []
            for i in knn:
                nnames.append(mapping.number_to_name(i))
            return nnames, dists


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hdf5', help='HDF5 for building index')
    parser.add_argument('--out', help='Output directory for the index')
    parser.add_argument('--trees', help='Numbers of trees to build', type=int, default=10)
    parser.add_argument('--residue', help='Residue level', action='store_true')
    parser.add_argument('--query', help='')
    parser.add_argument('-k', help='k-neighbors to return', type=int, default=3)
    parser.add_argument('--index', help='Directory with index')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()

    if args.hdf5 is not None:
        index = NNindex(args.out)
        index.build(args.hdf5, residue_level=args.residue, n_trees=args.trees)

    if args.query is not None:
        index = NNindex(args.index)
        nnames, dists = index.query(args.query, k=args.k)
        for name, dist in zip(nnames, dists):
            print(name, dist)
