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
import os
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
# from misc.Timer import Timer

# TIMER = Timer(autoreset=True)


class Bench(object):

    def __init__(self,
                 scopfilename='data/dir.des.scope.2.01-stable.txt',
                 results='data/foldseekaln.gz',
                 results_pickle='data/foldseekresults.pickle',
                 scop40_list='data/scop40_list.txt'):
        """
        >>> bench = Bench()
        >>> bench.scop_dict
        {'d1ux8a_': 'a.1.1.1', 'd1dlwa_': 'a.1.1.1', 'd1uvya_': 'a.1.1.1', ...
        >>> bench.all_scops
        array([['a', '1', '1', '1'],
               ['a', '1', '1', '1'],
               ['a', '1', '1', '1'],
               ...,
               ['g', '91', '1', '1'],
               ['g', '92', '1', '1'],
               ['g', '93', '1', '1']], dtype='<U3')
        >>> bench.all_scops.shape
        (11211, 4)

        >>> bench.get_scop('d1a1xa_')
        ['b', '63', '1', '1']
        >>> query, neighbors = bench.get_result('d1a1xa_')
        >>> query
        ['b', '63', '1', '1']
        >>> neighbors
        ['d1jsga_', 'd3saoa_', 'd2qcka_', ...

        Get the sensitivity at the family level
        >>> bench.get_sensitivity('d1mkya2')
        0.2389937106918239

        Get the sensitivity at the superfamily level
        >>> bench.get_sensitivity('d1mkya2', level=2)
        0.00066711140760507

        Get the sensitivity at the fold level
        >>> bench.get_sensitivity('d1mkya2', level=1)
        -1
        """
        self.scopfilename = scopfilename
        self.results = results
        self.results_pickle = results_pickle
        self.scop_dict = self.__parse_scop__()
        self.scop40 = self._filter_scop_list(scop40_list)
        self.den_dict = dict()
        if os.path.exists(self.results_pickle):
            self.results_dict = pickle.load(open(self.results_pickle, 'rb'))
        else:
            self.results_dict = self.__parse_results__()

    @staticmethod
    def selection(scop, level, target):
        sel = (scop[:, :level + 1] == target[:level + 1]).all(axis=1)
        if level < 3:
            sel = np.logical_and(sel, scop[:, level + 1] != target[level + 1])
        return sel

    def _filter_scop_list(self, scop40_list):
        scop40_list = set(np.genfromtxt(scop40_list, dtype=str))
        scop40 = np.asarray([v.split('.') for k, v in self.scop_dict.items() if k in scop40_list])
        return scop40
        # all_scops = np.asarray([e.split('.') for e in self.scop_dict.values()])

    def __parse_scop__(self):
        """
        """
        data = np.genfromtxt(self.scopfilename, usecols=(2, 3), dtype=str)
        sel = (data[:, 1] != '-')
        data = data[sel]
        lookupdict = dict(zip(data[:, 1], data[:, 0]))
        return lookupdict

    def __parse_results__(self, max_rows=None):
        """
        """
        data = np.genfromtxt(self.results, usecols=(0, 1), dtype=str, max_rows=max_rows)
        p1 = data[:, 0]
        p2 = data[:, 1]
        inds = np.where(p1[:-1] != p1[1:])[0] + 1
        p1 = np.split(p1, inds)
        p2 = np.split(p2, inds)
        p1 = [e[0] for e in p1]
        out = dict(zip(p1, p2))
        pickle.dump(out, open(self.results_pickle, 'wb'))
        return out

    def get_scop(self, structure):
        """
        Returns: [Class, Fold, Superfamily, Family]
        """
        scop = self.scop_dict[structure]
        scop = scop.split('.')
        return scop

    def get_result(self, structure):
        """
        """
        neighbors = list(self.results_dict[structure])
        neighbors.remove(structure)
        query = self.get_scop(structure)
        # neighbors = [self.get_scop(e) for e in neighbors]
        return query, neighbors

    def get_sensitivity(self, structure, level=3):
        """
        Family: level=3
        Superfamily: level=2
        Fold: level=1
        """
        query, neighbors = self.get_result(structure)
        upFP = []
        for neig in neighbors:
            neig = self.get_scop(neig)
            if neig[1] != query[1] or neig[0] != query[0]:
                break
            upFP.append(neig)
        # print(upFP, [self.get_scop(e) for e in neighbors])

        upFP = np.asarray(upFP)
        if len(upFP) > 0:
            sel = self.selection(upFP, level, query)
            upFP = upFP[sel]
            num = len(upFP)
            # allhits = [e for e in neighbors if selection(e, level, query)]
            # neighbors = np.asarray(neighbors)
            key = (tuple(query[:level + 2]), level)
            if key not in self.den_dict:
                sel = self.selection(self.scop40, level, query)
                den = sel.sum()
                if level == 3:
                    den -= 1
                self.den_dict[key] = den
            else:
                den = self.den_dict[key]
            if den == 0:
                return -1
        else:
            return -1
        return num / den

    def all_sensitivity(self, outfilename, level=3, plotfile=None, ax=None):
        all_s = []
        for structure in tqdm(self.results_dict):
            s = self.get_sensitivity(structure, level=level)
            if s != -1:
                all_s.append(s)
        np.savetxt(outfilename, all_s)
        if plotfile is not None:
            ax.plot(np.linspace(0, 1, len(all_s)), sorted(all_s, reverse=True))
            ax.set_xlabel('Fraction of queries')
            ax.set_ylabel('Sensitivity up to 1st FP')
            if level == 3:
                plt.title('Family')
            if level == 2:
                plt.title('Super Family')
            if level == 1:
                plt.title('Fold')
            plt.savefig(plotfile)
        return all_s


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
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    parser.add_argument('--foldseek', help='Foldseek SCOP bench', action='store_true')
    parser.add_argument('--foldmine', help='Foldmine SCOP bench', action='store_true')
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

    os.makedirs('out', exist_ok=True)
    if args.foldseek:
        results = 'data/foldseekaln.gz'
        results_pickle = 'data/foldseekresults.pickle'
        outbasename = 'out/foldseek'
    if args.foldmine:
        results = None
        results_pickle = 'data/pickles/scope_dict_result.p'
        # results_pickle = 'data/scope_dict_result.p'
        outbasename = 'out/foldmine'
    bench = Bench(results=results, results_pickle=results_pickle)
    for level in [1, 2, 3]:
        print('level', level)
        outfilename = f'{outbasename}_{level}'
        fig, ax = plt.subplots()
        all_s = bench.all_sensitivity(outfilename=outfilename + '.txt',
                                      level=level,
                                      plotfile=outfilename + '.svg',
                                      ax=ax)
        # print(bench.den_dict)
