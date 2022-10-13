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

SCOPFILENAME = 'data/dir.des.scope.2.01-stable.txt'
FOLDSEEKRESULTS = 'data/alignResults/rawoutput/foldseekaln'
FOLDSEEKRESULTS_PICKLE = 'data/foldseekresults.pickle'


def parse_scop(scopfilename=SCOPFILENAME):
    """
    >>> scop_dict = parse_scop()
    >>> scop_dict
    {'d1ux8a_': 'a.1.1.1', 'd1dlwa_': 'a.1.1.1', 'd1uvya_': 'a.1.1.1', ...

    Can retrieve the scop class of a system:
    >>> scop_dict['d1a1xa_']
    'b.63.1.1'
    """
    data = np.genfromtxt(scopfilename, usecols=(2, 3), dtype=str)
    sel = (data[:, 1] != '-')
    data = data[sel]
    lookupdict = dict(zip(data[:, 1], data[:, 0]))
    return lookupdict


SCOP_DICT = parse_scop()


def parse_foldseek_results(foldseek_results=FOLDSEEKRESULTS, max_rows=None, outfilename=FOLDSEEKRESULTS_PICKLE):
    """
    >>> foldseek_res = parse_foldseek_results(max_rows=100000, outfilename='test.pickle')
    >>> foldseek_res
    {'d1a1xa_': array(['d1a1xa_', 'd1jsga_', 'd3saoa_', ...

    foldseek_res retrieve sorted neighbors for a given structure
    >>> foldseek_res['d1a1xa_']
    array(['d1a1xa_', 'd1jsga_', 'd3saoa_', ..., 'd1mkna_', 'd1vkza3',
           'd1vd4a_'], dtype='<U7')
    """
    data = np.genfromtxt(foldseek_results, usecols=(0, 1), dtype=str, max_rows=max_rows)
    p1 = data[:, 0]
    p2 = data[:, 1]
    inds = np.where(p1[:-1] != p1[1:])[0] + 1
    p1 = np.split(p1, inds)
    p2 = np.split(p2, inds)
    p1 = [e[0] for e in p1]
    out = dict(zip(p1, p2))
    pickle.dump(out, open(outfilename, 'wb'))
    return out


if os.path.exists(FOLDSEEKRESULTS_PICKLE):
    FOLDSEEKRESULTS_DICT = pickle.load(open(FOLDSEEKRESULTS_PICKLE, 'rb'))
else:
    FOLDSEEKRESULTS_DICT = parse_foldseek_results()


def get_scop(structure, scop_dict=SCOP_DICT):
    """
    Returns: [Class, Fold, Superfamily, Family]
    >>> get_scop('d1a1xa_')
    ['b', '63', '1', '1']
    """
    scop = scop_dict[structure]
    scop = scop.split('.')
    return scop


def get_foldseek_result(structure, foldseekresults_dict=FOLDSEEKRESULTS_DICT):
    """
    >>> query, neighbors = get_foldseek_result('d1a1xa_')
    >>> query
    ['b', '63', '1', '1']
    >>> neighbors
    [['b', '63', '1', '1'], ['b', '60', '1', '1'], ['b', '45', '1', '0'], ...
    """
    neighbors = list(foldseekresults_dict[structure])
    neighbors.remove(structure)
    query = get_scop(structure)
    neighbors = [get_scop(e) for e in neighbors]
    return query, neighbors


def get_sensitivity(structure, level=3):
    """
    Family: level=3
    Superfamily: level=2
    Fold: level=1
    >>> get_sensitivity('d1mkya2')
    0.5428571428571428
    >>> get_sensitivity('d1mkya2', level=2)
    0.0036153289949385392
    >>> get_sensitivity('d1mkya2', level=1)
    -1
    """
    query, neighbors = get_foldseek_result(structure)
    upFP = []
    for neig in neighbors:
        if neig[1] != query[1] or neig[0] != query[0]:
            break
        upFP.append(neig)

    def filter(scop, level, target):
        scoptest = scop[level] == target[level]
        if level == 3:
            return scoptest
        else:
            scoptest = (scoptest and scop[level + 1] != target[level + 1])
            return scoptest

    upFP = [e for e in upFP if filter(e, level, query)]
    num = len(upFP)
    allhits = [e for e in neighbors if filter(e, level, query)]
    den = len(allhits)
    if den == 0:
        return -1
    return num / den


def all_sensitivity(foldseekresults_dict=FOLDSEEKRESULTS_DICT, level=3):
    all_s = []
    for structure in tqdm(foldseekresults_dict):
        s = get_sensitivity(structure, level=level)
        if s != -1:
            all_s.append(s)
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

    all_s = all_sensitivity()
    np.savetxt('sens.txt', all_s)
