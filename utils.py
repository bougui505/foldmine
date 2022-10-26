import os
import sys

import datetime
import h5py
import logging
import time
import torch
from tqdm import tqdm


class ETA(object):
    """
    >>> total_steps = 5
    >>> eta = ETA(total_steps)
    >>> for i in range(total_steps):
    ...     time.sleep(1)
    ...     print(eta())
    0:00:04.00...
    0:00:03.00...
    0:00:02.00...
    0:00:01.00...
    0:00:00
    """

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.step = 0
        self.start = time.perf_counter()

    def __call__(self, step=None, return_str=True):
        self.step += 1
        if step is None:
            step = self.step
        runtime = time.perf_counter() - self.start
        eta = self.total_steps * runtime / step - runtime  # in seconds
        if return_str:
            return str(datetime.timedelta(seconds=eta))
        else:
            return eta


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()


def pdbfile_to_chain(pdb_file):
    # # toto/tata/1ycr_A.pdb => 1ycr_A
    # pdb_name = os.path.basename(pdb_file).split('.')[0]

    # Careful with dots in name !
    # toto/tata/1ycr_A.pdb.gz => 1ycr_A.pd
    pdb_name = os.path.basename(pdb_file)[:-4]
    return pdb_name


def pdbchain_to_hdf5path(pdb_chain):
    # 1ycr_A => yc/1ycr_A
    pdb_dir = f"{pdb_chain[1:3]}/{pdb_chain}"
    return pdb_dir


def pdbfile_to_hdf5path(pdb_file):
    # toto/tata/1ycr_A.pdb.gz => yc/1ycr_A
    return pdbchain_to_hdf5path(pdbfile_to_chain(pdb_file))


def load_model(filename, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    model.eval()
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def len_hdf5(hdf5f):
    """
    Returns the len of a hdf5
    """
    n = 0
    for key in hdf5f.keys():
        n += len(hdf5f[key].keys())
    return n


# Read the embeddings to retrieve all the graph level embeddings
def read_embeddings(infilename='data/hdf5/embeddings_scope.hdf5',
                    return_level='residue',
                    return_res_ids=False,
                    early_stop=None):
    """
    Read an embeddings hdf5 file and return the list of systems along with their embeddings
    at the res or the graph level

    @param infilename:
    @param return_level:
    @param return_res_ids:
    @param early_stop:
    @return:
    """
    with h5py.File(infilename, 'r') as f:
        n = len_hdf5(f)
        pbar = tqdm(total=n)
        all_systems = []
        all_embeddings = []
        all_res_ids = []
        i = 0
        # all_embeddings.append(f['1d']['d1dlwa_']['res_embs'][()])
        for key in f.keys():
            for system in f[key].keys():  # iterate pdb systems
                embs_to_get = 'res_embs' if return_level == 'residue' else 'graph_embs'
                v = f[key][system][embs_to_get][()]
                all_systems.append(system)
                all_embeddings.append(v)
                if return_res_ids:
                    all_res_ids.append(f[key][system]['res_ids'][()])
                pbar.update(1)
                i += 1
            if early_stop is not None and i > early_stop:
                break
        pbar.close()
    if return_res_ids:
        return all_systems, all_embeddings, all_res_ids
    return all_systems, all_embeddings


if __name__ == '__main__':
    import sys
    import doctest
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
