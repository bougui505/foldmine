# d1a04a1 ['a', '4', '6', '2']
# d1zlja_ ['a', '4', '6', '0']
# d2o38a1 ['a', '35', '1', '13']
#
#
# d1a04a2 ['c', '23', '1', '1']
# d3arcb_ ['f', '55', '1', '1']
# d1jbea_ ['c', '23', '1', '1']
#
#
# d12asa_ ['d', '104', '1', '1']
# d1twfb_ ['e', '29', '1', '1']

from scipy.optimize import linear_sum_assignment
import itertools
import numpy as np

scop_map = {'d1a04a1': ['a', '4', '6', '2'],
            'd1zlja_': ['a', '4', '6', '0'],
            'd2o38a1': ['a', '35', '1', '13'],
            'd1a04a2': ['c', '23', '1', '1'],
            'd3arcb_': ['f', '55', '1', '1'],
            'd12asa_': ['d', '104', '1', '1'],
            'd1twfb_': ['e', '29', '1', '1'],
            'd1jbea_': ['c', '23', '1', '1']
            }
mapping = {name: f"{name[1:3]}/{name}" for name in scop_map.keys()}

import h5py
import scipy.spatial.distance as scidist
import numpy as np

infilename = 'data/hdf5/embeddings_scope_32_ap_pca.hdf5'


# infilename = 'data/hdf5/embeddings_scope_32.hdf5'


def query(name):
    with h5py.File(infilename, 'r') as f:
        path = f"{name[1:3]}/{name}"
        embs_to_get = 'res_embs'
        v = f[path][embs_to_get][()]
        # return None, v
        resids = f[path]['res_ids'][()]
        return resids, v


def print_res(name1, name2):
    res1, vec1 = query(name1)
    res2, vec2 = query(name2)
    distances = scidist.cdist(vec1, vec2)
    argmin = np.argmin(distances)
    argmin_idx = np.unravel_index(argmin, distances.shape)
    dist = distances[argmin_idx]

    # if name2 == 'd1jbea_':
    #     a = list(res1).index(57)
    #     b = list(res2).index(55)
    #     print(distances[a, b])

    matching_res1 = res1[argmin_idx[0]] if res1 is not None else None
    matching_res2 = res2[argmin_idx[1]] if res2 is not None else None
    print(
        f"{name1}, {name1[2:4]}/{name1}.pdb , {scop_map[name1]}, {name2}, {name2[2:4]}/{name2}.pdb , {scop_map[name2]} "
        f": {dist:.3f}, matching res = {matching_res1}, {matching_res2}")


# Failed d1a04a1 ['a', '4', '6', '2']
print_res('d1a04a1', 'd1zlja_')  # d1zlja_ ['a', '4', '6', '0'],
print_res('d1a04a1', 'd2o38a1')  # d2o38a1 ['a', '35', '1', '13']
print()

# Failed d1a04a2 ['c', '23', '1', '1']
print_res('d1a04a2', 'd3arcb_')  # d3arcb_ ['f', '55', '1', '1']
print_res('d1a04a2', 'd1jbea_')  # d1jbea_ ['c', '23', '1', '1'] # Should be matched : d1a04a2 : 57 and d1jbea_ 55


def align(name1, name2, cutoff=None, pymol_cmd=False):
    res1, vec1 = query(name1)
    res2, vec2 = query(name2)
    distances = scidist.cdist(vec1, vec2)

    # cost = 1/(1 + np.exp(-distances/20))
    row_ind, col_ind = linear_sum_assignment(distances)

    # row_ind, col_ind = np.arange(len(vec1)), np.argmin(distances, axis=1)

    scores = distances[row_ind, col_ind]
    sorter = np.argsort(scores)
    sorted_scores = scores[sorter]

    match1, match2 = res1[row_ind], res2[col_ind]
    sorted_m1, sorted_m2 = match1[sorter], match2[sorter]

    # print(np.mean(scores))
    # print(np.sum(scores))
    # print(scores.shape)
    # print()
    sorted_m1 = sorted_m1[:cutoff]
    sorted_m2 = sorted_m2[:cutoff]
    sorted_scores = sorted_scores[:cutoff]
    print(np.mean(sorted_scores))

    if pymol_cmd:
        pymol_align(sorted_m1, sorted_m2, name1, name2)


def matrix(a, b, score_mat, gap_cost=2.):
    """
    # >>> a = 'ggttgacta'
    # >>> len(a)
    # 9
    # >>> b = 'tgttacgg'
    # >>> len(b)
    # 8
    # >>> H = matrix(a, b)
    # >>> H.shape
    # (10, 9)
    """
    H = np.zeros((len(a) + 1, len(b) + 1))
    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (score_mat[i - 1, j - 1])
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


def traceback(H, b, b_=[], old_i=0, score=0):
    """
    >>> a = 'ggttgacta'
    >>> b = []
    >>> H = matrix(a, b)
    >>> traceback(H, b)
    ('gtt-ac', 1, 41.0)

    # b can be None:
    # >>> traceback(H, b=None)
    # (None, 1, 41.0)
    """
    # flip H to get index of **last** occurrence of H.max() with np.argmax()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()
    score += H[i, j]
    if H[i, j] == 0:
        return b_, j, score
    if b is not None:
        if old_i - i > 1:
            b_ = [b[j - 1], None] + b_
        else:
            b_ = [b[j - 1]] + b_
    else:
        b_ = None
    return traceback(H=H[0:i, 0:j], b=b, b_=b_, old_i=i, score=score)


def smith_waterman(a, b, match_score, gap_cost=2):
    """
    """
    H = matrix(a, b, match_score, gap_cost)
    b_, pos, score = traceback(H, b)
    return pos, pos + len(b_), score


def sw_align(name1, name2, pymol_cmd=False):
    """
    Return two matching lists of residues
    @param name1:
    @param name2:
    @return:
    """
    res1, vec1 = query(name1)
    res2, vec2 = query(name2)
    distances = scidist.cdist(vec1, vec2)

    H = matrix(res1, res2, score_mat=distances, gap_cost=0.01)
    b_, pos, score = traceback(H, res2)
    cropped_res_1 = res1[pos:pos + len(b_)]

    filtered_res = [(r1, r2) for r1, r2 in zip(cropped_res_1, b_) if r2 is not None]
    res1, res2 = list(map(list, zip(*filtered_res)))
    if pymol_cmd:
        pymol_align(res1, res2, name1, name2)
    return res1, res2


def pymol_align(reslist1, reslist2, name1, name2, pymol_cmd='temp.pml'):
    with open('temp.pml', 'w') as f:
        atom_list_1 = []
        atom_list_2 = []
        lines = []
        for r1, r2 in zip(reslist1, reslist2, ):
            if r2 is not None:
                sel1 = f"(resi {r1} and {name1} and name CA)"
                sel2 = f"(resi {r2} and {name2} and name CA)"
                atom_list_1.append(sel1)
                atom_list_2.append(sel2)
                s = f"distance {sel1}, {sel2}\n"
                lines.append(s)
        atom_list1 = '+'.join(atom_list_1)
        atom_list2 = '+'.join(atom_list_2)

        f.write(f"remove not alt ''+A \n")
        f.write(f"alter all, alt='' \n")
        f.write(f"delete dist* \n")
        f.write(f"pair_fit {atom_list1}, {atom_list2}\n")
        f.writelines(lines)
        f.write(f"orient {name1} or {name2} \n")


if __name__ == '__main__':
    # align('d1a04a1', 'd1zlja_')  # d1zlja_ ['a', '4', '6', '0'],
    # align('d1a04a1', 'd2o38a1')
    # print()
    #
    align('d1a04a2', 'd3arcb_', cutoff=4, pymol_cmd=True)
    # align('d1a04a2', 'd1jbea_', cutoff=20, pymol_cmd=True)
    # res1, res2 = sw_align('d1a04a2', 'd1jbea_', pymol_cmd=True)
    # res1, res2 = sw_align('d1a04a2', 'd3arcb_', pymol_cmd=True)
