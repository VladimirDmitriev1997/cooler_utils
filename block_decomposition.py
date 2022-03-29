import os.path as op
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import time
from scipy.sparse.linalg import LinearOperator
import h5py
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from cooltools.lib.common import make_cooler_view, align_track_with_cooler
import cooler
import cooltools
from cooltools.api.eigdecomp import _phase_eigs, eigs_cis
from cooltools.lib.numutils import _logbins_numba
from cooltools.api.expected import expected_cis
import bioframe
from multiprocessing import Pool
import multiprocessing
from cooltools.lib import numutils
from timeit import timeit


"""
The general pipeline is taken from cooltools eigdecomp.py I've reused parts of the code here.
The only difference is block structure of eigendecomposition

"""


def my_cys_eig(c, 
            balance,
            block_size=100,
            n_eigs=3, 
            phasing_track=None, 
            ignore_diags=0, 
            clip_percentile=99.9, 
            sort_metric=None, 
            divide_by_mean = False, 
            subtract_mean = False):
    
    
    
    """
    c - cooler

    balance - True to upload balanced matrix
    block_size - number of lines to upload 
    
    other pararmeters are the same as in cis_eig of cooltools eigdecomp.py
    
    define cooler view and runs over cis regions 
    maybe better to implement more selective way to choose regions
    
    
    """
    
    view_df = make_cooler_view(c)

    
    result = expected_cis(clr=c,
                           ignore_diags=ignore_diags)
    
    region_pairs = [(i, result[result["region1"]==i[0]]["balanced.avg"]) for i in view_df.values]

    
    compute_eig = lambda region_pair: _my_cys_eig(c=c, 
            _region=region_pair[0], 
            balance=balance,
            S=region_pair[1],
            block_size=block_size,
            n_eigs=n_eigs, 
            phasing_track=phasing_track, 
            ignore_diags=ignore_diags, 
            clip_percentile=clip_percentile, 
            sort_metric=sort_metric, 
            divide_by_mean=divide_by_mean, 
            subtract_mean=subtract_mean)
    
    results = [compute_eig(i) for i in  region_pairs]
    
    bins = c.bins()[:]
    clr_weight_name="weight"
    if phasing_track is not None:
        phasing_track = align_track_with_cooler(
            phasing_track,
            c,
            view_df=view_df,
            clr_weight_name=clr_weight_name,
            mask_bad_bins=True,
        )
    
    # prepare output table for eigen vectors
    eigvec_table = bins.copy()
    eigvec_columns = [f"E{i + 1}" for i in range(n_eigs)]
    for ev_col in eigvec_columns:
        eigvec_table[ev_col] = np.nan
    
    eigvals_table = view_df.copy()
    eigval_columns = [f"eigval{i + 1}" for i in range(n_eigs)]
    for eval_col in eigval_columns:
        eigvals_table[eval_col] = np.nan
    
    for i, res in enumerate(results):
        _region = res[0]
        _eigvals = res[1]
        _eigvecs = res[2]
        idx = bioframe.select(eigvec_table, _region).index
        eigvec_table.loc[idx, eigvec_columns] = _eigvecs.T
        idx = bioframe.select(eigvals_table, _region).index
        eigvals_table.loc[idx, eigval_columns] = _eigvals

    return eigvals_table, eigvec_table


def _my_cys_eig(c, 
            _region, 
            balance,
            S,
            block_size=100,
            n_eigs=3, 
            phasing_track=None, 
            ignore_diags=0, 
            clip_percentile=99.9, 
            sort_metric=None, 
            divide_by_mean = False, 
            subtract_mean = False):
    
    
    """
    c - cooler
    _region - tuple samled from cooler view
    balance - True to upload balanced matrix
    S - 1-dim np.array with diagonal decay
    block_size - number of lines to upload 
    
    other pararmeters are the same as in cis_eig of cooltools eigdecomp.py
    
    analogy of _each function
    
    create linear operator which implement blockwise matrix multiplication
    pass this operator to scipy function for sparse eigendecomposition
    
    """
    
    _region = _region[:3]
    
    S = S.to_numpy()
    S[~np.isfinite(S)]=1
    
    map_coordinates = c.matrix(balance=balance)._fetch(_region)
    
    pseudo_shape = (map_coordinates[1] - map_coordinates[0], map_coordinates[3]-map_coordinates[2])

    
    mask, mean = compute_mask_and_mean(c, map_coordinates, balance, pseudo_shape, block_size)
    

    if pseudo_shape[0] <= ignore_diags + 3 or mask.sum() <= ignore_diags + 3:
        return _region, np.array([np.nan for i in range(n_eigs)]), np.array([np.ones(pseudo_shape[0]) * np.nan for i in range(n_eigs)])
    
    #for reduction
    effective_shape = mask[mask].shape[0]
    L = LinearOperator((effective_shape, effective_shape), lambda x: multiply_A_block(x, 
                                                     c, 
                                                     mask, 
                                                     mean, 
                                                     map_coordinates, 
                                                     ignore_diags, 
                                                     clip_percentile,
                                                     S,
                                                     balance,
                                                     block_size,
                                                     divide_by_mean = divide_by_mean, 
                                                     subtract_mean = subtract_mean))
    eigvals, eigvecs = eigsh(L, k=n_eigs)
    
    
    #ordering 
    eigvecs = eigvecs.T
    order = np.argsort(-np.abs(eigvals))
    eigvecs=eigvecs[order]
    eigvals=eigvals[order]
    #completing
    eigvecs_full = np.full((n_eigs, pseudo_shape[0]), np.nan)
    for i in range(n_eigs):
        eigvecs_full[i][mask] = eigvecs[i]
    
    #here the interface from eigdecomp.py
    eigvecs = eigvecs_full
    eigvecs /=np.sqrt(np.nansum(eigvecs ** 2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    # Orient
    if phasing_track is not None:
        eigvals, eigvecs = _phase_eigs(eigvals, eigvecs, phasing_track, sort_metric)

    return _region, eigvals, eigvecs



def compute_mask_and_mean(c, map_coordinates, balance, pseudo_shape, block_size):
    
    
    """
    precomputing mask and mean for cis-region
    c - cooler
    map_coordinates - array(y_start, y_finish, x_start, x_finish) for square region 
    balance - True to upload balanced matrix
    pseudo_shape - shape of the A[mask,:][:,mask]
    block_size - number of lines to upload 
    
    returns mask and mean value for the region
    """
    
    mask = np.empty(pseudo_shape[0])
    
    N = map_coordinates[1]-map_coordinates[0]
    part=N//block_size
    
    for x in range(0, part+1):
        _block = c.matrix(balance = balance,sparse=False)[map_coordinates[0]+x*block_size:map_coordinates[0]+min((x+1)*block_size, N),map_coordinates[2]:map_coordinates[3]]
        _block[~np.isfinite(_block)] = 0 
        mask[x*block_size:min((x+1)*block_size, N)] = _block.sum(axis=1)
    
    #for i in range(map_coordinates[1]-map_coordinates[0]):
        
    #    _row = c.matrix(balance = balance,sparse=False)[map_coordinates[0]:map_coordinates[1],i]
    #    _row[~np.isfinite(_row)] = 0
    #    mask[i] = _row.sum()
    mean = mask.sum()/((map_coordinates[1]-map_coordinates[0])*(map_coordinates[3]-map_coordinates[2]))
    mask = mask > 0
    
    return mask, mean 


def multiply_A_block(v, c, mask, mean, map_coordinates, ignore_diags, clip_percentile, S, balance, block_size, divide_by_mean, subtract_mean):
    
    """
    
    Linear operator implementing blockwise loading of matrix
    v - vector to multiply 
    
    c - cooler
    
    mask - masking zero rows (precomputed)
    
    S - 1-dim np.array with diagonal decay
    block_size - number of lines to upload
    map_coordinates - array(y_start, y_finish, x_start, x_finish) for square region
    
    others are the same as in inital cooltools
    
    
    returns result of multiplication A@v
    """
    N = map_coordinates[1] - map_coordinates[0]
    v_shape = v.shape
    v = v.reshape(-1)
    
    # here nan issue is fixed with matrix.data method 
    # not sure if it is optimal
    # but scipy eigh operate with operator on dense vectors
    #row_sums = np.empty(v.shape)
    #to implement stuff from 121-122 strings in eigdecomp.py
    
    
    part = N//block_size
    #pool = multiprocessing.Pool(threadNumber)
    _results = [BlockMult((block_size,
                                    x,
                                    v,
                                    c, 
                                    mask, 
                                    mean, 
                                    map_coordinates, 
                                    ignore_diags, 
                                    clip_percentile, 
                                    S,
                                    divide_by_mean, 
                                    subtract_mean,
                                    balance)) for x in range(0, part+1)]
    
    

    #combine results for blocks
    results = np.concatenate(_results)
    result = results.reshape(v_shape)
    
    return result



def BlockMult(t):
    
    
    """
    Single block multiplication 
    
    returns result for single block
    """
    
    block_size = t[0]
    x = t[1]
    v = t[2]
    c = t[3]
    mask = t[4] ### with respect to this region
    mean = t[5]
    map_coordinates = t[6] 
    ignore_diags = t[7]
    clip_percentile = t[8] 
    S = t[9]
    divide_by_mean = t[10]
    subtract_mean = t[11]
    balance = t[12]
    
    N = map_coordinates[1]-map_coordinates[0]
    
    shift = x*block_size
    
    
    pid = os.getpid()
    
    _block = c.matrix(balance = balance,sparse=False)[map_coordinates[0]+shift:map_coordinates[0]+min(shift+block_size, N),map_coordinates[2]:map_coordinates[3]]

    
    #_row, row_sum = fetch_row(i, c, fetching_params)#
    #fix Nans/ infs with zeros
    _block[~np.isfinite(_block)] = 0
        
        
    #ignoring the diags
    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(_block[:,shift:shift+block_size], 1.0, d)
        
    _block /= np.stack([np.concatenate((S[i:0:-1],S[:N-i])) for i in range(shift, min(shift+block_size, N))]) 
    
    
    #clipping
    if clip_percentile and clip_percentile < 100:
        _block = np.clip(_block, 0, np.percentile(_block[mask[shift:min(shift+block_size, N)]][:,mask], clip_percentile))
        
    # subtract 1.0
    _block -= 1.0
        
    ##what about diagonal decay?

    _block[:,~mask] = 0
    _block[~mask[shift:min(shift+block_size, N)],:] = 0   
    
    _block = _block[mask[shift:min(shift+block_size, N)],:][:, mask]
    if subtract_mean:
        _block -= mean
    if divide_by_mean:
        _block /= mean
    result = _block.dot(v)
    #row_sums[i] = row_sum
    del _block
    
    #mask = row_sums > 0
    result = result.reshape(-1)
    
    return result