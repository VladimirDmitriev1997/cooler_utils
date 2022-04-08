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
import pickle

"""
The general pipeline is taken from cooltools eigdecomp.py I've reused parts of the code here.
The only difference is block structure of eigendecomposition

"""


def cys_eig_blockwise(c, 
                      view_df=None,
                      balance=True,
                      block_size=10000,
                      n_eigs=3, 
                      phasing_track=None, 
                      ignore_diags=0, 
                      clip_percentile=99.9, 
                      sort_metric=None, 
                      divide_by_mean=False, 
                      subtract_mean=False,
                      diagonal_decay=None,
                      collect_decay=None,
                      collect_mean_mask=None,
                      bad_bins=None):
    
    """
    Compute eigenvectors for cis, uploading matrix blockwise
    
    c - cooler
    
    balance - bool, if true, uploads balanced blocks
    blocksize - int, size of block to upload in memory
    n_eigs - int, number of eigenvectors to compute
    phasing_track - key for tuning eigvecs correlations
    clip_percentile - key for value clipping
    sort_metric - key for sorting of eigvecs
    divide_by_mean - bool, if true, will divide over mean for region
    subtract_mean - bool, if true, will subtract mean from region
    collect_decay - filename to write diagonal decay in or None
    collect_mean_mask - filename prefix to write mean and mask values in or None
    bad_bins - indecies of bins to exclude manually
    
    """
    
    view_df = make_cooler_view(c) if view_df is None else view_df

    
    if diagonal_decay == None:
        diagonal_decay = expected_cis(clr=c, 
                                      view_df=view_df,
                                      ignore_diags=ignore_diags)
        if collect_decay:
            with open(collect_deacy, 'wb') as f:
                pickle.dump(diagonal_decay, f)
            
            

    
    
    region_decay_pair = [(i, diagonal_decay[diagonal_decay["region1"]==i[0]]["balanced.avg"]) for i in view_df.values]
    
    
    
    compute_eig = lambda reg, loc_expected: _cys_eig_blockwise(c=c, 
            _region=reg, 
            balance=balance,
            S=loc_expected,
            block_size=block_size,
            n_eigs=n_eigs, 
            phasing_track=phasing_track, 
            ignore_diags=ignore_diags, 
            clip_percentile=clip_percentile, 
            sort_metric=sort_metric, 
            divide_by_mean=divide_by_mean, 
            subtract_mean=subtract_mean,
            collect_mean_mask=collect_mean_mask,
            bad_bins=bad_bins)
    
    results = [compute_eig(reg, loc_expected) for reg, loc_expected in region_decay_pair]
    
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


def _cys_eig_blockwise(c, 
            _region, 
            balance,
            S,
            block_size=100,
            n_eigs=3, 
            phasing_track=None, 
            ignore_diags=0, 
            clip_percentile=99.9, 
            sort_metric=None, 
            divide_by_mean=False, 
            subtract_mean=False,
            collect_mean_mask=None,
            bad_bins=None):
    
    
    """
    Compute eigenvectors for single cis region, uploading matrix blockwise
    
    c - cooler
    _region - tuple (chrom, start, end, name)
    S - np.array of N elements with diagonal decay
    balance - bool, if true, uploads balanced blocks
    blocksize - int, size of block to upload in memory
    n_eigs - int, number of eigenvectors to compute
    phasing_track - key for tuning eigvecs correlations
    clip_percentile - key for value clipping
    sort_metric - key for sorting of eigvecs
    divide_by_mean - bool, if true, will divide over mean for region
    subtract_mean - bool, if true, will subtract mean from region
    collect_decay - filename to write diagonal decay in or None
    collect_mean_mask - filename prefix to write mean and mask values in or None
    bad_bins - indecies of bins to exclude manually
    """
    
    _region = _region[:3]
    
    #bad bins 
    
    if bad_bins is not None:
        # filter bad_bins for the _region and turn relative:
        lo, hi = c.extent(_region)
        bad_bins_region = bad_bins[(bad_bins >= lo) & (bad_bins < hi)]
        bad_bins_region -= lo
    else:
        bad_bins_region = None
    
    
    
    S = S.to_numpy()
    S[~np.isfinite(S)]=1
    
    region_boundaries = c.matrix(balance=balance)._fetch(_region)
    
    region_shape = (region_boundaries[1] - region_boundaries[0], region_boundaries[3]-region_boundaries[2])

    
    mask, mean = compute_mask_and_mean(c, region_boundaries, balance, region_shape, block_size)
    
    if collect_mean_mask is not None:
        with open(collect_mean_mask+'_'+str(_region[0]), 'wb') as f:
            pickle.dump((mask, mean), f)

    if region_shape[0] <= ignore_diags + 3 or mask.sum() <= ignore_diags + 3:
        return _region, np.array([np.nan for i in range(n_eigs)]), np.array([np.ones(region_shape[0]) * np.nan for i in range(n_eigs)])
    
    #for reduction
    mask_shape = mask[mask].shape[0]
    L = LinearOperator((mask_shape, mask_shape), lambda x: multiply_A_block_par(x, 
                                                     c, 
                                                     mask, 
                                                     mean, 
                                                     region_boundaries, 
                                                     ignore_diags, 
                                                     clip_percentile,
                                                     S,
                                                     balance,
                                                     block_size,
                                                     divide_by_mean=divide_by_mean, 
                                                     subtract_mean=subtract_mean,
                                                     bad_bins_region=bad_bins_region,
                                                     nproc=4))
    
    eigvals, eigvecs = eigsh(L, k=n_eigs, tol=1e-3)
    
    
    #ordering 
    eigvecs = eigvecs.T
    order = np.argsort(-np.abs(eigvals))
    eigvecs=eigvecs[order]
    eigvals=eigvals[order]
    #completing
    eigvecs_full = np.full((n_eigs, region_shape[0]), np.nan)
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



def compute_mask_and_mean(c, 
                          region_boundaries, 
                          balance, 
                          region_shape, 
                          block_size):
    
    
    """
    precomputing mask and mean for cis-region
    c - cooler
    region_boundaries - array(y_start, y_finish, x_start, x_finish) for square region 
    balance - True to upload balanced matrix
    region_shape - shape of the A[mask,:][:,mask]
    block_size - number of lines to upload 
    
    returns mask and mean value for the region
    """
    
    mask = np.empty(region_shape[0])
    
    N = region_boundaries[1]-region_boundaries[0]
    part = N//block_size
    
    for x in range(0, part+1):
        _block = c.matrix(balance = balance,sparse=False)[region_boundaries[0]+x*block_size:region_boundaries[0]+min((x+1)*block_size, N),region_boundaries[2]:region_boundaries[3]]
        _block[~np.isfinite(_block)] = 0 
        mask[x*block_size:min((x+1)*block_size, N)] = _block.sum(axis=1)
    
    #for i in range(map_coordinates[1]-map_coordinates[0]):
        
    #    _row = c.matrix(balance = balance,sparse=False)[map_coordinates[0]:map_coordinates[1],i]
    #    _row[~np.isfinite(_row)] = 0
    #    mask[i] = _row.sum()
    mean = mask.sum()/((region_boundaries[1]-region_boundaries[0])*(region_boundaries[3]-region_boundaries[2]))
    mask = mask > 0
    
    return mask, mean 


def multiply_A_block(v, 
                     c, 
                     mask, 
                     mean, 
                     region_boundaries, 
                     ignore_diags, 
                     clip_percentile, 
                     S, 
                     balance, 
                     block_size, 
                     divide_by_mean, 
                     subtract_mean, 
                     bad_bins_region):
    
    """
    
    Linear operator implementing blockwise loading of matrix
    v - vector to multiply 
    c - cooler
    mask - np.array, masking zero rows (precomputed)
    mean - float, mean value over region
    region_boundaries - array(y_start, y_finish, x_start, x_finish) for square region
    ignore_diags, int, number from diags on each side to ignore
    clip_percentile - key for value clipping
    S - 1-dim np.array with diagonal decay N elements
    balance - bool, if true, uploads balanced blocks
    block_size - number of lines to upload
    divide_by_mean - bool, if true, will divide over mean for region
    subtract_mean - bool, if true, will subtract mean from region
    bad_bins - indecies of bins to exclude manually
    returns result of multiplication A@v
    
    """
    
    N = region_boundaries[1] - region_boundaries[0]
    v_shape = v.shape
    v = v.reshape(-1)
    
    
    par = False
    part = N//block_size
    if N%block_size == 0:
        part = part
    else:
        part = part+1
    #pool = multiprocessing.Pool(threadNumber)
    _results = [BlockMult(block_size=block_size,
                          x=x,
                          v=v,
                          c=c, 
                          mask=mask, 
                          mean=mean, 
                          region_boundaries=region_boundaries, 
                          ignore_diags=ignore_diags, 
                          clip_percentile=clip_percentile, 
                          S=S,
                          divide_by_mean=divide_by_mean, 
                          subtract_mean=subtract_mean,
                          balance=balance,
                          bad_bins_region=bad_bins_region,
                          par=par) for x in range(0, part)]
    
    

    #combine results for blocks
    
    results = np.concatenate(_results)
    result = results.reshape(v_shape)
    
    return result



def BlockMult(block_size,
              x,
              v,
              c, 
              mask, 
              mean, 
              region_boundaries, 
              ignore_diags, 
              clip_percentile, 
              S,
              divide_by_mean, 
              subtract_mean,
              balance,
              bad_bins_region,
              par):
    
    
    """
    Single block multiplication
    block_size - int, size of the block
    x - int, number of block
    v - np.array, vector to multiply 
    c - cooler
    mask - np.array, masking zero rows (precomputed)
    mean - float, mean value over region
    region_boundaries - array(y_start, y_finish, x_start, x_finish) for square region
    ignore_diags, int, number from diags on each side to ignore
    clip_percentile - key for value clipping
    S - 1-dim np.array with diagonal decay N elements
    balance - bool, if true, uploads balanced blocks
    block_size - number of lines to upload
    divide_by_mean - bool, if true, will divide over mean for region
    subtract_mean - bool, if true, will subtract mean from region
    bad_bins - indecies of bins to exclude manually
    par - bool, True if parallel
    returns result for single block multiplication
    """
    
    
    N = region_boundaries[1]-region_boundaries[0]
    
    shift = x*block_size
    
    
    pid = os.getpid()

    _block = c.matrix(balance = balance,sparse=False)[region_boundaries[0]+shift:region_boundaries[0]+min(shift+block_size, N),region_boundaries[2]:region_boundaries[3]]
    
    if bad_bins_region is not None:
        if len(bad_bins_region) > 0:
            # apply bad bins to _block of symmetric matrix:
        
            _block[:, bad_bins_region] = np.nan
        
            bad_bins_region -=shift
            bad_bins_region=bad_bins_region[(bad_bins_region>=0) & (bad_bins_region<_block.shape[0])]
            _block[bad_bins_region, :] = np.nan
    
    #_row, row_sum = fetch_row(i, c, fetching_params)#
    #fix Nans/ infs with zeros
    _block[~np.isfinite(_block)] = 0
        
        
    #ignoring the diags
    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(_block[:,shift:shift+block_size], 1.0, d)
    
    
    
    _block /= np.stack([np.concatenate((S[i:0:-1],S[:N-i])) for i in range(shift, min(shift+block_size, N))]) 
    
    
    #clipping
    if clip_percentile and clip_percentile < 100 and min(_block[mask[shift:min(shift+block_size, N)],:][:,mask].shape)>0:
        _block = np.clip(_block, 0, np.percentile(_block[mask[shift:min(shift+block_size, N)],:][:,mask], clip_percentile))
        
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
    partial = _block.dot(v)
    #row_sums[i] = row_sum
    del _block
    
    #mask = row_sums > 0
    partial = partial.reshape(-1)
    if par:
        return partial, pid
    else:
        return partial



def multiply_A_block_par(v, 
                     c, 
                     mask, 
                     mean, 
                     region_boundaries, 
                     ignore_diags, 
                     clip_percentile, 
                     S, 
                     balance, 
                     block_size, 
                     divide_by_mean, 
                     subtract_mean, 
                     bad_bins_region,
                     nproc):
    
    """
    
    Linear operator implementing blockwise loading of matrix
    v - vector to multiply 
    c - cooler
    mask - np.array, masking zero rows (precomputed)
    mean - float, mean value over region
    region_boundaries - array(y_start, y_finish, x_start, x_finish) for square region
    ignore_diags, int, number from diags on each side to ignore
    clip_percentile - key for value clipping
    S - 1-dim np.array with diagonal decay N elements
    balance - bool, if true, uploads balanced blocks
    block_size - number of lines to upload
    divide_by_mean - bool, if true, will divide over mean for region
    subtract_mean - bool, if true, will subtract mean from region
    bad_bins - indecies of bins to exclude manually
    nproc - int, number of processes
    
    
    
    returns result of multiplication A@v
    
    """
    
    N = region_boundaries[1] - region_boundaries[0]
    v_shape = v.shape
    v = v.reshape(-1)
    
    
    
    part = N//block_size
    
    if N%block_size == 0:
        part = part
    else:
        part = part+1
    
    if nproc > 1 and __name__ == '__main__':
        par = True
        pool = multiprocessing.Pool(nproc)
        data = [(block_size,
                 x,
                 v,
                 c, 
                 mask, 
                 mean, 
                 region_boundaries, 
                 ignore_diags, 
                 clip_percentile, 
                 S,
                 divide_by_mean, 
                 subtract_mean,
                 balance,
                 bad_bins_region,
                 par) for x in range(0, part)]
        
        
        _full_multiplication_result = pool.starmap(BlockMult, data)
        
        pool.close()
        pool.join()
        print(_full_multiplication_result)
        
        _full_multiplication_result = sorted(_full_multiplication_result, key=lambda x: x[1]) 
        full_multiplication_result = np.concatenate([x[0] for x in _full_multiplication_result])
    
    else:
        par = False
        _full_multiplication_result = [BlockMult(block_size=block_size,
                                      x=x,
                                      v=v,
                                      c=c, 
                                      mask=mask, 
                                      mean=mean, 
                                      region_boundaries=region_boundaries, 
                                      ignore_diags=ignore_diags, 
                                      clip_percentile=clip_percentile, 
                                      S=S,
                                      divide_by_mean=divide_by_mean, 
                                      subtract_mean=subtract_mean,
                                      balance=balance,
                                      bad_bins_region=bad_bins_region,
                                      par=par) for x in range(0, part)]
        
        full_multiplication_result = np.concatenate(_full_multiplication_result)
    
    
    
    #combine results for blocks
    full_multiplication_result = full_multiplication_result.reshape(v_shape)
    
    return full_multiplication_result

