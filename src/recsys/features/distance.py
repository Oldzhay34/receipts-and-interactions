"""Sparse-friendly distance metrics for recipe ingredient vectors."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from recsys.registry import register


def _to_csr(matrix: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


@register("distance", "jaccard")
class JaccardDistance:
    """Pairwise Jaccard distance for binary/sparse matrices (DistanceMetric Protocol)."""

    def pairwise(
        self,
        A: np.ndarray | sparse.spmatrix,
        B: np.ndarray | sparse.spmatrix,
    ) -> np.ndarray:
        a_bin = (_to_csr(A) > 0).astype(np.float32)
        b_bin = (_to_csr(B) > 0).astype(np.float32)

        # Only pairs that share an ingredient have a distance below 1, and they are few (about
        # 2% of the pairs in the Food.com catalogue): compute those from the sparse
        # intersection and fill the rest with 1. Same values as the dense formula.
        inter = (a_bin @ b_bin.T).tocoo()
        a_sum = np.asarray(a_bin.sum(axis=1), dtype=np.float32).ravel()
        b_sum = np.asarray(b_bin.sum(axis=1), dtype=np.float32).ravel()
        out = np.ones((a_bin.shape[0], b_bin.shape[0]), dtype=np.float32)
        if inter.nnz:
            data = inter.data.astype(np.float32, copy=False)
            union = a_sum[inter.row] + b_sum[inter.col] - data
            out[inter.row, inter.col] = np.clip(1.0 - data / union, 0.0, 1.0)
        return out


@register("distance", "cosine")
class CosineDistance:
    """Pairwise Cosine distance for sparse/dense matrices (DistanceMetric Protocol)."""

    def pairwise(
        self,
        A: np.ndarray | sparse.spmatrix,
        B: np.ndarray | sparse.spmatrix,
    ) -> np.ndarray:
        a_csr = _to_csr(A).astype(np.float32)
        b_csr = _to_csr(B).astype(np.float32)

        # As in Jaccard: only pairs with a non-zero dot product have a distance below 1.
        dot = (a_csr @ b_csr.T).tocoo()
        a_norm = np.sqrt(np.asarray(a_csr.multiply(a_csr).sum(axis=1), dtype=np.float32)).ravel()
        b_norm = np.sqrt(np.asarray(b_csr.multiply(b_csr).sum(axis=1), dtype=np.float32)).ravel()
        out = np.ones((a_csr.shape[0], b_csr.shape[0]), dtype=np.float32)
        if dot.nnz:
            data = dot.data.astype(np.float32, copy=False)
            denom = a_norm[dot.row] * b_norm[dot.col]
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(denom > 0, data / denom, np.float32(0.0))
            out[dot.row, dot.col] = np.clip(1.0 - sim, 0.0, 1.0)
        return out
