"""Sparse-friendly distance metrics for recipe ingredient vectors."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _to_csr(matrix: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


class JaccardDistance:
    """Pairwise Jaccard distance for binary/sparse matrices (DistanceMetric Protocol)."""

    def pairwise(
        self,
        A: np.ndarray | sparse.spmatrix,
        B: np.ndarray | sparse.spmatrix,
    ) -> np.ndarray:
        a_bin = (_to_csr(A) > 0).astype(np.float32)
        b_bin = (_to_csr(B) > 0).astype(np.float32)

        intersection = (a_bin @ b_bin.T).toarray()
        a_sum = np.asarray(a_bin.sum(axis=1)).ravel()[:, None]
        b_sum = np.asarray(b_bin.sum(axis=1)).ravel()[None, :]
        union = a_sum + b_sum - intersection

        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, intersection / union, 0.0)
        return np.clip(1.0 - sim, 0.0, 1.0).astype(np.float32)


class CosineDistance:
    """Pairwise Cosine distance for sparse/dense matrices (DistanceMetric Protocol)."""

    def pairwise(
        self,
        A: np.ndarray | sparse.spmatrix,
        B: np.ndarray | sparse.spmatrix,
    ) -> np.ndarray:
        a_csr = _to_csr(A).astype(np.float32)
        b_csr = _to_csr(B).astype(np.float32)

        dot = (a_csr @ b_csr.T).toarray()
        a_norm = np.sqrt(np.asarray(a_csr.multiply(a_csr).sum(axis=1)).ravel())[:, None]
        b_norm = np.sqrt(np.asarray(b_csr.multiply(b_csr).sum(axis=1)).ravel())[None, :]
        denom = a_norm * b_norm

        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, dot / denom, 0.0)
        return np.clip(1.0 - sim, 0.0, 1.0).astype(np.float32)