#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distances classiques entre embeddings (8 features).

Cosine, euclidean, pearson, etc.
Refactoré depuis compute_features.py de Nour.
"""

from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist


# =============================================================================
# Distances individuelles
# =============================================================================

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance euclidienne."""
    return float(np.linalg.norm(v1 - v2))


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance cosinus (1 - similarité)."""
    return float(cdist([v1], [v2], metric="cosine")[0][0])


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similarité cosinus."""
    return 1.0 - cosine_distance(v1, v2)


def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Corrélation de Pearson."""
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    corr, _ = pearsonr(v1, v2)
    return float(corr) if not np.isnan(corr) else 0.0


def spearman_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Corrélation de Spearman."""
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    corr, _ = spearmanr(v1, v2)
    return float(corr) if not np.isnan(corr) else 0.0


def minkowski_distance(v1: np.ndarray, v2: np.ndarray, p: int = 3) -> float:
    """Distance de Minkowski (p=3 par défaut)."""
    return float(cdist([v1], [v2], metric="minkowski", p=p)[0][0])


def canberra_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance de Canberra."""
    return float(cdist([v1], [v2], metric="canberra")[0][0])


def chebyshev_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance de Chebyshev (max des différences absolues)."""
    return float(cdist([v1], [v2], metric="chebyshev")[0][0])


# =============================================================================
# Liste des features
# =============================================================================

CLASSICAL_FEATURES = [
    "cls_euclidean",
    "cls_cosine_dist",
    "cls_cosine_sim",
    "cls_pearson",
    "cls_spearman",
    "cls_minkowski",
    "cls_canberra",
    "cls_chebyshev",
]


def compute_classical_features(v1: np.ndarray, v2: np.ndarray) -> Dict[str, float]:
    """
    Calcule toutes les distances/similarités classiques entre deux vecteurs.

    Args:
        v1: Premier vecteur (embedding)
        v2: Second vecteur (embedding)

    Returns:
        Dictionnaire {feature_name: value}
    """
    return {
        "cls_euclidean": euclidean_distance(v1, v2),
        "cls_cosine_dist": cosine_distance(v1, v2),
        "cls_cosine_sim": cosine_similarity(v1, v2),
        "cls_pearson": pearson_correlation(v1, v2),
        "cls_spearman": spearman_correlation(v1, v2),
        "cls_minkowski": minkowski_distance(v1, v2),
        "cls_canberra": canberra_distance(v1, v2),
        "cls_chebyshev": chebyshev_distance(v1, v2),
    }


def compute_classical_features_batch(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    show_progress: bool = True
) -> List[Dict[str, float]]:
    """
    Calcule les features classiques pour un batch de paires.

    Args:
        embeddings1: Array (n_pairs, embedding_dim)
        embeddings2: Array (n_pairs, embedding_dim)
        show_progress: Afficher la progression

    Returns:
        Liste de dictionnaires de features
    """
    from tqdm import tqdm

    n = len(embeddings1)
    iterator = tqdm(range(n), desc="Classical features") if show_progress else range(n)

    results = []
    for i in iterator:
        results.append(compute_classical_features(embeddings1[i], embeddings2[i]))

    return results


def compute_classical_features_vectorized(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Version vectorisée (plus rapide) pour de grands batches.

    Args:
        embeddings1: Array (n_pairs, embedding_dim)
        embeddings2: Array (n_pairs, embedding_dim)

    Returns:
        Dictionnaire {feature_name: array de valeurs}
    """
    n = len(embeddings1)

    # Euclidean
    diff = embeddings1 - embeddings2
    euclidean = np.linalg.norm(diff, axis=1)

    # Cosine
    dot = np.sum(embeddings1 * embeddings2, axis=1)
    norm1 = np.linalg.norm(embeddings1, axis=1)
    norm2 = np.linalg.norm(embeddings2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-10)
    cosine_dist = 1.0 - cosine_sim

    # Chebyshev
    chebyshev = np.max(np.abs(diff), axis=1)

    # Les autres nécessitent une boucle (corrélations)
    pearson = np.zeros(n)
    spearman = np.zeros(n)
    minkowski = np.zeros(n)
    canberra = np.zeros(n)

    for i in range(n):
        pearson[i] = pearson_correlation(embeddings1[i], embeddings2[i])
        spearman[i] = spearman_correlation(embeddings1[i], embeddings2[i])
        minkowski[i] = minkowski_distance(embeddings1[i], embeddings2[i])
        canberra[i] = canberra_distance(embeddings1[i], embeddings2[i])

    return {
        "cls_euclidean": euclidean,
        "cls_cosine_dist": cosine_dist,
        "cls_cosine_sim": cosine_sim,
        "cls_pearson": pearson,
        "cls_spearman": spearman,
        "cls_minkowski": minkowski,
        "cls_canberra": canberra,
        "cls_chebyshev": chebyshev,
    }


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    # Test avec des vecteurs aléatoires
    np.random.seed(42)

    v1 = np.random.randn(384)
    v2 = np.random.randn(384)
    v3 = v1 + np.random.randn(384) * 0.1  # Proche de v1

    print("v1 vs v2 (aléatoires):")
    for k, v in compute_classical_features(v1, v2).items():
        print(f"  {k}: {v:.4f}")

    print("\nv1 vs v3 (proches):")
    for k, v in compute_classical_features(v1, v3).items():
        print(f"  {k}: {v:.4f}")

    print("\nv1 vs v1 (identiques):")
    for k, v in compute_classical_features(v1, v1).items():
        print(f"  {k}: {v:.4f}")
