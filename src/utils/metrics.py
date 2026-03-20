from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


def rf(source, generated, split=0.8, seed=42):
    source = np.asarray(source)
    generated = np.asarray(generated)
    n = min(len(source), len(generated))
    np.random.seed(seed)
    source = source[np.random.choice(len(source), n, replace=False)]
    generated = generated[np.random.choice(len(generated), n, replace=False)]
    X = np.vstack((source, generated))
    y = np.hstack((np.zeros(len(source)), np.ones(len(generated))))
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=seed)
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    if acc < 0.5:
        acc = 1 - acc
    return acc


def _median_precision(X, Y) -> float:
    X = np.asarray(X)
    Y = np.asarray(Y)
    tmp = euclidean_distances(X, Y, squared=True)
    tmp = tmp - np.tril(tmp, -1)
    tmp = tmp.reshape(-1, 1)
    positive = tmp[tmp > 0]
    if positive.size == 0:
        return 1.0
    return 1 / np.median(positive)


def mmd(source, generated, prec=None):
    source = np.asarray(source)
    generated = np.asarray(generated)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    if generated.ndim == 1:
        generated = generated.reshape(-1, 1)
    precision = _median_precision(source, generated) if prec is None else prec
    xx = rbf_kernel(source, source, precision).mean()
    yy = rbf_kernel(generated, generated, precision).mean()
    xy = rbf_kernel(source, generated, precision).mean()
    return xx + yy - 2 * xy


def centering(M):
    n = M.shape[0]
    unit = np.ones([n, n])
    identity = np.eye(n)
    H = identity - unit / n
    return np.matmul(M, H)


def gaussian_grammat(x, sigma=None):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(x.shape[0], 1)
    xxT = np.matmul(x, x.T)
    xnorm = np.diag(xxT) - xxT + (np.diag(xxT) - xxT).T
    if sigma is None:
        mdist = np.median(xnorm[xnorm != 0])
        sigma = np.sqrt(mdist * 0.5)
    if sigma == 0:
        sigma += np.finfo(float).eps
    KX = -0.5 * xnorm / sigma / sigma
    np.exp(KX, KX)
    return KX


def dHSIC_calc(K_list):
    if not isinstance(K_list, list):
        K_list = list(K_list)
    n = K_list[0].shape[0]
    term1 = np.prod(K_list, axis=0).sum() / n ** 2
    term2 = 1
    for K in K_list:
        term2 *= K.sum() / n ** 2
    term3 = 2 * np.prod([K.sum(axis=0) for K in K_list], axis=0).sum() / n ** (len(K_list) + 1)
    return term1 + term2 - term3


def HSIC(x, y):
    Kx = centering(gaussian_grammat(x))
    Ky = centering(gaussian_grammat(y))
    return np.trace(np.matmul(Kx, Ky))


def dHSIC(*argv):
    if len(argv) < 2:
        raise ValueError("dHSIC requires at least two variables.")
    K_list = [centering(gaussian_grammat(x)) for x in argv]
    return dHSIC_calc(K_list)
