#!/usr/bin/env python3

import numpy as np
from tqdm.auto import tqdm
from argparse import ArgumentParser


def bootstrap_hypothesis_testing(y, z, B=1000, mode='mean', warn=True, mem_lim=1e6, n=None, m=None):
    """
    Two-sample hypothesis testing through bootstrapping

    CALL ASL, t = bootstrap_hypothesis_testing(y, z, B, mode)

    INPUT
        y: input values from sample 1
        z: input values from sample 2
        B: number of bootstrapping iterations
            default: 1000. It can be tuned to keep the coefficient of variation sigma/mu < 0.1
        mode: 'F=G' or 'mean' for two different algorithms (def:'mean')
            mean: are the means of y and z the same?
            F=G: are F and G the same?
        warn: True/False whether to suggest increasing B
        mem_lim: maximum number of elements in an array at any time,
            decrease to reduce memory use at the cost of performance

    OUTPUT
        ASL: achieved significance level (aka alpha/p-value) in steps of 2/B,
        t: bootstrapping t parameters

    Reference: An introduction to the bootstrap - Efron & Tibshirani
    Chapter 16 for the algorithms

    SC @LenstraLab 13/10/2020 > Matlab
    WP                        > Python
    """

    np.seterr(divide='ignore')

    n = n or int(np.round(len(z) ** (2. / 3)))
    m = m or int(np.round(len(y) ** (2. / 3)))
    B = int(B)
    mem_lim = int(mem_lim / (n + m))  # number of bootstraps at once
    if mem_lim == 0:
        mem_lim = 1
        if warn:
            print(f'mem_lim < {B * (n + m)}, ignoring...')
    rng = np.random.default_rng()

    # pool together y and z into x
    x = np.hstack((z.flatten(), y.flatten()))
    bs = (B // mem_lim) * (mem_lim,) + (B % mem_lim,)
    t = []
    with tqdm(total=B, disable=len(bs) < 2, desc='Bootstrapping', leave=False) as bar:
        if mode == 'F=G':  # Algorithm 16.1 page 221
            t0 = np.mean(z) - np.mean(y)
            for b in bs:
                z_ = rng.choice(x, (n, b))
                y_ = rng.choice(x, (m, b))
                t.append(np.mean(z_, 0) - np.mean(y_, 0))
                bar.update(b)
        elif mode == 'mean':  # Algorithm 16.2 page 224
            t0 = (np.mean(z) - np.mean(y)) / np.sqrt(np.var(z, ddof=1) / len(z) + np.var(y, ddof=1) / len(y))
            z0 = z - np.mean(z) + np.mean(x)
            y0 = y - np.mean(y) + np.mean(x)
            for b in bs:
                z_ = rng.choice(z0, (n, b))
                y_ = rng.choice(y0, (m, b))
                t.append((np.mean(z_, 0) - np.mean(y_, 0)) /
                         np.sqrt(np.var(z_, 0, ddof=1) / n + np.var(y_, 0, ddof=1) / m))
                bar.update(b)
        else:
            raise(ValueError('Unrecognized mode, options: mean or F=G'))

    t = np.hstack(t)
    ASL = 2 * min(np.sum(t <= t0), np.sum(t > t0)).astype(float) / B

    if ASL == 0 and warn:
        print('ASL < 2/B, suggest increasing B')
    return ASL, t


def main():
    parser = ArgumentParser(description=bootstrap_hypothesis_testing.__doc__)
    parser.add_argument('y_file', help='text file with distribution(s) in column(s)')
    parser.add_argument('z_file', help='text file with distribution(s) in column(s)')
    parser.add_argument('-B', '--bootstraps', help='number of bootstraps', type=int, default=1000)
    parser.add_argument('-m', '--mode', help='mode: mean or F=G', default='mean')
    args = parser.parse_args()

    y = np.loadtxt(args.y_file, ndmin=2, dtype=float)
    z = np.loadtxt(args.z_file, ndmin=2, dtype=float)

    ASL, t = zip(*[bootstrap_hypothesis_testing(y[:, i], z[:, i], args.bootstraps, args.mode)
                   for i in range(y.shape[1])])

    print('ASL: ' + ' '.join(['{:.5e}'] * len(ASL)).format(*ASL))


if __name__ == '__main__':
    main()
