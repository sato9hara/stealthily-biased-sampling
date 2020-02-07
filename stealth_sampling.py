import os
import numpy as np
import subprocess
from sklearn.datasets import dump_svmlight_file

def stealth_sampling(X, K, path='./', prefix='tmp', timeout=10.0):
    assert len(X)==len(K)
    C = len(X)
    while True:
        Y = np.concatenate(X, axis=0)
        Y += 1e-10 * np.random.randn(*Y.shape)
        z = np.concatenate([i*np.ones(X[i].shape[0]) for i in range(C)])
        com = ' '.join('%d' % (k,) for k in K)
        dump_svmlight_file(Y, z, './%s_input.txt' % (prefix,), comment=com)
        cmd = '%sstealth-sampling/main ./%s_input.txt > %s_output.txt' % (path, prefix, prefix)
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
    res = np.loadtxt('./%s_output.txt' % (prefix,))
    p = np.split(res, np.cumsum([X[i].shape[0] for i in range(C)]))
    os.remove('./%s_input.txt' % (prefix,))
    os.remove('./%s_output.txt' % (prefix,))
    return np.array(p[:-1]) / (Y.shape[0] * sum(K))

def stealth_sampling_bootstrap(X, K, path='./', prefix='tmp', ratio=0.3, num_sample=10, num_process=2, seed=0, timeout=10.0):
    c = 0
    c2 = 0
    C = len(X)
    q = [1e-10*np.ones(X[j].shape[0]) for j in range(C)]
    for i in range(num_sample):
        while True:
            c2 += 1
            commands = []
            N = []
            M = []
            idx = []
            for p in range(num_process):
                prefix_p = '%s_p%05d' % (prefix, p)
                np.random.seed(seed+c)
                idx_p = []
                Ni = []
                Xi = []
                Ki = []
                for j in range(C):
                    nj = X[j].shape[0]
                    mj = int(np.round(ratio * nj))
                    idxj = np.random.permutation(nj)[:mj]
                    idx_p.append(idxj)
                    Ni.append(mj)
                    Xi.append(X[j][idxj, :])
                    Ki.append(int(np.round(ratio * K[j])))
                N.append(Ni)
                M.append(Ki)
                idx.append(idx_p)
                np.random.seed(seed+c2)
                Yi = np.concatenate(Xi, axis=0)
                Yi += 1e-10 * np.random.randn(*Yi.shape)
                zi = np.concatenate([j*np.ones(Xi[j].shape[0]) for j in range(C)])
                com = ' '.join('%d' % (k,) for k in Ki)
                dump_svmlight_file(Yi, zi, './%s_input.txt' % (prefix_p,), comment=com)
                cmd = '%sstealth-sampling/main ./%s_input.txt > %s_output.txt' % (path, prefix_p, prefix_p)
                commands.append(cmd)
            procs = [subprocess.Popen('exec ' + cmd, shell=True) for cmd in commands]
            try:
                for p in procs:
                    p.wait(timeout)
                c += 1
                break
            except:
                for p in procs:
                    p.kill()
                print('killed')
        for p in range(num_process):
            prefix_p = '%s_p%05d' % (prefix, p)
            res = np.loadtxt('./%s_output.txt' % (prefix_p,))
            res = np.split(res, np.cumsum([N[p][i] for i in range(C)]))[:-1]
            for j in range(C):
                q[j][idx[p][j]] += res[j] / (sum(N[p]) * sum(M[p]) * num_sample * num_process)
            os.remove('./%s_input.txt' % (prefix_p,))
            os.remove('./%s_output.txt' % (prefix_p,))
    qsum = np.sum(np.concatenate(q))
    q = [qq/qsum for qq in q]
    return q

def compute_wasserstein(X1, X2, path='./', prefix='tmp', timeout=10.0):
    assert X1.shape[1] == X2.shape[1]
    while True:
        dump_svmlight_file(X1+1e-10*np.random.randn(*X1.shape), np.zeros(X1.shape[0]), './%s_input1.txt' % (prefix,))
        dump_svmlight_file(X2+1e-10*np.random.randn(*X2.shape), np.zeros(X2.shape[0]), './%s_input2.txt' % (prefix,))
        cmd = '%swasserstein/main ./%s_input1.txt ./%s_input2.txt > %s_output.txt' % (path, prefix, prefix, prefix)
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
    d = np.loadtxt('./%s_output.txt' % (prefix,))
    os.remove('./%s_input1.txt' % (prefix,))
    os.remove('./%s_input2.txt' % (prefix,))
    os.remove('./%s_output.txt' % (prefix,))
    return d

def compute_wasserstein_bootstrap(X1, X2, n, path='./', prefix='tmp', num_sample=10, num_process=2, seed=0, timeout=10.0):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n = min([n, n1, n2])
    c = 0
    c2 = 0
    d = 0.0
    for i in range(num_sample):
        while True:
            c2 += 1
            commands = []
            for p in range(num_process):
                prefix_p = '%s_%05d' % (prefix, p)
                np.random.seed(seed+c)
                idx1 = np.random.permutation(n1)[:n]
                idx2 = np.random.permutation(n2)[:n]
                np.random.seed(seed+c2)
                dump_svmlight_file(X1[idx1, :]+1e-10*np.random.randn(idx1.size, X1.shape[1]), np.zeros(idx1.size), './%s_input1.txt' % (prefix_p,))
                dump_svmlight_file(X2[idx2, :]+1e-10*np.random.randn(idx2.size, X2.shape[1]), np.zeros(idx2.size), './%s_input2.txt' % (prefix_p,))
                cmd = '%swasserstein/main ./%s_input1.txt ./%s_input2.txt > %s_output.txt' % (path, prefix_p, prefix_p, prefix_p)
                commands.append(cmd)
            procs = [subprocess.Popen('exec ' + cmd, shell=True) for cmd in commands]
            try:
                for p in procs:
                    p.wait(timeout)
                c += 1
                break
            except:
                for p in procs:
                    p.kill()
                print('killed')
        for p in range(num_process):
            prefix_p = '%s_%05d' % (prefix, p)
            dp = np.loadtxt('./%s_output.txt' % (prefix_p,))
            d += dp / (num_sample * num_process)
            os.remove('./%s_input1.txt' % (prefix_p,))
            os.remove('./%s_input2.txt' % (prefix_p,))
            os.remove('./%s_output.txt' % (prefix_p,))
    return d