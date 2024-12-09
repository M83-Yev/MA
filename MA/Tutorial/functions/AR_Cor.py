import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import find_peaks
# 1. 输入数据




# 2. Levinson-Durbin 估计 AR 模型参数
def levinson_durbin(r, p):
    # r: 自相关向量，p: AR 模型阶数
    phi = np.zeros(p)
    E = r[0]
    for k in range(1, p + 1):
        lambda_k = (r[k] - np.dot(phi[:k - 1], r[1:k][::-1])) / E
        phi[:k] = phi[:k] - lambda_k * phi[:k][::-1]
        phi[k - 1] = lambda_k
        E *= (1 - lambda_k ** 2)
    return phi, E


# 3. 构造 Q^{-1}
def construct_precision_matrix(phi, T):
    Q_inv = np.eye(T)
    for i, coeff in enumerate(phi):
        Q_inv += np.diag([-coeff] * (T - i - 1), k=i + 1)  # 添加带状部分
    return Q_inv


# 4. 预白化
def prewhiten(y, X, p):
    V, T = y.shape
    W = []
    for v in range(V):
        # Step 1: 拟合 GLM
        beta_v = np.linalg.pinv(X.T @ X) @ X.T @ y[v, :]
        residuals = y[v, :] - X @ beta_v

        # Step 2: AR 模型参数估计
        r = np.correlate(residuals, residuals, mode='full')[-T:]  # 自相关
        phi, noise_var = levinson_durbin(r, p)

        # Step 3: 构造白化矩阵
        Q_inv = construct_precision_matrix(phi, T)
        W_v = np.linalg.cholesky(Q_inv)  # 稀疏 Cholesky 分解
        W.append(W_v)

    return W

import numpy as np

from MA.Tutorial.functions.CV_Tool import CV_Tool
from MA.Tutorial.functions.Duncan_prep import Duncan_Prep
from MA.Tutorial.functions.MCCAtool import whiten
from MA.Tutorial.functions.config import CONFIG

prep = Duncan_Prep(sub_range=np.array([1, 2, 3, 4, 5, 6]), VT_atlas='HA')
design_matrix, _ = prep.design_matrix(plot=False)
data = prep.masker(vt_idx = [16]) # 7


y_whitened_all = []
X_whitened_all = []
for X,y in zip(design_matrix, data):
    y = data[0].T  # (V x T) 顶点时间序列
    X = design_matrix[0]  # (T x P) 设计矩阵
    y = np.asarray(y)
    X = np.asarray(X)
    # 5. 合并并应用
    W = prewhiten(y, X, p=6)
    y_whitened = np.concatenate([W_v @ y_v for W_v, y_v in zip(W, y)]).reshape(*y.shape)
    X_whitened = np.stack([W_v @ X for W_v in W], axis=0)

    y_whitened_all.append(y_whitened)
    X_whitened_all.append(X_whitened)




window_size = 5  # find peak in a certain window size
peaks = []
X = []
Y = []
events_types = ['Words', 'Objects', 'Scrambled objects', 'Consonant strings']
for sub in range(len(y_whitened_all)):
    x_sub = []
    y_sub = []
    for idx in range(len(events_types)):
        x = []
        y = []
        for v in range(X_whitened_all[0].shape[0]):
            dat = X_whitened_all[sub][v, :, idx]
            # find_peaks with window size of 5: find local maximum
            # as window size is not fixed as 5 in design matrix, can be 6, or 7
            # just find the local maximum, and extract the two neighbours to form the window with len 5
            peak, _ = find_peaks(dat, distance=window_size)

            x = np.array([y_whitened_all[sub].T[p - 2:p + 3, :] for p in peak])
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
            y = np.array([[idx + 1] * 5 for p in peak])
            y = y.reshape(-1)
            # peaks.append(peak)

        x_sub.append(x)
        y_sub.append(y)

    X.append(x_sub)
    Y.append(y_sub)

# Nr_sub = len(X)
# Nr_event = len(X[0])
# Nr_trial = len(X[0][0])
# Nr_rep = X[0][0][0].shape[0]
# Nr_vox = X[0][0][0].shape[1]

X_array = [np.array(sub) for sub in X]
X_array = [sub.reshape(-1, sub.shape[-1]) for sub in X_array]
Y_array = [np.array(label) for label in Y]
Y_array = [label.reshape(-2) for label in Y_array]