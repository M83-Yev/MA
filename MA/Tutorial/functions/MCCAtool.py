import numpy as np
from sklearn.base import TransformerMixin
from scipy.linalg import eigh, norm
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .config import CONFIG


class MCCA(TransformerMixin):
    def __init__(self, n_components_mcca=10, r=0):
        self.n_ccs = n_components_mcca if n_components_mcca is not None else CONFIG["MCCA"]["n_components_mcca"]
        self.r = r if r is not None else CONFIG["MCCA"]["r"]
        self.mcca_weights = None
        self.mcca_weights_inverse_ = None
        self.X_mcca_avg = None

    # Assuming X is an 3D array
    # def fit(self, X):
    #     n_subjects, _, n_pcs = X.shape
    #     R_kl, R_kk = self._compute_cross_covariance(X)
    #
    #     # TODO: regularization to be added in
    #
    #     p, h = eigh(R_kl, R_kk, subset_by_index=(n_subjects * n_pcs - self.n_ccs, n_subjects * n_pcs - 1))
    #     h = np.flip(h, axis=1).reshape((n_subjects, n_pcs, self.n_ccs))
    #     self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
    #     return self

    # Assuming X with subs in diff. length (voxels/channels)
    def fit(self, X):

        R_kl, R_kk = self._compute_cross_covariance(X)
        # R_kl, R_kk = self._compute_cross_covariance(X.reshape(X.shape[0]*X.shape[1], X.shape[2]))
        # TODO: regularization to be added in
        # solving non-positive definite problem in scipy linalg\_decomp (Solving generalized eigenvalue problem)
        R_kk = R_kk + 1e-9 * np.eye(R_kk.shape[0])
        if self.n_ccs > self.n_pcs_list[0]:
            self.n_ccs = self.n_pcs_list[0]
        p, h_all = eigh(R_kl, R_kk, subset_by_index=(self.size - self.n_ccs, self.size - 1))

        # # h = np.flip(h, axis=1).reshape((n_subjects, n_pcs, self.n_ccs))
        # h, self.mcca_weights = [], []
        # idx_start = 0
        # for sub, n_pcs in enumerate(self.n_pcs_list):
        #     idx_end = idx_start + n_pcs
        #     # h.append(np.flip(h_all[idx_start:idx_end, :self.n_ccs], axis=0))
        #     h.append(np.flip(h_all[idx_start:idx_end, :self.n_ccs], axis=1))
        #     idx_start = idx_end
        #     self.mcca_weights.append(h[sub] / norm(h[sub], ord=2, keepdims=True))
        #     # self.mcca_weights.append(h[sub] / norm(X[sub] @ h[sub], ord=2, keepdims=True))

        h_all = np.flip(h_all, axis=1)
        # Reshape h from (subjects * PCs, CCs) to (subjects, PCs, CCs)
        h = h_all.reshape((len(self.n_pcs_list), self.n_pcs_list[0], self.n_ccs))
        # Normalize eigenvectors per subject
        self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
        # self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)

        return self

    # def fit(self, X):
    #
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    #     R_kl, R_kk = self._compute_cross_covariance(X)
    #
    #     # TODO: regularization to be added in
    #     # solving non-positive definite problem in scipy linalg\_decomp (Solving generalized eigenvalue problem)
    #     R_kk = R_kk + 1e-9 * np.eye(R_kk.shape[0])
    #
    #     R_kl = torch.tensor(R_kl, dtype=torch.float32).to(device)
    #     R_kk = torch.tensor(R_kk, dtype=torch.float32).to(device)
    #
    #     # torch does not support GEP, so Av = lambda B V must trans to: B_inv A v = lambda v
    #     R_kk_inv = torch.inverse(R_kk)
    #     R_kk_invRkl = torch.matmul(R_kk_inv, R_kl)
    #
    #     p, h_all = torch.linalg.eig(R_kk_invRkl)
    #     p = torch.real(p)
    #
    #     _, indices = torch.sort(p, descending=True)
    #     h_all = h_all[:, indices]
    #
    #     # h = np.flip(h, axis=1).reshape((n_subjects, n_pcs, self.n_ccs))
    #     h, self.mcca_weights = [], []
    #     idx_start = 0
    #     for sub, n_pcs in enumerate(self.n_pcs_list):
    #         idx_end = idx_start + n_pcs
    #         h.append(np.flip(h_all[idx_start:idx_end, :self.n_ccs], axis=0))
    #         idx_start = idx_end
    #         self.mcca_weights.append(h[sub] / norm(h[sub], ord=2, keepdims=True))
    #         # self.mcca_weights.append(h[sub] / norm(X[sub] @ h[sub], ord=2, keepdims=True))
    #
    #     # self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
    #     return self

    def transform(self, X):
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling transform')
        self.X_mcca_transformed = [np.matmul(X[i], self.mcca_weights[i]) for i in range(len(X))]
        # self.X_mcca_avg = np.mean(self.X_mcca_transformed, axis=0)
        return self.X_mcca_transformed

    def mcca_space(self):
        self.X_mcca_avg = np.mean(self.X_mcca_transformed, axis=0)
        return self.X_mcca_avg

    def fit_new_data(self, X_new, train_idx = None):
        if self.X_mcca_avg is None:
            raise NotFittedError('MCCA average needs to be computed before calling fit_new_data')
        # print(f"X_new {X_new.shape}, self.X_mcca_avg {self.X_mcca_avg.shape}")
        if train_idx is not None:
            self.new_mcca_weights = np.dot(np.linalg.pinv(X_new), self.X_mcca_avg[train_idx])
        else:
            self.new_mcca_weights = np.dot(np.linalg.pinv(X_new), self.X_mcca_avg)

        return self

    def transform_new_data(self, X_new):
        if self.new_mcca_weights is None:
            raise NotFittedError('New MCCA weights need to be fitted before calling transform_new_data')
        # print(f"X_new {X_new.shape}, self.new_mcca_weights {self.new_mcca_weights.shape}")
        X_new_mcca_transformed = np.dot(X_new, self.new_mcca_weights)
        return X_new_mcca_transformed

    # Assuming X is an 3D array
    # def _compute_cross_covariance(self, X):
    #     n_subjects, n_samples, n_pcs = X.shape
    #     R = np.cov(X.swapaxes(1, 2).reshape(n_subjects * n_pcs, n_samples))
    #     R_kk = R * np.kron(np.eye(n_subjects), np.ones((n_pcs, n_pcs)))
    #     R_kl = R - R_kk
    #
    #     return R_kl, R_kk

    # Assuming X with subs in diff. length (voxels/channels)

    def _compute_cross_covariance(self, X):

        self.n_subjects = len(X)
        self.n_pcs_list = [x.shape[1] for x in X]  #
        self.size = np.sum(self.n_pcs_list)
        # self.r_kk_list = []

        R_kk = np.zeros((self.size, self.size))
        R_kl = np.zeros((self.size, self.size))
        outer_idx_start = 0  # row index

        for idx, x in enumerate(X):
            # r_kk is the covariance matrix of each subject
            r_kk = np.cov(x.T)
            n_pcs = self.n_pcs_list[idx]
            outer_idx_end = outer_idx_start + n_pcs

            # r_kk locates on diagonal position
            R_kk[outer_idx_start:outer_idx_end, outer_idx_start:outer_idx_end] = r_kk
            # self.r_kk_list.append(r_kk)
            # r_kl is a list of the covariance matrix of two diff subjects, will locate on non-diagonal positions
            # initializing column index in outer loop
            inner_idx_start = 0
            for j in range(self.n_subjects):

                n_pcs_j = self.n_pcs_list[j]
                inner_idx_end = inner_idx_start + n_pcs_j

                if j != idx:
                    r_kl = np.cov(x.T, X[j].T)
                    # n_pcs_j = self.n_pcs_list[j]
                    # inner_idx_end = inner_idx_start + n_pcs_j

                    # locate r_kl on corresponding potions
                    R_kl[outer_idx_start:outer_idx_end, inner_idx_start:inner_idx_end] = r_kl[:n_pcs, n_pcs:]
                    R_kl[inner_idx_start:inner_idx_end, outer_idx_start:outer_idx_end] = r_kl[:n_pcs, n_pcs:].T

                inner_idx_start = inner_idx_end

            # update row index from outer loop
            outer_idx_start = outer_idx_end

        return R_kl, R_kk


def center(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    return centered_data


def zscore(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    zscored_data = (data - mean) / std
    return zscored_data


# def whiten(data):
#     data_centered = center(data)
#     cov = np.cov(data_centered, rowvar=False)
#     U, S, V = np.linalg.svd(cov)
#     whitening_matrix = np.dot(U, np.diag(1.0 / np.sqrt(S)))
#     whitened_data = np.dot(data_centered, whitening_matrix)
#     return whitened_data

# def whiten(data):
#     # Center the data
#     whitened_data = np.empty_like(data, dtype=object)
#     for sub in data:
#         data_centered = data[sub] - np.mean(data[sub], axis=0)
#
#     # in case there is nan calculated
#     # std_dev = np.nanstd(data_centered, axis=0)
#     # std_dev[std_dev == 0] = 1e-10
#     # Apply PCA
#         pca = PCA(whiten=True)
#         whitened_data[sub] = pca.fit_transform(data_centered)
#
#     return whitened_data
def whiten(data):
    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # in case there is nan calculated
    # std_dev = np.nanstd(data_centered, axis=0)
    # std_dev[std_dev == 0] = 1e-10
    # Apply PCA
    # pca = PCA(n_components=60,whiten=True)
    pca = PCA(whiten=True)
    whitened_data = pca.fit_transform(data_centered)

    return whitened_data


def PCA_60(data, n_pcs, return_pca = None):
    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # in case there is nan calculated
    # std_dev = np.nanstd(data_centered, axis=0)
    # std_dev[std_dev == 0] = 1e-10
    # Apply PCA
    # n_pcs = CONFIG["PCA"]["n_pcs"]
    pca = PCA(n_components=n_pcs, whiten=True)
    whitened_data = pca.fit_transform(data_centered)
    #whitened_data = pca.fit_transform(data)
    if return_pca is not None:
        return whitened_data, pca
    else:
        return whitened_data

class PCA_MA:
    def __init__(self, _whiten=False, n_components=None):
        self.Vt = None
        self.S = None
        self.U = None
        self.components_ = None
        self.whiten = _whiten
        self.n_components = n_components

    def demean(self, X):
        X_demean = X - np.mean(X, axis=0)
        return X_demean

    def fit(self, X_demean):
        self.U, self.S, self.Vt = np.linalg.svd(X_demean, full_matrices=False)
        return self

    def transform(self, X_demean):
        if self.n_components is not None:
            self.Vt = self.Vt[: self.n_components]
            self.S = self.S[: self.n_components]

        self.components_ = self.Vt

#        if not self.whiten:
#            self.U *= self.S
#        else:
#            self.U *= np.sqrt(X_demean.shape[0] - 1)
#
#       return self.U
        X_proj = np.dot(X_demean, self.Vt.T)
        if not self.whiten:
            return X_proj
        else:
            X_whitened = X_proj / self.S
            X_whitened *= np.sqrt(X_demean.shape[0]-1)
            return X_whitened

    def fit_transform(self, X):
        X_demean = self.demean(X)
        self.fit(X_demean)
        return self.transform(X_demean)