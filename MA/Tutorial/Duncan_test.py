import numpy as np

from MA.Tutorial.functions.CV_Tool import CV_Tool
from MA.Tutorial.functions.Duncan_prep import Duncan_Prep
from MA.Tutorial.functions.MCCAtool import whiten, center, PCA_60
from MA.Tutorial.functions.config import CONFIG

prep = Duncan_Prep(sub_range=np.array([1, 2, 3, 4, 5, 6]), VT_atlas='HA')
design_matrix, _ = prep.design_matrix(plot=False)
data = prep.masker(vt_idx = [23,34,35]) # 7
# prep.masker(vt_idx=[15, 16, 23, 34, 35, 38, 39, 40])  # 7
X, Y = prep.prepare_data()
CONFIG["PCA"]["n_pcs"] = int(120)
CONFIG["MCCA"]["n_components_mcca"] = int(100)
CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False, nested=True)
CV.inter_sub(X, Y)


# 15 nested CV
# 0 0.328125
# 1 0.22333829365079363
# 2 0.29558375652125657
# 3 0.27294146825396826
# 4 0.35607638888888893
# 5 0.3115575396825397

# [23,34,35]
# 0 0.36533189033189034
# 1 0.34487509018759016
# 2 0.4102316086691087
# 3 0.36307043650793647
# 4 0.34293154761904765
# 5 0.3433351370851371

# for vt_idx in np.arange(49):
#     prep.masker(vt_idx=vt_idx)
#     X, Y = prep.prepare_data()
#     X = [PCA_60(sub) for sub in X]
#     for n_pc in np.linspace(10, 100, 10):
#         CONFIG["PCA"]["n_pcs"] = int(n_pc)
#
#         for n_cc in np.linspace(10, 100, 10):
#             if n_cc > n_pc:
#                 continue
#             else:
#                 CONFIG["MCCA"]["n_components_mcca"] = int(n_cc)
#                 CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False)
#                 print(f'vt_idx: {vt_idx}, with n_pc: {n_pc}, n_cc: {n_cc}')
#                 CV.inter_sub(X, Y)

# PCA 30, atlas 7 Precentral Gyrus,random_voxel=True
# 0 0.6166666666666667
# 1 0.6083333333333333
# 2 0.55
# 3 0.775
# 4 0.7166666666666667
# 5 0.7333333333333333

# PCA 30, atlas 7 Precentral Gyrus,random_train=True
# 0 0.4166666666666667
# 1 0.5083333333333333
# 2 0.5166666666666666
# 3 0.43333333333333335
# 4 0.3666666666666667
# 5 0.5083333333333333

# PCA 30, atlas 7 Precentral Gyrus,permute=True
# 0 0.24166666666666667
# 1 0.30000000000000004
# 2 0.29166666666666663
# 3 0.27499999999999997
# 4 0.29166666666666663
# 5 0.26666666666666666

# PCA 60, CC: 10 atlas 7 Precentral Gyrus,permute=True
# 0 0.44999999999999996
# 1 0.39166666666666666
# 2 0.4583333333333333
# 3 0.45833333333333337
# 4 0.4833333333333333
# 5 0.4333333333333333

# PCA 70, CC: 10 atlas 7 Precentral Gyrus,permute=True
# 0 0.44999999999999996
# 1 0.39166666666666666
# 2 0.4583333333333333
# 3 0.45833333333333337
# 4 0.4833333333333333
# 5 0.4333333333333333

# PCA 70, CC: 20 atlas 7 Precentral Gyrus,permute=True
# 0 0.5916666666666667
# 1 0.6083333333333333
# 2 0.5583333333333333
# 3 0.6166666666666667
# 4 0.6083333333333333
# 5 0.5916666666666667

# PC 70 CC 20, vt_idx=[15, 16, 23, 34, 35, 38, 39, 40]
# 0 0.775
# 1 0.7833333333333332
# 2 0.8916666666666667
# 3 0.6916666666666667
# 4 0.8
# 5 0.7666666666666666


# PCA 100, CC 100, atlas 7 Precentral Gyrus
# 0 1.0
# 1 0.9833333333333334
# 2 0.9916666666666667
# 3 1.0
# 4 1.0
# 5 1.0


# 16: Inferior Temporal Gyrus, temporooccipital part, PC=30, CC=100(but = 30),random_voxel=True
# 0 0.7083333333333333
# 1 0.7083333333333333
# 2 0.7583333333333334
# 3 0.8250000000000001
# 4 0.8333333333333333
# 5 0.7416666666666667

# 16: Inferior Temporal Gyrus, temporooccipital part, PC=30, CC=100(but = 30),random_train=True
# 0 0.5166666666666666
# 1 0.5083333333333333
# 2 0.4666666666666667
# 3 0.5583333333333333
# 4 0.4916666666666667
# 5 0.5

# 16: Inferior Temporal Gyrus, temporooccipital part, PC=30, CC=100(but = 30),random_train=True, random_voxel=True
# 0 0.43333333333333335
# 1 0.5083333333333333
# 2 0.44166666666666665
# 3 0.49166666666666664
# 4 0.49166666666666664
# 5 0.5583333333333333

# 16: Inferior Temporal Gyrus, temporooccipital part, PC=30, CC=100(but = 30), permute = True
# 0 0.20833333333333331
# 1 0.2
# 2 0.2
# 3 0.23333333333333334
# 4 0.2
# 5 0.20833333333333331
