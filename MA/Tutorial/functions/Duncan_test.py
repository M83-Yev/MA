import os
import numpy as np
from functions.CV_Tool import CV_Tool
from functions.Duncan_prep import Duncan_Prep
from functions.MCCAtool import whiten, center, PCA_60
from functions.config import CONFIG

prep = Duncan_Prep(sub_range=np.array([1, 2, 3, 4, 5, 6]), VT_atlas='HA')
design_matrix, _ = prep.design_matrix(plot=False)
# data = prep.masker(vt_idx = [15, 16, 23, 34, 35, 38, 39, 40]) # 7
# prep.masker(vt_idx=[15, 16, 23, 34, 35, 38, 39, 40])  # 7
# X, Y = prep.prepare_data()
# CONFIG["PCA"]["n_pcs"] = int(120)
# CONFIG["MCCA"]["n_components_mcca"] = int(100)
# CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False, nested=True)
# CV.inter_sub(X, Y)
mask, _ = prep.masker(vt_idx=[0])


print('Check HO atlas and cc pc numbers')
for vt_idx in range(48):
    #vt_idx = 38
    prep.masker(vt_idx=[vt_idx])
    # prep.masker(vt_idx=[15, 16, 23, 34, 35, 38, 39, 40])
    X, Y = prep.prepare_data()
    BAs = []
    #for n_pc in np.linspace(10, 100, 10):
    #    CONFIG["PCA"]["n_pcs"] = int(n_pc)
    #    X_whiten = [PCA_60(sub) for sub in X]
    #    for n_cc in np.linspace(10, 100, 10):
    #        if n_cc > n_pc:
    #            continue
    #        else:
    #            CONFIG["MCCA"]["n_components_mcca"] = int(n_cc)
    #            CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False,
    #                         nested=True)
    #            print(f'vt_idx: {vt_idx}, with n_pc: {n_pc}, n_cc: {n_cc}')
    #            ba = CV.inter_sub(X_whiten, Y)
    #    BAs.append(np.mean(ba))
    with open("HO atlas.txt") as file:
        l = file.readlines()
        area = l[vt_idx].rstrip("\n")
    for n_cc in np.linspace(90, 90, 1):
        CONFIG["MCCA"]["n_components_mcca"] = int(n_cc)
        CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False,
                     nested=True)

        print(f'vt_idx: {area}, with n_cc: {n_cc}')
        # print(f'with n_cc: {n_cc}')
        ba = CV.inter_sub(X, Y)
        BAs.append(np.mean(ba))

# Test on Duncan defined area
print('Now deal with data for Duncan defined brain area')
main_path = CONFIG["Prep"]["main_path"]
CONFIG["PCA"]["n_pcs"] = int(100)
X_array = np.load(os.path.join(main_path, 'X_array.npy'), allow_pickle=True)
Y_array = np.load(os.path.join(main_path, 'Y_array.npy'), allow_pickle=True)
# X_array = [PCA_60(X) for X in X_array]
CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False, nested=True)
CV.inter_sub(X_array, Y_array)

#0 0.3182539682539683
#1 0.2678217615717616
#2 0.18901289682539685
#3 0.1925843253968254
#4 0.18633432539682543
#5 0.17547123015873017

print('Check HO atlas and cc pc numbers')
for vt_idx in np.arange(49):
    prep.masker(vt_idx=[vt_idx])
    X, Y = prep.prepare_data()

    for n_pc in np.linspace(10, 100, 10):
        CONFIG["PCA"]["n_pcs"] = int(n_pc)
        X_whiten = [PCA_60(sub) for sub in X]
        for n_cc in np.linspace(10, 100, 10):
            if n_cc > n_pc:
                continue
            else:
                CONFIG["MCCA"]["n_components_mcca"] = int(n_cc)
                CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False,
                             nested=True)
                print(f'vt_idx: {vt_idx}, with n_pc: {n_pc}, n_cc: {n_cc}')
                CV.inter_sub(X_whiten, Y)

print('Check HO atlas and cc pc numbers')
BAs = []
for vt_idx in np.arange(49):
    CONFIG["PCA"]["n_pcs"] = 100
    CONFIG["MCCA"]["n_components_mcca"] = 100
    prep.masker(vt_idx=[vt_idx])
    X, Y = prep.prepare_data()
    X = [PCA_60(sub) for sub in X]

    CV = CV_Tool(method='MCCA', permute=False, seed=20241201, random_train=False, random_voxel=False, nested=True)
    print(f'vt_idx: {vt_idx}')
    ba = CV.inter_sub(X, Y)
    BAs.append(np.mean(ba))
a=1

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

# another run:
# 0 0.40295138888888893
# 1 0.42559142246642245
# 2 0.349934613997114
# 3 0.3882124819624819
# 4 0.3230654761904762
# 5 0.35022546897546897

# [15, 16, 23, 34, 35, 38, 39, 40] ~10h
# 0 0.3361967893217893
# 1 0.4036796536796537
# 2 0.3116026334776335
# 3 0.3638325216450216
# 4 0.5243461399711399
# 5 0.3214195526695527

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
