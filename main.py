# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
# from torchinfo import summary
from collections import OrderedDict
# Numpy, scipy, scikit-image, spectral
from visualize_tsne import visualize_tsne
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
from collections import Counter
# Visualization
import seaborn as sns
import visdom

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model, applyPCA
#from models_visual import get_model, train, test, save_model
import argparse


def select(train_x, indexes):
    # print(train_x.shape)
    temp = np.zeros((train_x.shape[0], train_x.shape[1], len(indexes)))
    for nb in range(0, len(indexes)):
        temp[:, :, nb] = train_x[:, :, indexes[nb]]
    train_x = temp.astype(np.float32)
    return train_x

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default="IndianPines", choices=dataset_names,
    help="Dataset to use  IndianPines GRSS  PaviaU HyRANK  PaviaC HyRANK KSC Salinas whulk whuhc XA."
)
parser.add_argument(
    "--model",
    type=str,
    default="SS_TMNet",
    # 2D-CNN 3D-CNN HybridSN SYCNN ViT Deep ViT CvT HiT SSFTT Morphformer ACTN  yang
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=0,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=0.01,
    help="Percentage of samples to use for training (default: 10%%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    #default="./Datasets/indian_train_gt.npy",
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    #default="./Datasets/indian_test_gt.npy",
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)

parser.add_argument(
    "--ablation",
    action="store_true",
    help="Run ablation study for BRWKVHSI model"
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
# img = applyPCA(img,numComponents=30) #ssftt
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

results = []
# indx = [16, 49, 126, 170] ## FGNBS for IP
# indx = [8, 48, 67, 173] ### OCF for IP
# indx = [7,53,123,168] ### ONR for IP

# indx = [11,39,46,88] ## FGNBS for PU
# indx = [19, 33, 61,88] ### OCF for PU
# indx = [17, 31, 58,91] ### ONR for PU


# indx = [15,55,88,123] ## FGNBS for grss
# indx = [23, 42, 58, 105] ### OCF for grss
# indx = [23, 42, 64, 107] ### ONR for grss
#
#
#
# img = select(img, indx)
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
        # np.save("./Datasets/indian_train_gt.npy", train_gt)
        # np.save("./Datasets/indian_test_gt.npy", test_gt)
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(test_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(gt), viz, caption="Test ground truth")

    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SGD":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight, learning_rate="optimal", tol=1e-3, average=10
        )
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "nearest":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.neighbors.KNeighborsClassifier(weights="distance")
        clf = sklearn.model_selection.GridSearchCV(
            clf, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    else:
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams["weights"] = torch.from_numpy(weights)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        # Split train set in train/val
        # train_gt, val_gt = sample_gt(train_gt, 0.9, mode="random")
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        # for train_data, label in train_dataset:
        #     print(label)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            # pin_memory=hyperparams['device'],
            shuffle=True,
        )
        # val_dataset = HyperX(img, val_gt, **hyperparams)
        # val_loader = data.DataLoader(
        #     val_dataset,
        #     # pin_memory=hyperparams['device'],
        #     batch_size=hyperparams["batch_size"],
        # )

        # print(hyperparams)
        # print("Network :")
        # with torch.no_grad():
        #     for input, _ in train_loader:
        #         break
        #     try:
        #         summary(model.to(hyperparams["device"]), input.size()[1:])
        #     except:
        #         print("get network framework error！")
        #     # We would like to use device=hyperparams['device'] altough we have
        #     # to wait for torchsummary to be fixed first.
        if args.ablation:
            from ablation_study import run_complete_ablation_study

            print("Running ablation study...")
            run_complete_ablation_study()
            quit()
        if MODEL == "hybridretnet":
            print(hyperparams)
            print("Network is not print:")
        else:
            print(hyperparams)
            print("Network :")
            # with torch.no_grad():
            #     for input, _ in train_loader:
            #         break
            #     try:
            #         # summary(model.to(hyperparams["device"]), input.size()[1:])
            #     except:
            #         print("get network framework error！")

        if CHECKPOINT is not None:
            pre_model = torch.load(CHECKPOINT, map_location='cuda:0')
            new_state_dict = OrderedDict()
            model_dict = model.state_dict()
            # for k, v in state_dict.items():
            #     name = k[:4]
            #     print(name)
            #     new_state_dict[name] = v
            state_dict = {k: v for k, v in pre_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
           # model.classifier.load_state_dict(checkpoint["classifier"])


        try:
            train(
                model,
                optimizer,
                loss,
                train_loader,
                hyperparams["epoch"],
                scheduler=hyperparams["scheduler"],
                device=hyperparams["device"],
                supervision=hyperparams["supervision"],
                # val_loader=test_loader,
                display=viz,
            )
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        t_sne = False
        if t_sne:
            # T-SNE
            # "checkpoints/fever_net/PaviaU/PaviaU_DCTN.pth"
            # checkpoints/cnn2_d/PaviaU/2023_09_18_21_09_40_epoch200_1.00.pth
            # train_gt, test_gt = sample_gt(gt, 0.2, mode=SAMPLING_MODE)
            if DATASET == "PaviaU":
                train_gt, test_gt = sample_gt(gt, 0.2, mode=SAMPLING_MODE)
            elif DATASET == "Houston2013":
                train_gt, test_gt = sample_gt(gt, 0.6, mode=SAMPLING_MODE)
            else:
                train_gt = gt
            model, _, _, hyperparams = get_model(MODEL, **hyperparams)
            # model.load_state_dict(torch.load(
                # "E:/UM/BF/DeepHyperX-Transformer/checkpoints/conformer/IndianPines/2024_03_08_10_08_27_epoch100_1.00.pth"))
            # model.load_state_dict(torch.load("checkpoints/hi_t/Houston/Houston_HiT.pth"))
            # model.load_state_dict(torch.load("checkpoints/hybrid_et_al/IndianPines/Indian_hybrid.pth"))
            # model.load_state_dict(torch.load("checkpoints/vi_t/PaviaU/PaviaU_ViT.pth"))
            # model.load_state_dict(torch.load("E:/UM/BF/DeepHyperX-Transformer/checkpoints/test_et_al/IndianPines/2023_09_07_15_20_08_epoch3_0.96.pth"))
            # model.load_state_dict(torch.load("checkpoints/cnn2_d/IndianPines/2023_09_19_10_34_02_epoch200_0.91.pth"))
            # model.load_state_dict(torch.load("checkpoints/morph_former/IndianPines/Indian_morph.pth"))
            # model.load_state_dict(torch.load("checkpoints/ss_tm_net/IndianPines/Indian_SS_TMNet.pth"))
            # model.load_state_dict(torch.load("E:/UM/BF/DeepHyperX-Transformer/checkpoints/ssft_tnet/IndianPines/2024_03_16_12_19_47_epoch20_1.00.pth"))
            # model.load_state_dict(torch.load(
            #     "E:/UM/BF/DeepHyperX-Transformer/checkpoints/hybrid_et_al/IndianPines/2024_04_01_15_54_23_epoch50_1.00.pth")) ##
            # model.load_state_dict(torch.load(
            #     "E:/UM/BF/DeepHyperX-Transformer/checkpoints/morph_former/IndianPines/2024_04_01_18_36_32_epoch20_0.91.pth"))  ##
            model.load_state_dict(torch.load(
                "E:/UM/BF/DeepHyperX-Transformer/checkpoints/efficient_fft_net2_d/IndianPines/2025_09_21_22_04_46_epoch100_0.00.pth"))  ##
            train_dataset = HyperX(img, train_gt, **hyperparams)
            # train_loader = data.DataLoader(
            #     train_dataset,
            #     batch_size=hyperparams["batch_size"],
            #     shuffle=True,
            # )
            # t_SNE(model, train_loader, device=hyperparams["device"], num_class=N_CLASSES - 1)
            visualize_tsne(
                model=model,
                data_loader=train_loader,
                num_classes=N_CLASSES,
                layer_name="classifier",  # 根据模型结构选择
                n_components=3,
                save_path="./t_sne/real_tsne.png",
                perplexity=30,
                learning_rate=500
            )

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(
        prediction,
        test_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        # gt=convert_to_color(gt),
        gt=None,
        caption="Prediction",
    )

    caption1 = "Prediction" + str(run + 1)
    txt_dir = "./cls_result" + "/" + DATASET + "/" + MODEL + "/"
    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir, exist_ok=True)
    filename = str(run+1)
    txt_path = txt_dir + filename + ".txt"
    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES, txt_path=txt_path)


if N_RUNS > 1:
    # show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
    txt_dir = "./cls_result" + "/" + DATASET + "/" + MODEL + "/"
    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir, exist_ok=True)
    filename = str(N_RUNS)+"_all"
    txt_path = txt_dir + filename + ".txt"
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True, txt_path=txt_path)
