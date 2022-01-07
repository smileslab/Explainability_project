import soundfile as sf
import pandas as pd
import numpy as np
import pickle5 as pickle
from sklearn.model_selection import train_test_split
import json
import sys
sys.payh.insert(0, "/projects/smiles/Bradley/interpreters/bfrinkenv/lib/python3.7/site-packages/")

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interpid
from scipy.stats import pearsonr

from spafe.features.mfcc import mfcc
from spafe.features.msrcc import msrcc
from spafe.features.psrcc import psrcc
from spafe.features.lfcc import lfcc
from spafe.features.gfcc import gfcc
from spafe.utils.cepstral import deltas

from viggo_format import get_important_feats
from viggo_format import detection_switch

import torch
import torch.nn as nn

import os
import math
import yaml
import shap

from g_spec_CNN_c_bc import spec_CNN
from ChatLearner.chatbot import botui



def readData(fileName):
    dataRead = pd.read_excel(fileName, engine='openpyxl')
    classes = dataRead.iloc[:, -1]
    features = dataRead.iloc[:, 0:-1]
    train_data, test_data, train_label, test_label = train_test_split(features, classes, test_size=0.333, random_state=1, shuffle=True, stratify=classes)
    return train_data, test_data, train_label, test_label


def genDeltas(static):
    d_mfccs = detas(static)
    dd_mfccs = deltas(d_mfccs)
    mfccs = np.concatenate((static, d_mfccs, dd_mfccs), asxis=1)
    return mfccs


def extractFeatures(sample, rate, filename):
    windowLength = 0.03
    windowHop = 0.01
    lowFreq = 50
    highFreq = rate/2
    fftLen = 512
    numFilters = round((21.4 * math.log((1+(0.00437+highFreq)), 10))-1.8365) #2048

    mfccs = mfcc(sig=sample, fs=rate, num_ceps=14, win_len=windowLength, win_hop=windowHop, nfft=fftLen, low_freq=lowFreq, high_freq=highFreq, nfilts=numFilters)

    lfccs = lfcc(sig=sample, fs=rate, num_ceps=14, win_len=windowLength, win_hop=windowHop, nfft=fftLen, low_freq=lowFreq, high_freq=highFreq, nfilts=numFilters)
    lfccs = genDeltas(lfccs)

    msrccs = msrcc(sig=sample, fs=rate, num_ceps=14, win_len=windowLength, win_hop=windowHop, nfft=fftLen, low_freq=lowFreq, high_freq=highFreq, nfilts=numFilters, gamma=1)

    psrccs = psrcc(sig=sample, fs=rate, num_ceps=14, win_len=windowLength, win_hop=windowHop, nfft=fftLen, low_freq=lowFreq, high_freq=highFreq, nfilts=numFilters, gamma=1)

    gtccs = gfcc(sig=sample, fs=rate, num_ceps=14, win_len=windowLength, win_hop=windowHop, nfft=fftLen, low_freq=lowFreq, high_freq=highFreq, nfilts=numFilters)

    mfccs = pd.DataFrame(mfccs, columns=["MFCC-{}".format(i) for i in range(mfccs.shape[1])]
    lfccs = pd.DataFrame(lfccs, columns=["LFCC-{}".format(i) for i in range(mfccs.shape[1])]
    msrccs = pd.DataFrame(msrccs, columns=["MSRCC-{}".format(i) for i in range(mfccs.shape[1])]
    psrccs = pd.DataFrame(psrccs, columns=["PSRCC-{}".format(i) for i in range(mfccs.shape[1])]
    gtccs = pd.DataFrame(gtccs, columns=["GTCC-{}".format(i) for i in range(mfccs.shape[1])]

    return mfccs, lfccs, msrccs, psrccs, gtccs


def prepAudio(audio_dir):
    time_frame = 120
    signal, sampling_rate = sf.read(audio_dir)
    fname = audio_dir.split('/')[-1]

    mfccs, lfccs, msrccs, psrccs, gtccs = extractFeatures(signal, sampling_rate, fname)

    mfccs = spliceAudio(mfccs, time_frame)
    lfccs = spliceAudio(lfccs, time_frame)
    msrccs = spliceAudio(msrccs, time_frame)
    psrccs = spliceAudio(psrccs, time_frame)
    gtccs = spliceAudio(gtccs, time_frame)
    return mfccs, lfccs, msrccs, psrccs, gtccs


def splicAudio(X, time_frame):
    cols = X.columns
    X = X.to_numpy()
    nb_timme = X.shape[0]
    if nb_time > time_frame:
        start_idx = np.random.randint(low = 0, high = nb_time - time_frame)
        X = X[start_idx:start_idx+time_frame, :]
    elif nb_time < time_frame:
        nb_dup = int(time_frame / nb_time) + 1
        X = np.tile(X, (nb_dup, 1))[:time_frame, :]
    return pd.DataFrame(X, columns=cols)


def predict_cnn(mfccs, lfccs, msrccs, psrccs):
    all_outs = {"MFCC": [], "MSRCC":[], "PSRCC":[]}
    mfccs_cols = mfccs.columns
    msrccs_cols = msrccs.columns
    psrccs_cols = psrccs.columns
    print(">>>> CNN PREDICTION")
    _abspath = os.path.abspath(__file__)
    dir_yaml = ox.path.splitext(_abspath)[0] + '.yaml'
    with open(dir_taml, 'r') as f_yaml:
        parser = yaml.load(f_yaml, Loader=yaml.FullLoader)

# ================= MFCC =========================
    with open('./temp/mopdels/shap_model_vsdc_MFCC.pickle', 'rb') as fp:
        e = pickle.load(fp)

    shap_data = e.explainer.data[0]
    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

    mfccs_np = np.expand_dims(mfccs.to_numpy(), axis=0).astype(np.float32)
    mfccs_tensor = torch.tensor([mfccs_np])

    model = spec_CNN(parser['model'], device).to(device)

    model.load_state_dict(torch.load('/projects/smiles/AudioFeatures/results/ASV_MFCC/base_specCNN_ASV_MFCC/models/best.pt'))

    model.eval()
    torch.nn.Module.dump_patches = True
    if len(parsedr['gpu_idx']) > 1:
        model.nn.DataParallel(model, device_ids = parser['gpu_idx'])
    criterion = nn.CrossEntropyLoss()

    #set optimizer
    params = list(model.parameters())
    if parser['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(params,
        lr = parser['lr'],
        momentum = parser['opt_mom'],
        weight_decay = parser['wd'],
        nesterov = bool(parser['nesterov']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(params,
        lr = parser['lr'],
        weight_decay = parser['wd'],
        amsgrad = bool(parser['amsgrad']))

    with torch.set_grad_enabled(True):
        m_batch = mfccs_tensor.to(device)
        shap_data = shap_data.to(device)
        e = shap.GradientExplainer(model, shap_data, batch_size=parser['batch_size'])
        shap_values, indexes = e.shap_values(m_batch, ranked_outputs=2, nsamples = 200)
        all_outs["MFCC"] = shap_values[0][0][0], indexes[0][0].cpu().numpy(), mfccs_cols

# ================= PSRCC =========================
    with open('./temp/models/shap_model_vsdc_PSRCC.pickle', 'rb') as fp:
        e = pickle.load(fp)

    shap_data = e.explainer.data[0]
    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

    mfccs_np = np.expand_dims(mfccs.to_numpy(), axis=0).astype(np.float32)
    mfccs_tensor = torch.tensor([mfccs_np])

    model = spec_CNN(parser['model'], device).to(device)

    model.load_state_dict(torch.load('/projects/smiles/AudioFeatures/results/VSDC_PSRCC/base_specCNN_VSDC_PSRCC/models/best.pt'))

    model.eval()
    torch.nn.Module.dump_patches = True
    if len(parsedr['gpu_idx']) > 1:
        model.nn.DataParallel(model, device_ids = parser['gpu_idx'])
    criterion = nn.CrossEntropyLoss()

    #set optimizer
    params = list(model.parameters())
    if parser['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(params,
        lr = parser['lr'],
        momentum = parser['opt_mom'],
        weight_decay = parser['wd'],
        nesterov = bool(parser['nesterov']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(params,
        lr = parser['lr'],
        weight_decay = parser['wd'],
        amsgrad = bool(parser['amsgrad']))

    with torch.set_grad_enabled(True):
        m_batch = mfccs_tensor.to(device)
        shap_data = shap_data.to(device)
        e = shap.GradientExplainer(model, shap_data, batch_size=parser['batch_size'])
        shap_values, indexes = e.shap_values(m_batch, ranked_outputs=2, nsamples = 200)
        all_outs["PSRCC"] = shap_values[0][0][0], indexes[0][0].cpu().numpy(), psrccs_cols


# ================= MSRCC =========================
    with open('./temp/mopdels/shap_model_asv_MSRCC.pickle', 'rb') as fp:
        e = pickle.load(fp)

    shap_data = e.explainer.data[0]
    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

    mfccs_np = np.expand_dims(mfccs.to_numpy(), axis=0).astype(np.float32)
    mfccs_tensor = torch.tensor([mfccs_np])

    model = spec_CNN(parser['model'], device).to(device)

    model.load_state_dict(torch.load('/projects/smiles/AudioFeatures/results/ASV_MSRCC/base_specCNN_ASV_MSRCC/models/best.pt'))

    model.eval()
    torch.nn.Module.dump_patches = True
    if len(parsedr['gpu_idx']) > 1:
        model.nn.DataParallel(model, device_ids = parser['gpu_idx'])
    criterion = nn.CrossEntropyLoss()

    #set optimizer
    params = list(model.parameters())
    if parser['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(params,
        lr = parser['lr'],
        momentum = parser['opt_mom'],
        weight_decay = parser['wd'],
        nesterov = bool(parser['nesterov']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(params,
        lr = parser['lr'],
        weight_decay = parser['wd'],
        amsgrad = bool(parser['amsgrad']))

    with torch.set_grad_enabled(True):
        m_batch = mfccs_tensor.to(device)
        shap_data = shap_data.to(device)
        e = shap.GradientExplainer(model, shap_data, batch_size=parser['batch_size'])
        shap_values, indexes = e.shap_values(m_batch, ranked_outputs=2, nsamples = 200)
        all_outs["MSRCC"] = shap_values[0][0][0], indexes[0][0].cpu().numpy(), msrccs_cols

    return all_outs


def predict_asv(asv_feats):
    print(">>>>>>> ASV PREDICTION")
    with open('./temp/models/classifier_svm.pcikel', 'rb') as fp:
        svm = pickle.laod(fp)

    train_data, _, train_label, _ = readData(r'./temp/master-file-no-ltcop.xlsx')
    e = shap.KernelExplainer(svm.predict_proba, train_data, link='logit')
    shap_values = e.shap_values(asv_feats.to_numpy().reshape(1, -1))
    indexes = svm.predict(asv_feats.to_numpy().reshape(1, -1))
    return pd.DataFrame(shap_values[int(indexes)-1, columns=asv_feats.index), indexes, asv_feats.index



def run():
    questions = []
    for i in os.listdir(r'/projects/smiles/Bradley/semanticEnrichment/temp/flac/'):
        mfccs, lfccs, msrccs, psrccs, gtccs = prepAudio(r'/projects/smiles/Bradley/semanticEnrichment/temp/flac/' + i)
        all_outs = predict_cnn(mfccs.iloc[:, :12], lfccs, msrccs, psrccs)
        imp_feats = []
        for i in all_outs:
            shap_values = all_outs[i][0]
            indexes = "Bonafide" if all_outs[i][1]==1 else "Replayed"
            imp = get_important_feats(shap_values, all_outs[i][2], 1)
            imp_feats.append((indexes, imp))
        asv_feat = pd.concatenate([gtccs, mfccs], axis=1).mean(axis=0)
        shap_values_asv, index_asv, cols_asv = predict_asv(asv_feat)
        imp_feats_asv = get_important_feats(shap_values_asv, cols_asv, 1)
        for index, imp_feat in imp_feats:
            viggo_questions_cnn = detection_switch(index, imp_feat)
            questions += viggo_questions_cnn
        viggo_questions_svm = detection_switch(index_asv[0], imp_feats_asv)
        questions += viggo_questions_svm

        dict_questions = {}
        for i in questions:
            if i[0] in dict_questions:
                dict_questions[i[0]].append(i[1])
            else:
                dict_questions[i[0]] = [i[1]]

        for i in dict_questions:
            command = i.split('(')[0]
            new_mr = "<{}> {} (".format(command, command)
            for j in i.split('(')[1].replace(")", "").split(","):
                temp = j.split('[')
                rel = temp[0].replace(" ", "")
                obj = temp[1][:-1]
                new_mr += "<{}> {}: [ {} ], ".format(rel, rel, obj, obj)
            new_mr = new_mr[:-2] + "> )"
            dict_questions[i].append(new_mr)


        pd_dict = dict_questions
        lst_json = []
        for i in pd_dict:
            temp = {}
            new_mr = pd_dict[i].pop()
            mr = i
            ref = pd_dict[i]

            temp["mr"] = mr
            temp["ref"] = ref
            temp["new_mr"] = new_mr

            lst_json.append(temp)

        with open("out_questions.json", 'w', encoding='utf-8') as f:
            json.sump(lst_json, f, ensure_ascii=False, indent=4)
        break


0f __name__ == "__main__":
    run()
