# standard libraries
import os
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim

# custom imports
# from cwru_dataset import get_raw_1d
# from cwru_dataset_all import get_raw_1d
from cwru_dataset_normal import get_raw_1d

from models.CNN.train_cnn import train_cnn
from models.dann_mm.train_dann_mm import train_dann_mm
from models.dann_mm2.train_dann_mm2 import train_dann_mm2
from models.DDC.train_ddc import train_ddc
from models.DeepCoral.train_deepcoral import train_deepcoral
from models.DAN_JAN.train_dan_jan import train_dan_jan
from models.dann.train_dann import train_dann
from models.adda.train_adda import train_adda
from models.CDAN.train_cdan import train_cdan
from models.Wasserstein.train_wasserstein import train_wasserstein
from models.MCD.train_mcd import train_mcd
from models.MCD_A.train_mcd_a import train_mcd_a

from models.CDAN_VAT.train_cdan_vat import train_cdan_vat
from models.CDAN_ICAN.train_cdan_ican import train_cdan_ican
from models.CDAN_IW.train_cdan_iw import train_cdan_iw
from models.DCTLN.train_dctln import train_dctln
from models.PADA.train_pada import train_pada
from models.dann_vat.train_dann_vat import train_dann_vat

def train(config):
    if config['models'] == 'sourceonly':
        train_cnn(config)
    if config['models'] == 'dann_mm':
       train_dann_mm(config)
    if config['models'] == 'dann_mm2':
       train_dann_mm2(config)
    elif config['models'] == 'deepcoral':
        train_deepcoral(config)
    elif config['models'] == 'ddc':
        train_ddc(config)
    elif config['models'] in ['JAN', 'JAN_Linear', 'DAN', 'DAN_Linear']:
        train_dan_jan(config)
    elif config['models'] == 'dann':
        train_dann(config)
    elif config['models'] == 'adda':
        train_adda(config)
    elif config['models'] == 'wasserstein':
        train_wasserstein(config)
    elif config['models'] in ['CDAN', 'CDAN-E', 'DANN']:
        train_cdan(config)
    elif config['models'] == 'MCD':
        train_mcd(config)
    elif config['models'] == 'MCD_A':
        train_mcd_a(config)
    elif config['models'] == 'CDAN_VAT':
        train_cdan_vat(config)
    elif config['models'] in ['CDAN_ICAN', 'DANN_IW', 'DANN_EIW']:
        train_cdan_ican(config)
    elif config['models'] == 'CDAN_IW':
        train_cdan_iw(config)
    elif config['models'] == 'DCTLN':
        train_dctln(config)
    elif config['models'] == 'PADA':
        train_pada(config)
    elif config['models'] == 'dann_vat':
        train_dann_vat(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    # dataset configs
    parser.add_argument('--dataroot', required=False, default='data', help='dataroot')
    parser.add_argument('--src', required=False, default='0HP', help='folder name of src dataset')
    parser.add_argument('--dest', required=False, default='3HP', help='folder name of dest dataset')
    parser.add_argument('--normal', type=int, required=False, default=0, help='')
    parser.add_argument('--network', required=False, default='cnn', help='which type of network to use. cnn / inceptionv1 / inceptionv1s')
    parser.add_argument('--dilation', type=int, required=False, default=1, help='')

    parser.add_argument('--snr', type=int, required=False, default=0, help='')
    parser.add_argument('--testonly', type=int, required=False, default=0, help='')
    parser.add_argument('--split', type=float, required=False, default=0.5, help='')
    parser.add_argument('--inception', type=int, required=False, default=0, help='')
    parser.add_argument('--aux_classifier', type=int, required=False, default=1, help='')

    # model & loss
    parser.add_argument('--models', type=str, default='CDAN_ICAN', help="choose which model to run")
    parser.add_argument('--n_epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--loss_name', type=str, nargs='?', default='JAN', help="loss name")
    parser.add_argument('--tradeoff', type=float, nargs='?', default=1, help="tradeoff")
    parser.add_argument('--n_flattens', type=int, nargs='?', default=32, help="")
    parser.add_argument('--n_hiddens', type=int, nargs='?', default=500, help="")
    parser.add_argument('--TEST_INTERVAL', type=int, nargs='?', default=10, help="")
    parser.add_argument('--VIS_INTERVAL', type=int, nargs='?', default=100, help="")

    # model specific configs
    parser.add_argument('--lr', required=False, type=float, default=1e-3, help='')
    parser.add_argument('--flr', required=False, type=float, default=0.0001, help='')
    parser.add_argument('--clr', required=False, type=float, default=0.0001, help='')

    # for wasserstein
    parser.add_argument('--w_weight', required=False, type=float, default=1.0, help='')
    parser.add_argument('--w_gamma', required=False, type=float, default=10.0, help='')
    parser.add_argument('--t_weight', required=False, type=float, default=1.0, help='')
    parser.add_argument('--t_margin', required=False, type=float, default=1.0, help='')
    parser.add_argument('--t_confidence', required=False, type=float, default=0.9, help='')
    parser.add_argument('--triplet_type', required=False, type=str, default='none', help='')
    parser.add_argument('--mmd_gamma', required=False, type=float, default=1.0, help='')

    # for cdan
    parser.add_argument('--random_layer', required=False, default=False, help='')

    # from mcd
    parser.add_argument('--mcd_onestep', required=False, type=int, default=0, help='')
    parser.add_argument('--mcd_vat', required=False, type=int, default=1, help='')
    parser.add_argument('--mcd_entropy', required=False, type=int, default=1, help='')
    parser.add_argument('--mcd_swd', required=False, type=int, default=0, help='')

    parser.add_argument('--pada_cons_w', required=False, type=float, default=1.0, help='')
    parser.add_argument('--slim', required=False, type=int, default=0, help='')
    parser.add_argument('--target_labeling', required=False, type=int, default=0, help='')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    res_dir = 'snapshots_{}/{}--{}'.format(args.models, args.src, args.dest)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    src_dataset = os.path.join(args.dataroot, args.src)
    tgt_dataset = os.path.join(args.dataroot, args.dest)

    config = {}

    config['source_train_loader'], config['source_test_loader'], classes = get_raw_1d(src_dataset, 
                                                                                        batch_size=args.batch_size, 
                                                                                        trainonly=False, 
                                                                                        split=0.8, 
                                                                                        snr=args.snr, 
                                                                                        normal=args.normal,
                                                                                        slim=0)
    config['target_train_loader'], config['target_test_loader'], _ = get_raw_1d(tgt_dataset, 
                                                                                        batch_size=args.batch_size, 
                                                                                        trainonly=False, 
                                                                                        split=args.split, 
                                                                                        snr=args.snr, 
                                                                                        normal=args.normal,
                                                                                        slim=args.slim)

    config['models'] = args.models
    config['network'] = args.network
    config['dilation'] = args.dilation
    config['testonly'] = args.testonly
    config['n_class'] = len(classes)
    config['n_epochs'] = args.n_epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['res_dir'] = res_dir
    config['n_flattens'] = args.n_flattens
    config['n_hiddens'] = args.n_hiddens
    config['TEST_INTERVAL'] = args.TEST_INTERVAL
    config['VIS_INTERVAL'] = args.VIS_INTERVAL
    config['snr'] = args.snr
    config['normal'] = args.normal
    config['inception'] = args.inception
    config['aux_classifier'] = args.aux_classifier

    config['w_weight'] = args.w_weight
    config['w_gamma'] = args.w_gamma
    config['triplet_type'] = args.triplet_type
    config['t_weight'] = args.t_weight
    config['t_margin'] = args.t_margin
    config['t_confidence'] = args.t_confidence

    config['mmd_gamma'] = args.mmd_gamma
    config['random_layer'] = args.random_layer
    config['mcd_onestep'] = args.mcd_onestep
    config['mcd_vat'] = args.mcd_vat
    config['mcd_entropy'] = args.mcd_entropy
    config['mcd_swd'] = args.mcd_swd

    config['pada_cons_w'] = args.pada_cons_w
    config['slim'] = args.slim
    config['target_labeling'] = args.target_labeling

    train(config)
