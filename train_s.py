#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-identifier]
"""

import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from docopt import docopt
import numpy as np
import torch
import yaml
import scipy.io as sio


import ht_lcnn.lcnn
from ht_lcnn.lcnn.models import hg
from ht_lcnn.lcnn.datasets import WireframeDataset, collate
from ht_lcnn.lcnn.config import C, M
from ht_lcnn.lcnn.models.line_vectorizer import LineVectorizer
from ht_lcnn.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from ht_lcnn.lcnn.models.HT import hough_transform
from ht_lcnn.lcnn.trainer_s import Trainer

from ht_lcnn.lcnn.postprocess import postprocess
from ht_lcnn.lcnn.utils import recursive_to

from CONSAC.utils.misc import *
from CONSAC.models.cn_net import CNNet
from CONSAC.utils import sampling
from CONSAC.utils.em_algorithm import  em_for_vp
from CONSAC.utils.evaluation import calc_labels

data_dim = 9
model_dim = 3
minimal_set_size = 2
outerhyps = 2
hyps = 2
instances = 3
samplecount = 4
threshold = 0.001
min_prob = 1e-8
max_num_data = 256
loss_clamp = 0.3
#max_prob_loss = 0
unconditional = False
selfsupervised = False
max_prob_loss_only = False
inlier_fun = sampling.soft_inlier_fun_gen(5. / threshold, threshold)


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    return outdir


def main():
    """args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"""
    config_file = "config/wireframe.yaml"
    #config_file = "config/wireframe3D.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")

    device = torch.device(device_name)

    # 1. dataset

    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(datadir, split="train")[0]
    # sys.exit(0)

    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,
        "pin_memory": False,
    }
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="train"),
        shuffle=True,
        batch_size=M.batch_size,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )
    epoch_size = len(train_loader)
    # print("epoch_size (train):", epoch_size)
    # print("epoch_size (valid):", len(val_loader))

    if resume_from:
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth"))
    else:
        checkpoint = torch.load("./trained_model/ht_lcnn_model.pth.tar")
        print("use pre-trainend ht-lcnn model")

    # 2. model
    ### load vote_index matrix for Hough transform
    ### defualt settings: (128, 128, 3, 1)
    if os.path.isfile(C.io.vote_index):
        print('load vote_index ... ')
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        print('compute vote_index ... ')
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('vote_index loaded', vote_index.shape)

    if M.backbone == "stacked_hourglass":
        model = hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
            vote_index=vote_index,
        )
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    #print("model:", model)
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', train_params)

    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])
    #else:
        #model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    # CONSAC part

    model_dim = 3
    ddim = 9
    batch_norm = True
    model_gen_fun = sampling.vp_from_lines_batched
    consistency_fun = sampling.vp_consistency_measure_angle_batched
    min_set_size = 2
    em_fun = em_for_vp
    if resume_from:
        ckpt = './trained_model/consac_vp.net'
    threshold = 0.001
    threshold2 = 0.001
    hyps = 32
    outerhyps = 32
    instances = 3
    em = 10

    CONSAC_net = CNNet(6, ddim, batch_norm=batch_norm)
    CONSAC_net = CONSAC_net.to(device)

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            [
                {'params': model.parameters()},
                {'params': CONSAC_net.parameters(), 'lr': 1e-4}
            ],
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            [
                {'params': model.parameters()},
                {'params': CONSAC_net.parameters()}
            ],
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = C.io.outdir #or get_outdir(args["--identifier"])
    print("outdir:", outdir)


    try:
        trainer = Trainer(
            device=device,
            model=model,
            CONSAC_net=CONSAC_net,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
