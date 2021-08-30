import atexit
import os
import os.path as osp
import shutil
import signal
import subprocess
import threading
import time
from timeit import default_timer as timer
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

import ht_lcnn.lcnn
from ht_lcnn.lcnn.models import hg
from ht_lcnn.lcnn.config import C, M
from ht_lcnn.lcnn.models.line_vectorizer import LineVectorizer
from ht_lcnn.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from ht_lcnn.lcnn.models.HT import hough_transform

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
selfsupervised = True
max_prob_loss_only = False
inlier_fun = sampling.soft_inlier_fun_gen(5. / threshold, threshold)


class Trainer(object):
    def __init__(self, device, model,CONSAC_net, optimizer, train_loader, val_loader, out):
        self.device = device

        self.model = model
        self.CONSAC_net = CONSAC_net
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        '''
        ## unable to run tensorboard on the cluster
        self.run_tensorboard()
        time.sleep(1)
        self.board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(self.board_out):
            os.makedirs(self.board_out)
        self.writer = SummaryWriter(self.board_out)
        '''

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000
        self.CONSAC_loss = self.best_CONSAC_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

        self.max_prob_loss = 0

    def run_tensorboard(self):
        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        p = subprocess.Popen(
            ["tensorboard", f"--logdir={board_out}", f"--port={C.io.tensorboard_port}"]
        )
        def killme():
            os.kill(p.pid, signal.SIGTERM)
        atexit.register(killme)

    def _loss(self, result):
        losses = result["losses"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["CONSAC"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()
        self.CONSAC_net.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * M.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * M.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }
                result = self.model(input_dict)

                total_loss += self._loss(result)

                H = result["preds"]
                lines = H["lines"]
                # scores = H["score"]


                num_actual_line_segments = []
                for i in range(lines.shape[0]):
                    flag = True
                    for j in range(1, len(lines[i])):
                        if (lines[i, j, :, :] == lines[i, 0, :, :]).all():
                            num_actual_line_segments.append(j + 1)
                            flag = False
                            break
                    if flag:
                        num_actual_line_segments.append(j + 1)

                line_segments_scaled = lines.clone()
                height = 128
                width = 128
                line_segments_scaled[:, :, :, 0] -= height / 2
                line_segments_scaled[:, :, :, 1] -= width / 2
                line_segments_scaled[:, :, :, 0] /= height / 2
                line_segments_scaled[:, :, :, 1] /= width / 2

                line_segments_scaled_h = []
                for i in range(lines.shape[0]):
                    nlines = []
                    for li in range(lines.shape[1]):
                        p1 = torch.cat((line_segments_scaled[i, li, 0],torch.tensor([1]).cuda()),0)
                        p2 = torch.cat((line_segments_scaled[i, li, 1],torch.tensor([1]).cuda()),0)
                        centroid = 0.5 * (p1 + p2)
                        line = torch.cross(p1, p2)
                        tn = torch.norm(line[0:2])
                        line /= torch.norm(line[0:2])
                        line_segment = torch.cat([p1, p2, line, centroid, tn.reshape(-1)],0)
                        nlines.append(line_segment.reshape(1,-1))
                    nlines = torch.cat(nlines,0)
                    valid_line = nlines[nlines[:,12] > 0]
                    nlines[nlines[:,12] == 0] = valid_line[np.random.randint(valid_line.shape[0])]
                    line_segments_scaled_h.append(nlines[:, 0:12].reshape(1,-1,12))
                
                data = torch.cat(line_segments_scaled_h,0)
                num_data = num_actual_line_segments
                num_models = 3
                CONSAC_loss = []

                data_and_state = torch.zeros((outerhyps, instances, data.size(0), data.size(1), data_dim),
                                             device=self.device)
                all_grads = torch.zeros((outerhyps, hyps, instances, data.size(0), data.size(1)),
                                        device=self.device)
                all_inliers = torch.zeros((outerhyps, hyps, instances, data.size(0), data.size(1)),
                                          device=self.device)
                all_best_inliers = torch.zeros((outerhyps, instances, data.size(0), data.size(1)),
                                               device=self.device)
                all_models = torch.zeros((outerhyps, hyps, instances, data.size(0), model_dim),
                                         device=self.device)
                best_models = torch.zeros((outerhyps, instances, data.size(0), model_dim), device=self.device)
                all_best_hypos = torch.zeros((outerhyps, instances,), device=self.device, dtype=torch.int)
                all_log_probs = torch.zeros((hyps, instances, data.size(0), data.size(1)), device=self.device)
                all_entropies = torch.zeros((outerhyps, instances, data.size(0)), device=self.device)

                for oh in range(outerhyps):

                    for mi in range(instances):

                        data_and_state[oh, mi, :, :, 0:(data_dim - 1)] = data[:, :, 0:(data_dim - 1)]
                        log_probs = self.CONSAC_net(data_and_state[oh, mi])

                        for bi in range(0, data.size(0)):

                            log_prob_grads = []
                            losses = []

                            cur_probs = torch.softmax(log_probs[bi, :, 0:num_data[bi]].squeeze(), dim=-1)

                            models, _, choices, distances = sampling.sample_model_pool(
                                data[bi], num_data[bi], hyps, minimal_set_size, inlier_fun,
                                sampling.vp_from_lines, sampling.vp_consistency_measure_angle, cur_probs, device=self.device)

                            all_grads[oh, :, mi, bi] = choices
                            inliers = sampling.soft_inlier_fun(distances, 5. / threshold, threshold)

                            all_inliers[oh, :, mi, bi] = inliers
                            all_models[oh, :, mi, bi, :] = models

                            inlier_counts = torch.sum(inliers, dim=-1)
                            best_hypo = torch.argmax(inlier_counts)
                            best_inliers = inliers[best_hypo]
                            all_best_hypos[oh, mi] = best_hypo
                            all_best_inliers[oh, mi, bi] = best_inliers

                            best_models[oh, mi, bi] = models[best_hypo]

                            entropy = torch.distributions.categorical.Categorical(
                                probs=cur_probs).entropy()
                            all_entropies[oh, mi, bi] = entropy

                            if not unconditional and mi + 1 < instances:
                                data_and_state[oh, mi + 1, bi, :, data_dim - 1] = torch.max(
                                    data_and_state[oh, mi, bi, :, data_dim - 1], best_inliers)

                exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
                inlier_counts = torch.sum(exclusive_inliers, dim=-1)
                best_hypo = torch.argmax(inlier_counts, dim=0)

                sampled_models = torch.zeros((data.size(0), instances, model_dim), device=self.device)

                for bi in range(0, data.size(0)):
                    sampled_models[bi] = best_models[best_hypo[bi], :, bi]

                if selfsupervised:
                    for bi in range(0, data.size(0)):
                        inlier_count = 0
                        for mi in range(instances):
                            exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[bi], 0:mi + 1, bi], dim=0)
                            inlier_count += torch.sum(exclusive_inliers, dim=-1)

                        loss = -(inlier_count * 1. / max_num_data * 1. / instances)
                        CONSAC_loss += [loss.cpu().numpy()]
                else:
                    '''for bi in range(0, data.size(0)):
                        models = sampled_models[bi]
                        tp_models = torch.transpose(models[:num_models], 0, 1)
                        cost_matrix = 1 - torch.matmul(gt_models[bi, :num_models, 0:3], tp_models[0:3]).abs()

                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                        loss = cost_matrix[row_ind, col_ind].sum().detach()
                        CONSAC_loss += [loss.cpu().numpy()]'''
                    print("no ground truth vp")
                closs = np.mean(CONSAC_loss)

        pprint(
            f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
            + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
            + f"| {closs:04.1f} "
        )
        #self._write_metrics(len(self.val_loader), total_loss, "validation", False)
        self.mean_loss = total_loss / len(self.val_loader)
        self.CONSAC_loss = closs

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "CONSAC_model_state_dict": self.CONSAC_net.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best_net.pth"),
            )

        if self.CONSAC_loss < self.best_CONSAC_loss:
            self.best_CONSAC_loss = self.CONSAC_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best_CONSAC.pth"),
            )


        if training:
            self.model.train()
        del input_dict, result
        torch.cuda.empty_cache()


    def train_epoch(self):
        self.model.train()
        self.CONSAC_net.train()

        time = timer()
        avg_losses_vp_epoch = []
        avg_per_model_vp_losses_epoch = [[] for _ in range(3)]

        # 正向传播时：开启自动求导的异常侦测
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (image, meta, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0
            

            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "mode": "training",
            }
            result = self.model(input_dict)
            net_loss = self._loss(result)
            if np.isnan(net_loss.item()):
                raise ValueError("loss is nan while training")
            #net_loss.backward()

            H = result["preds"]
            lines = H["lines"]
            #scores = H["score"]

            print(lines.shape)
            
            num_actual_line_segments = []
            for i in range(lines.shape[0]):
                flag = True
                for j in range(1, len(lines[i])):
                    if (lines[i,j,:,:] == lines[i,0,:,:]).all():
                        num_actual_line_segments.append(j + 1)
                        flag = False
                        break
                if flag:
                    num_actual_line_segments.append(j + 1)

            line_segments_scaled = lines.clone()
            height = 128
            width = 128
            line_segments_scaled[:, :, :, 0] -= height / 2
            line_segments_scaled[:, :, :, 1] -= width / 2
            line_segments_scaled[:, :, :, 0] /= height / 2
            line_segments_scaled[:, :, :, 1] /= width / 2

            line_segments_scaled_h = []
            for i in range(lines.shape[0]):
                nlines = []
                for li in range(lines.shape[1]):
                    p1 = torch.cat((line_segments_scaled[i, li, 0],torch.tensor([1]).cuda()),0)
                    p2 = torch.cat((line_segments_scaled[i, li, 1],torch.tensor([1]).cuda()),0)
                    centroid = 0.5 * (p1 + p2)
                    line = torch.cross(p1, p2)
                    tn = torch.norm(line[0:2])
                    if tn != 0:
                        line_norm = line/ torch.norm(line[0:2])
                        line_segment = torch.cat([p1, p2, line_norm, centroid, tn.reshape(-1)],0)
                        nlines.append(line_segment.reshape(1,-1))
                num_of_lines_ = len(nlines)
                if num_of_lines_ <2500:
                    for i in range(2500-num_of_lines_):
                        nlines.append(nlines[i])
                line_segments_scaled_h.append(torch.cat(nlines,0)[:, 0:12].reshape(1,-1,12))

            data = torch.cat(line_segments_scaled_h,0)
            num_data = num_actual_line_segments
            num_models = 3
            data_and_state = torch.zeros(
                (outerhyps, instances, samplecount, data.size(0), data.size(1), data_dim), device=self.device)
            all_grads = torch.zeros(
                (outerhyps, instances, samplecount, data.size(0), data.size(1)), device=self.device)
            all_inliers = torch.zeros(
                (outerhyps, hyps, instances, samplecount, data.size(0), data.size(1)), device=self.device)
            all_best_inliers = torch.zeros(
                (outerhyps, instances, samplecount, data.size(0), data.size(1)), device=self.device)
            all_models = torch.zeros(
                (outerhyps, hyps, instances, samplecount, data.size(0), model_dim), device=self.device)
            best_models = torch.zeros((outerhyps, instances, samplecount, data.size(0), model_dim),
                                      device=self.device)
            all_best_hypos = torch.zeros((outerhyps, instances, samplecount, data.size(0)), device=self.device)
            all_log_probs = torch.zeros((hyps, instances, samplecount, data.size(0), data.size(1)),
                                        device=self.device)
            all_entropies = torch.zeros((outerhyps, instances, samplecount, data.size(0),), device=self.device)
            all_losses = torch.zeros((samplecount, data.size(0)), device=self.device)
            all_losses_per_model = torch.zeros((samplecount, data.size(0), instances), device=self.device)
            all_max_probs = torch.ones((outerhyps, instances, samplecount, data.size(0), data.size(1)),
                                       device=self.device)
            all_joint_inliers = torch.zeros(
                (outerhyps, hyps, instances, samplecount, data.size(0), data.size(1)), device=self.device)

            self.model.eval()
            self.CONSAC_net.eval()

            neg_inliers = torch.ones((samplecount, outerhyps, instances + 1, data.size(0), data.size(1)),
                                     device=self.device)
            for mi in range(instances):

                for oh in range(outerhyps):
                    for si in range(0, samplecount):

                        data_and_state[oh, mi, si, :, :, 0:(data_dim - 1)] = data[:, :, 0:(data_dim - 1)]

                data_and_state_batched = data_and_state[:, mi].contiguous().view((-1, data.size(1), data_dim))
                log_probs_batched = self.CONSAC_net(data_and_state_batched)
                log_probs = log_probs_batched.view((outerhyps, samplecount, data.size(0), data.size(1)))
                probs = torch.exp(log_probs)

                for oh in range(outerhyps):
                    for bi in range(0, data.size(0)):

                        all_max_probs[oh, mi, :, bi, :] = neg_inliers[:, oh, mi, bi, :]

                        for si in range(0, samplecount):

                            cur_probs = probs[oh, si, bi, 0:num_data[bi]]

                            entropy = torch.distributions.categorical.Categorical(
                                probs=cur_probs).entropy()
                            all_entropies[oh, mi, si, bi] = entropy

                            models, _, choices, distances = \
                                sampling.sample_model_pool(data[bi], num_data[bi], hyps, minimal_set_size,
                                                           inlier_fun, sampling.vp_from_lines,
                                                           sampling.vp_consistency_measure_angle, cur_probs,
                                                           device=self.device, min_prob=min_prob)

                            all_grads[oh, mi, si, bi] = choices.sum(0)

                            inliers = sampling.soft_inlier_fun(distances, 5. / threshold, threshold)

                            all_inliers[oh, :, mi, si, bi] = inliers
                            all_models[oh, :, mi, si, bi, :] = models

                            if mi > 0:
                                all_joint_inliers[oh, :, mi, si, bi] = torch.max(inliers, all_best_inliers[
                                    oh, mi - 1, si, bi].unsqueeze(0).expand(hyps, -1))
                            else:
                                all_joint_inliers[oh, :, mi, si, bi] = inliers

                            inlier_counts = torch.sum(inliers, dim=-1)
                            best_hypo = torch.argmax(inlier_counts)
                            best_inliers = inliers[best_hypo]
                            all_best_hypos[oh, mi, si, bi] = best_hypo
                            all_best_inliers[oh, mi, si, bi] = best_inliers

                            best_joint_inliers = all_joint_inliers[oh, best_hypo, mi, si, bi]
                            neg_inliers[si, oh, mi + 1, bi, :] = 1 - best_joint_inliers

                            best_models[oh, mi, si, bi] = models[best_hypo]

                            if not unconditional and mi + 1 < instances:
                                data_and_state[oh, mi + 1, si, bi, :, (data_dim - 1)] = torch.max(
                                    data_and_state[oh, mi, si, bi, :, (data_dim - 1)], best_inliers)

            exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
            inlier_counts = torch.sum(exclusive_inliers, dim=-1)
            best_hypo = torch.argmax(inlier_counts, dim=0)

            sampled_models = torch.zeros((samplecount, data.size(0), instances, 3), device=self.device)

            for si in range(0, samplecount):
                for bi in range(0, data.size(0)):
                    sampled_models[si, bi] = best_models[best_hypo[si, bi], :, si, bi]

            if selfsupervised:
                for bi in range(0, data.size(0)):
                    for si in range(0, samplecount):

                        inlier_count = 0
                        last_inlier_count = 0
                        for mi in range(instances):
                            exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[si, bi], 0:mi + 1, si, bi],
                                                             dim=0)
                            current_inlier_count = torch.sum(exclusive_inliers, dim=-1)
                            inlier_count += current_inlier_count
                            inlier_increase = current_inlier_count - last_inlier_count
                            all_losses_per_model[si, bi, mi] = -(inlier_increase * 1. / max_num_data)
                            last_inlier_count = current_inlier_count

                        loss = -(inlier_count * 1. / max_num_data * 1. / instances)
                        all_losses[si, bi] = loss

            else:
                """for bi in range(0, data.size(0)):
                    for si in range(0, samplecount):

                        models = sampled_models[si, bi]

                        gt_tp_models = torch.transpose(gt_models[bi, : num_models], 0, 1)

                        tp_models_np = models.detach().cpu().numpy()
                        if False in np.isfinite(tp_models_np):
                            print(tp_models_np)

                        cost_matrix = 1 - torch.matmul(models[:, 0:3], gt_tp_models[0:3]).abs()

                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                        loss = cost_matrix[row_ind, col_ind].sum().detach()
                        all_losses[si, bi] = loss
                        for mi in range(num_models):
                            all_losses_per_model[si, bi, mi] = cost_matrix[mi, col_ind[mi]]"""
                print("no ground truth vp")

            baselines = all_losses.mean(dim=0)
            avg_per_model_losses = all_losses_per_model.mean(dim=0).mean(dim=0)

            for bi in range(0, data.size(0)):
                baseline = baselines[bi]
                for s in range(0, samplecount):
                    all_grads[:, :, s, bi, :] *= (all_losses[s, bi] - baseline)

            self.model.train()
            self.CONSAC_net.train()

            data_and_state_batched = data_and_state.view((-1, data.size(1), data_dim))
            log_probs_batched = self.CONSAC_net(data_and_state_batched)
            grads_batched = all_grads.view((-1, 1, data.size(1), 1))

            if loss_clamp > 0:
                grads_batched = torch.clamp(grads_batched, max=loss_clamp, min=-loss_clamp)

            mean_entropy = torch.mean(all_entropies)
            # 反向传播时：在求导时开启侦测
            with torch.autograd.detect_anomaly(): #enable for debugging
            #if True:
                if self.max_prob_loss > 0:
                    log_probs = log_probs_batched.view(outerhyps, instances, samplecount, data.size(0),
                                                   data.size(1))
                    probs = torch.softmax(log_probs, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
                    probs = probs / torch.clamp(max_probs, min=1e-8)
                    max_prob_loss = torch.clamp(probs - all_max_probs, min=0)
                    max_prob_grad = max_prob_loss * torch.ones_like(max_prob_loss, device=self.device)
                    if max_prob_loss_only:
                        torch.autograd.backward((max_prob_loss), (max_prob_grad))
                    else:
                        torch.autograd.backward((log_probs_batched, max_prob_loss), (grads_batched, max_prob_grad))

                    avg_max_prob_loss = torch.sum(max_prob_loss)
                else:
                    torch.autograd.backward((log_probs_batched), (grads_batched))

            self.optim.step()

            avg_loss = all_losses.mean()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            #self._write_metrics(1, net_loss.item(), "training", do_print=False)


            if self.iteration % 10 == 0:
                pprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {avg_loss:04.1f} "
                )
                time = timer()
            if self.iteration % 500 == 0:
                checkpoint = {
                "iteration": self.iteration,
                "optim_state_dict": self.optim.state_dict(),
                "model": self.model.state_dict(),
                "CONSAC_model": self.CONSAC_net.state_dict()
                }
                if not os.path.isdir("./checkpoints"):
                    os.mkdir("./checkpoints")
                torch.save(checkpoint,"./checkpoints/ckpt_%s.pth"%(str(self.iteration)))
            del input_dict, result, net_loss
            torch.cuda.empty_cache()
            num_images = self.batch_size * self.iteration
            if num_images % self.validation_interval == 0 or num_images == 600:
                self.validate()
                time = timer()
            
            

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )
        return total_loss

    def _plot_samples(self, i, index, result, meta, target, prefix):
        fn = self.val_loader.dataset.filelist[index][:-10].replace("_a0", "") + ".png"
        img = io.imread(fn)
        imshow(img), plt.savefig(f"{prefix}_img.jpg"), plt.close()

        mask_result = result["jmap"][i].cpu().numpy()
        mask_target = target["jmap"][i].cpu().numpy()
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia), plt.savefig(f"{prefix}_mask_{ch}a.jpg"), plt.close()
            imshow(ib), plt.savefig(f"{prefix}_mask_{ch}b.jpg"), plt.close()

        line_result = result["lmap"][i].cpu().numpy()
        line_target = target["lmap"][i].cpu().numpy()
        imshow(line_target), plt.savefig(f"{prefix}_line_a.jpg"), plt.close()
        imshow(line_result), plt.savefig(f"{prefix}_line_b.jpg"), plt.close()

        def draw_vecl(lines, sline, juncs, junts, fn):
            imshow(img)
            if len(lines) > 0 and not (lines[0] == 0).all():
                for i, ((a, b), s) in enumerate(zip(lines, sline)):
                    if i > 0 and (lines[i] == lines[0]).all():
                        break
                    plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
            if not (juncs[0] == 0).all():
                for i, j in enumerate(juncs):
                    if i > 0 and (i == juncs[0]).all():
                        break
                    plt.scatter(j[1], j[0], c="red", s=64, zorder=100)
            if junts is not None and len(junts) > 0 and not (junts[0] == 0).all():
                for i, j in enumerate(junts):
                    if i > 0 and (i == junts[0]).all():
                        break
                    plt.scatter(j[1], j[0], c="blue", s=64, zorder=100)
            plt.savefig(fn), plt.close()

        junc = meta[i]["junc"].cpu().numpy() * 4
        jtyp = meta[i]["jtyp"].cpu().numpy()
        juncs = junc[jtyp == 0]
        junts = junc[jtyp == 1]
        rjuncs = result["juncs"][i].cpu().numpy() * 4
        rjunts = None
        if "junts" in result:
            rjunts = result["junts"][i].cpu().numpy() * 4

        lpre = meta[i]["lpre"].cpu().numpy() * 4
        vecl_target = meta[i]["lpre_label"].cpu().numpy()
        vecl_result = result["lines"][i].cpu().numpy() * 4
        score = result["score"][i].cpu().numpy()
        lpre = lpre[vecl_target == 1]

        draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts, f"{prefix}_vecl_a.jpg")
        draw_vecl(vecl_result, score, rjuncs, rjunts, f"{prefix}_vecl_b.jpg")

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        torch.backends.cudnn.enabled = False
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()
            self.validate()



cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(sm, fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


# def _launch_tensorboard(board_out, port, out):
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])
#
#     def kill():
#         os.kill(p.pid, signal.SIGTERM)
#
#     atexit.register(kill)
