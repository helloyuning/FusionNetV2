"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import math
import torch
import numpy as np
import os, sys
from utils.utils import AverageMeter
from utils.eval import calc_all_errs, Evaluation
from utils.img import im_norm_255
import cv2
import ref
from progress.bar import Bar
import os
import utils.fancy_logger as logger
from utils.tictoc import tic, toc
from builtins import input
from utils.fs import mkdir_p

from scipy.linalg import logm
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt
from numba import jit, njit

from ops.pnp.camera import PerspectiveCamera
from ops.pnp.cost_fun import AdaptiveHuberPnPCost
from ops.pnp.levenberg_marquardt import LMSolver
from ops.pnp.epropnp import EProPnP6DoF
from scipy.spatial.transform import Rotation as R
from utils.draw_orient_density import draw_orient_density
import torch.nn as nn
import time
from sobel_operator.sobel_edge import get_edge

# running_time = []


def test(epoch, cfg, data_loader, model, obj_vtx, obj_info, criterions, count=0):

    model.eval()
    Eval = Evaluation(cfg.dataset, obj_info, obj_vtx)
    if 'trans' in cfg.pytorch.task.lower():
        Eval_trans = Evaluation(cfg.dataset, obj_info, obj_vtx)

    if not cfg.test.ignore_cache_file:
        est_cache_file = cfg.test.cache_file
        # gt_cache_file = cfg.test.cache_file.replace('pose_est', 'pose_gt')
        gt_cache_file = cfg.test.cache_file.replace('_est', '_gt')
        if os.path.exists(est_cache_file) and os.path.exists(gt_cache_file):
            Eval.pose_est_all = np.load(est_cache_file, allow_pickle=True).tolist()
            Eval.pose_gt_all = np.load(gt_cache_file, allow_pickle=True).tolist()
            fig_save_path = os.path.join(cfg.pytorch.save_path, str(epoch))
            mkdir_p(fig_save_path)
            if 'all' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
                Eval.evaluate_pose_add(fig_save_path)
                Eval.evaluate_pose_arp_2d(fig_save_path)
            elif 'pose' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
            elif 'add' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_add(fig_save_path)
            elif 'arp' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_arp_2d(fig_save_path)
            else:
                raise Exception("Wrong test mode: {}".format(cfg.test.test_mode))

            return None, None

        else:
            logger.info("test cache file {} and {} not exist!".format(est_cache_file, gt_cache_file))
            userAns = input("Generating cache file from model [Y(y)/N(n)]:")
            if userAns.lower() == 'n':
                sys.exit(0)
            else:
                logger.info("Generating test cache file!")

    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)
    ##edge filtering
    edge_threshold = 0.5
    edge_filtering = True

    time_monitor = False#False
    vis_dir = os.path.join(cfg.pytorch.save_path, 'test_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    cam_intrinsic_np = cfg.dataset.camera_matrix.astype(np.float32)
    cam_intrinsic = torch.from_numpy(cam_intrinsic_np).cuda(cfg.pytorch.gpu)

    epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=3)).cuda(cfg.pytorch.gpu)

    for i, (obj, obj_id, inp, pose, c_box, s_box, box, trans_local) in enumerate(data_loader):

        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu, non_blocking=False).float()#修改了 async=True
            c_box = c_box.to(inp_var.device)
            s_box = s_box.to(inp_var.device)
            box = box.to(inp_var.device)
        else:
            inp_var = inp.float()

        bs = len(inp)

        # 我家的
        # edge_conf = np.zeros((bs, 6, 6))

        # forward propagation
        with torch.no_grad():
            # start = time.time()#我家的
            (noc, w2d, scale), pred_trans = model(inp_var)
            # noc:[16, 3, 64, 64]
            # print("model_time",time.time()- start)#我家的
            # running_time.append(df)#我家的


            ###我家的
            if edge_filtering:
                img = (inp_var[0].cpu().numpy().copy() * 255)[[2, 1, 0], :, :].astype(np.uint8)# [3,256,256]
                # print("img1", img.shape)
                img = img.transpose(1, 2, 0)[:, :, ::-1]# [256,256,3]
                # print("img1", img.shape)
                img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                edge_map = get_edge(img1)#原始edge_map
                edge_map = cv2.resize(edge_map, (64, 64))

            # print("三维edge 图大小:", b.shape)
            # if i % cfg.test.disp_interval == 0: #每50轮视觉化一次
            #     cv2.imwrite(
            #         os.path.join('/home/ivclab/path/EPro-PnP-main/EPro-PnP-6DoF/resized_edge/', '{}_edge_map.png'.format(i)), edge_map)
                # cv2.imwrite(
                #     os.path.join('/home/ivclab/path/EPro-PnP-main/EPro-PnP-6DoF/edge_mape/', '{}_inp.png'.format(i)), img)


            ##############
            w2d = w2d.flatten(2)

            # Due to a legacy design decision, we use an alternative to standard softmax, i.e., normalizing
            # the mean before exponential map.
            w2d = (w2d - w2d.mean(dim=-1, keepdim=True)
                   - math.log(w2d.size(-1))).exp().reshape(bs, 2, 64, 64) * scale[..., None, None]
            # To use standard softmax, comment out the two lines above and uncomment the line below:
            # w2d = w2d.softmax(dim=-1).reshape(bs, 2, 64, 64) * scale[..., None, None]
        ##可视化卷积


        if i % cfg.test.disp_interval == 0:
            # display input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[[2, 1, 0], :, :].astype(np.uint8)
            cfg.writer.add_image('input_image', inp_rgb, i)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                # display coordinates map
                pred_coor = noc[0].data.cpu().numpy().copy()#[3, 64, 64]
                # print("pred coor:",pred_coor.shape)
                pred_coor[0] = im_norm_255(pred_coor[0])
                pred_coor[1] = im_norm_255(pred_coor[1])
                pred_coor[2] = im_norm_255(pred_coor[2])
                pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                # print("输入坐标图大小：",)
                plt.imsave(os.path.join(vis_dir, '{}_coor_x_pred.png'.format(i)), pred_coor[0])
                plt.imsave(os.path.join(vis_dir, '{}_coor_y_pred.png'.format(i)), pred_coor[1])
                plt.imsave(os.path.join(vis_dir, '{}_coor_z_pred.png'.format(i)), pred_coor[2])
                plt.imsave(os.path.join(vis_dir, '{}_coor_xyz.png'.format(i)), pred_coor.transpose(1, 2, 0))
                # display confidence map
                pred_conf = w2d[0].mean(dim=0).data.cpu().numpy().copy()
                pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                cfg.writer.add_image('test_conf_pred', np.expand_dims(pred_conf, axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)

        dim = [[abs(obj_info[obj_id_]['min_x']),
                abs(obj_info[obj_id_]['min_y']),
                abs(obj_info[obj_id_]['min_z'])] for obj_id_ in obj_id.cpu().numpy()]
        dim = noc.new_tensor(dim)  # (n, 3)

        pose_gt = pose.cpu().numpy()



        # print("noc 的值:", noc)
        # print(" 的类型:", noc.shape)
        # print("w2d 形状:", w2d[0].mean(dim=0).shape)
        if 'rot' in cfg.pytorch.task.lower():

            #####################
            # 我家的edge filtering
            if edge_filtering:
                edge_map = edge_map[np.newaxis, :]
                pred_coor = noc[0].data.cpu().numpy().copy()
                col = list(
                    zip(obj, obj_id.cpu().numpy().copy(), pred_coor,edge_map))
                # print("存储的数据", obj_id.cpu().numpy().copy())
                # print("id 的类型:", type(obj_id.cpu().numpy().copy()))
                for idx in range(len(col)):
                    _, obj_id_, pred_coor_, edge_ = col[0]

                    select_pts_3d = []
                    pred_coor_ = noc[0].permute(1, 2, 0)
                    pred_coor_[:, :, 0] = pred_coor_[:, :, 0] * abs(obj_info[obj_id_]['min_x'])
                    pred_coor_[:, :, 1] = pred_coor_[:, :, 1] * abs(obj_info[obj_id_]['min_y'])
                    pred_coor_[:, :, 2] = pred_coor_[:, :, 2] * abs(obj_info[obj_id_]['min_z'])

                    pred_coor_ = pred_coor_.tolist()
                    # print("edge mape:", len(edge_map[np.newaxis, : ]), len(edge_map[np.newaxis, : ][0]))
                    edge_conf = edge_.tolist()

                    min_x = 0.001 * abs(obj_info[obj_id_]['min_x'])
                    min_y = 0.001 * abs(obj_info[obj_id_]['min_y'])
                    min_z = 0.001 * abs(obj_info[obj_id_]['min_z'])
                    for x in range(cfg.dataiter.out_res):
                        for y in range(cfg.dataiter.out_res):
                            if edge_conf[x][y] < edge_threshold:  #cfg.test.mask_threshold 阈值为0.5
                                continue
                            if abs(pred_coor_[x][y][0]) < min_x and abs(pred_coor_[x][y][1]) < min_y and \
                                    abs(pred_coor_[x][y][2]) < min_z:
                                continue
                            select_pts_3d.append(pred_coor_[x][y])
                    mat_pts_3d = np.array(select_pts_3d)
                    # print("mat 3d",len(mat_pts_3d),len(mat_pts_3d[0]))
                    # print("pred coord", len(pred_coor_),len(pred_coor_[0]))
                    # 还原三维坐标图，用0填充. 原来的三维特征图为64x64x3
                    pad_mat = np.pad(mat_pts_3d, ((0, 4096 - mat_pts_3d.shape[0]), (0, 0)), mode='constant', constant_values=0)
                    pad_mat = pad_mat.reshape((64, 64, 3))
                    # print("pad mat:", len(pad_mat), len(pad_mat[0]))
                    pad_mat = torch.from_numpy(pad_mat)
                    pad_mat = pad_mat.permute(2, 1, 0)
                    # print("pad_mat", pad_mat)
                    # print("pred coord", pred_coor_)
                    noc[0] = pad_mat  # [1, 3, 64, 64]


            # print("置换成功:", noc.shape)
            ########

            # building 2D-3D correspondences
            x3d = noc.permute(0, 2, 3, 1) * dim[:, None, None, :]
            w2d = w2d.permute(0, 2, 3, 1)  # (n, h, w, 2)

            s = s_box.to(torch.int64)  # (n, )
            wh_begin = c_box.to(torch.int64) - s[:, None] / 2.  # (n, 2)
            wh_unit = s.to(torch.float32) / cfg.dataiter.out_res  # (n, )

            wh_arange = torch.arange(cfg.dataiter.out_res, device=x3d.device, dtype=torch.float32)
            y, x = torch.meshgrid(wh_arange, wh_arange)  # (h, w)
            # (n, h, w, 2)
            x2d = torch.stack((wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
                               wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]), dim=-1)

        if 'trans' in cfg.pytorch.task.lower():
            # compute T from translation head
            ratio_delta_c = pred_trans[:, :2]  # (n, 2)
            ratio_depth = pred_trans[:, 2]  # (n, )
            pred_depth = ratio_depth * (cfg.dataiter.out_res / s_box)  # (n, )
            pred_c = ratio_delta_c * box[:, 2:] + c_box  # (n, 2)
            pred_x = (pred_c[:, 0] - cfg.dataset.camera_matrix[0, 2]) * pred_depth / cfg.dataset.camera_matrix[0, 0]
            pred_y = (pred_c[:, 1] - cfg.dataset.camera_matrix[1, 2]) * pred_depth / cfg.dataset.camera_matrix[1, 1]
            T_vector_trans = torch.stack([pred_x, pred_y, pred_depth], dim=-1)  # (n, 3)
            pose_est_trans = torch.cat((torch.eye(3, device=pred_x.device).expand(bs, -1, -1),
                                        T_vector_trans.reshape(bs, 3, 1)), dim=-1).cpu().numpy()  # (n, 3, 4)



        if 'rot' in cfg.pytorch.task.lower():
            #############################
            # pred_coor_ = noc[0].data.cpu().numpy().copy()
            # print("坐标图:", pred_coor_[:, :, 0])
            # pred_coor_[:, :, 0] = pred_coor_[:, :, 0] * abs(obj_info[obj_id]['min_x'])
            # pred_coor_[:, :, 1] = pred_coor_[:, :, 1] * abs(obj_info[obj_id]['min_y'])
            # pred_coor_[:, :, 2] = pred_coor_[:, :, 2] * abs(obj_info[obj_id]['min_z'])
            # pred_coor_ = pred_coor_.tolist()
            # print("转换成功")


            # edge_conf = edge_map.tolist()
            # for x in range(cfg.dataiter.out_res):
            #     for y in range(cfg.dataiter.out_res):
            #         if edge_conf[x][y] < cfg.test.mask_threshold:
            #             continue
                    # if abs(pred_coor_[x][y][0]) < min_x and abs(pred_coor_[x][y][1]) < min_y and \
                    #         abs(pred_coor_[x][y][2]) < min_z:
                    #     continue
                    # print("挑选成功")

            ########################

            dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no lens distortion

            # for fair comparison we use EPnP initialization
            pred_conf_np = w2d.mean(dim=-1).cpu().numpy()  # (n, h, w)
            binary_mask = pred_conf_np >= np.quantile(pred_conf_np.reshape(bs, -1), 0.8,
                                                      axis=1, keepdims=True)[..., None]
            R_quats = []
            T_vectors = []
            x2d_np = x2d.cpu().numpy() #[16, 64, 64, 2]
            x3d_np = x3d.cpu().numpy() #[16, 64, 64, 3]

            for x2d_np_, x3d_np_, mask_np_ in zip(x2d_np, x3d_np, binary_mask):
                _, R_vector, T_vector = cv2.solvePnP(
                    x3d_np_[mask_np_], x2d_np_[mask_np_], cam_intrinsic_np, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
                q = R.from_rotvec(R_vector.reshape(-1)).as_quat()[[3, 0, 1, 2]]
                R_quats.append(q)
                T_vectors.append(T_vector.reshape(-1))
            R_quats = x2d.new_tensor(R_quats)
            T_vectors = x2d.new_tensor(T_vectors)
            pose_init = torch.cat((T_vectors, R_quats), dim=-1)  # (n, 7)

            # Gauss-Newton optimize
            x2d = x2d.reshape(bs, -1, 2)
            w2d = w2d.reshape(bs, -1, 2)
            x3d = x3d.reshape(bs, -1, 3)
            camera = PerspectiveCamera(
                cam_mats=cam_intrinsic[None].expand(bs, -1, -1), z_min=0.01)
            cost_fun = AdaptiveHuberPnPCost(
                relative_delta=0.1)

            if time_monitor:
                torch.cuda.synchronize(device=x3d.device)
                t_begin = time.time()

            cost_fun.set_param(x2d, w2d)
            pose_opt = epropnp(
                x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, fast_mode=True)[0]

            if time_monitor:
                torch.cuda.synchronize(device=x3d.device)
                t_end = time.time()
                logger.info("Batch PnP time: {:04f}".format(t_end - t_begin))

            if i % cfg.test.disp_interval == 0:
                _, _, _, pose_samples, pose_sample_logweights, _ = epropnp.monte_carlo_forward(
                    x3d, x2d, w2d, camera, cost_fun,
                    pose_init=pose_opt, force_init_solve=False, fast_mode=True)
                draw = draw_orient_density(
                    pose_opt[:1], pose_samples[:, :1], pose_sample_logweights[:, :1]).squeeze(0)  # (h, w, 3)
                plt.imsave(os.path.join(vis_dir, '{}_orient_distr.png'.format(i)),
                           (draw * 255).clip(min=0, max=255).astype(np.uint8))

            T_vectors, R_quats = pose_opt.split([3, 4], dim=-1)  # (n, [3, 4])
            R_matrix = R.from_quat(R_quats[:, [1, 2, 3, 0]].cpu().numpy()).as_matrix()  # (n, 3, 3)
            pose_est = np.concatenate([R_matrix, T_vectors.reshape(bs, 3, 1).cpu().numpy()], axis=-1)

            if 'trans' in cfg.pytorch.task.lower():
                pose_est_trans = np.concatenate((R_matrix, T_vector_trans.reshape(bs, 3, 1)), axis=-1)
            ##########我家的，制做refine数据
            #print("obj:", obj[0])
            # print("gt_pose:",pose_gt)
            # print("gt_est:", pose_est)
            # ad = '/home/ivclab/path/my_data/'
            # ct = time.ctime()
            # pcnt = 0
            # for obj_, pose_est_, pose_gt_ in zip(obj, pose_est, pose_gt):
            #     np.savetxt(ad + str(obj[0])+"/gt_pose/" + str(pcnt+count) + '.txt',pose_gt_)
            #     np.savetxt(ad + str(obj[0]) + "/pred_pose/" + str(pcnt+count) + '.txt', pose_est_)
            #     pcnt += 1
            #     if pcnt >= 31:
            #         count = pcnt + count
            # count += 1

            ###############

            #原来的
            for obj_, pose_est_, pose_gt_ in zip(obj, pose_est, pose_gt):
                Eval.pose_est_all[obj_].append(pose_est_)
                Eval.pose_gt_all[obj_].append(pose_gt_)
                Eval.num[obj_] += 1
                Eval.numAll += 1

        if 'trans' in cfg.pytorch.task.lower():
            for obj_, pose_est_trans_, pose_gt_ in zip(obj, pose_est_trans, pose_gt):
                Eval_trans.pose_est_all[obj_].append(pose_est_trans_)
                Eval_trans.pose_gt_all[obj_].append(pose_gt_)
                Eval_trans.num[obj_] += 1
                Eval_trans.numAll += 1

        Bar.suffix = 'test Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | Loss_trans {loss_trans.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans)
        bar.next()

    epoch_save_path = os.path.join(cfg.pytorch.save_path, str(epoch))
    if not os.path.exists(epoch_save_path):
        os.makedirs(epoch_save_path)
    if 'rot' in cfg.pytorch.task.lower():
        logger.info("{} Evaluate of Rotation Branch of Epoch {} {}".format('-'*40, epoch, '-'*40))
        preds['poseGT'] = Eval.pose_gt_all
        preds['poseEst'] = Eval.pose_est_all
        if cfg.pytorch.test:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_test.npy'), Eval.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_test.npy'), Eval.pose_gt_all)
        else:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_epoch{}.npy'.format(epoch)), Eval.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_epoch{}.npy'.format(epoch)), Eval.pose_gt_all)
        # evaluation
        if 'all' in cfg.test.test_mode.lower():
            Eval.evaluate_pose()
            Eval.evaluate_pose_add(epoch_save_path)
            Eval.evaluate_pose_arp_2d(epoch_save_path)
        else:
            if 'pose' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
            if 'add' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_add(epoch_save_path)
            if 'arp' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_arp_2d(epoch_save_path)

    if 'trans' in cfg.pytorch.task.lower():
        logger.info("{} Evaluate of Translation Branch of Epoch {} {}".format('-'*40, epoch, '-'*40))
        preds['poseGT'] = Eval_trans.pose_gt_all
        preds['poseEst'] = Eval_trans.pose_est_all
        if cfg.pytorch.test:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_test_trans.npy'), Eval_trans.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_test_trans.npy'), Eval_trans.pose_gt_all)
        else:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_trans_epoch{}.npy'.format(epoch)), Eval_trans.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_trans_epoch{}.npy'.format(epoch)), Eval_trans.pose_gt_all)
        # evaluation
        if 'all' in cfg.test.test_mode.lower():
            Eval_trans.evaluate_pose()
            Eval_trans.evaluate_pose_add(epoch_save_path)
            Eval_trans.evaluate_pose_arp_2d(epoch_save_path)

        else:
            if 'pose' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose()
            if 'add' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose_add(epoch_save_path)
            if 'arp' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose_arp_2d(epoch_save_path)

    bar.finish()
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg}, preds#, np.mean(running_time)

