import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SC_MSF.utils.visualization import Visualizer
from SC_MSF.utils.box_utils import cxcywh_to_x1y1x2y2
from SC_MSF.utils.dataset_utils import restore
from SC_MSF.modeling.gmm2d import GMM2D
from SC_MSF.modeling.gmm4d import GMM4D
from .evaluate import evaluate_multimodal, compute_kde_nll
from .utils import print_info, viz_results, post_process

from tqdm import tqdm
import pickle as pkl
import pdb


def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    max_iters = len(dataloader)
    obd_speed_max = 0
    heading_angle_max = 0
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            # print(iters,batch)
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            image_file = batch['image_file']
            resolution = batch['pred_resolution'].numpy()
            # obs_pid =  batch['obs_pid']
            pose = batch['pose']
            # print(image_file,pose)
            pose = batch['pose']
            cur_pose = batch['cur_pose']
            str_pose = batch['str_pose']
            key_pose = batch['key_pose']
            image_features = batch['image_features']
            cur_img = image_features[:, -1]
            # print(image_features.shape,cur_img.shape)
            # print(cfg.DATASET.NAME)
            if cfg.DATASET.NAME == 'PIE':
                obd_speed = batch['obd_speed'] / 54
                # print(obd_speed)
                heading_angle = batch['heading_angle'] / 360
            else:
                obd_speed = batch['obd_speed'] / 5
                heading_angle = None

            # NOTE: These optical-flow fields exist for PIE_.py dataloader only.
            # PIE.py (pkl-only) and JAAD dataloader do not provide ped/ego flow fields.
            if cfg.DATASET.NAME == 'PIE' and 'obs_ped_flow' in batch:
                obs_ped_flow = batch['obs_ped_flow'].to(device)
                obs_ped_flow = torch.mean(obs_ped_flow, dim=-2)
                obs_cam_flow = batch['obs_ego_flow'].to(device)
                obs_cam_flow = torch.mean(obs_cam_flow, dim=-2)
                pred_ped_flow = batch['pred_ped_flow'].to(device)
                pred_ped_flow = torch.mean(pred_ped_flow, dim=-2)
                pred_cam_flow = batch['pred_ego_flow'].to(device)
                pred_cam_flow = torch.mean(pred_cam_flow, dim=-2)
            else:
                obs_ped_flow = None
                obs_cam_flow = None
                pred_ped_flow = None
                pred_cam_flow = None
            # print(obd_speed,heading_angle)
            ########################################################确定车速和角度极值
            # if obd_speed_max < torch.max(obd_speed):
            #   obd_speed_max = torch.max(obd_speed)
            #   print(' obd_speed_max ', obd_speed_max )
            # if heading_angle_max  < torch.max(heading_angle):
            #     heading_angle_max = torch.max(heading_angle )
            #     print(' oheading_angle_max  ', heading_angle_max )
            ############################################################
            # batchsize = str_pose.shape[0]
            # print(batchsize)
            # img_sq = np.zeros( (batchsize,128,128,3))直接输入图片
            # # print(img_path)
            # id = 0
            # for path in img_path:
            #     img = cv2.imread(path)
            #     resize_img = cv2.resize(img,(128,128))
            #     img_sq[id] = resize_img
            #     id = id +1
            #     # print(resize_img.shape)
            #     # cv2.imshow('1',resize_img)
            #     # cv2.waitKey(1000)

            ##########################################加入光流数据
            flow = batch['flow']
            key_flow_seq = batch['key_flow_seq']
            # print(flow.shape)
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                # print('input_x  长度', len(input_x))
                # print('input_x 是********',input_x ,input_x.shape)
                neighbors_st, adjacency, first_history_indices = None, None, None

            if cfg.DATASET.NAME == 'PIE':
                pred_goal, pred_traj, loss_dict, dist_goal, dist_traj = model(
                    input_x,
                    y_global,
                    cur_pose,
                    str_pose,
                    key_pose,
                    flow,
                    key_flow_seq,
                    pose,
                    cur_img,
                    obd_speed,
                    heading_angle,
                    obs_cam_flow=obs_cam_flow,
                    obs_ped_flow=obs_ped_flow,
                    neighbors_st=neighbors_st,
                    adjacency=adjacency,
                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )
            else:
                pred_goal, pred_traj, loss_dict, dist_goal, dist_traj = model(
                    input_x,
                                                                          y_global,
                                                                          cur_pose,
                                                                          str_pose,
                                                                          key_pose,
                                                                          flow,
                                                                          key_flow_seq,
                                                                          pose,
                                                                          cur_img,
                                                                          obd_speed,
                                                                          heading_angle,
                                                                          neighbors_st=neighbors_st,
                                                                          adjacency=adjacency,
                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )
            # print('pred_goal是******',pred_goal, pred_goal.shape,'pred_traj是********',pred_traj,pred_traj.shape ,'loss_dict是********',loss_dict, 'dist_goal是********',dist_goal, 'dist_traj是*****',dist_traj)

            '''

            input_x [128, 15, 4])
            pred_goal ([128, 20, 4])
            pred_traj ([128, 45, 20, 4])
            loss_dict是******** {'loss_goal': tensor(0.0389, device='cuda:0', grad_fn=<MeanBackward0>),
            'loss_traj': tensor(1.9385, device='cuda:0', grad_fn=<MeanBackward0>),
            'loss_kld': tensor(0.0010,device='cuda:0', grad_fn=<ClampBackward1>)}
            dist_goal是******** None dist_traj是***** None


            '''
            if cfg.MODEL.LATENT_DIST == 'categorical':
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                       1. * loss_dict['mutual_info_p']
            else:
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + model.param_scheduler.kld_weight * loss_dict['loss_kld']
            model.param_scheduler.step()
            # loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad()  # avoid gradient accumulate from loss.backward()
            loss.backward()

            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)

            if cfg.VISUALIZE and iters % max(int(len(dataloader) / 5), 1) == 0:
                ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_goal=dist_goal)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                # print(X_global.shape,img_path) # (128, 15, 4),15张图片
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_train')


def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_goal_val = 0.0
    loss_traj_val = 0.0
    loss_KLD_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            ############################################################
            pose = batch['pose']
            cur_pose = batch['cur_pose']
            str_pose = batch['str_pose']
            key_pose = batch['key_pose']
            flow = batch['flow']
            key_flow_seq = batch['key_flow_seq']
            image_features = batch['image_features']
            cur_img = image_features[:, -1]
            if cfg.DATASET.NAME == 'PIE':
                obd_speed = batch['obd_speed'] / 54
                # print(obd_speed)
                heading_angle = batch['heading_angle'] / 360
            else:
                obd_speed = batch['obd_speed'] / 5
                heading_angle = None
            if cfg.DATASET.NAME == 'PIE' and 'obs_ped_flow' in batch:
                obs_ped_flow = batch['obs_ped_flow'].to(device)
                obs_ped_flow = torch.mean(obs_ped_flow, dim=-2)
                obs_cam_flow = batch['obs_ego_flow'].to(device)
                obs_cam_flow = torch.mean(obs_cam_flow, dim=-2)
                pred_ped_flow = batch['pred_ped_flow'].to(device)
                pred_ped_flow = torch.mean(pred_ped_flow, dim=-2)
                pred_cam_flow = batch['pred_ego_flow'].to(device)
                pred_cam_flow = torch.mean(pred_cam_flow, dim=-2)
            else:
                obs_ped_flow = None
                obs_cam_flow = None
                pred_ped_flow = None
                pred_cam_flow = None
            # batchsize = str_pose.shape[0]
            # img_sq = np.zeros((batchsize, 128, 128, 3))
            # id = 0
            # for path in img_path:
            #     img = cv2.imread(path)
            #     resize_img = cv2.resize(img, (128, 128))
            #     img_sq[id] = resize_img
            #     id = id + 1

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            if cfg.DATASET.NAME == 'PIE':
                pred_goal, pred_traj, loss_dict, _, _ = model(
                    input_x,
                    y_global,
                    cur_pose,
                    str_pose,
                    key_pose,
                    flow,
                    key_flow_seq,
                    pose,
                    cur_img,
                    obd_speed,
                    heading_angle,
                    obs_cam_flow=obs_cam_flow,
                    obs_ped_flow=obs_ped_flow,
                    neighbors_st=neighbors_st,
                    adjacency=adjacency,
                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )
            else:
                pred_goal, pred_traj, loss_dict, _, _ = model(
                    input_x,
                                                          y_global,
                                                          cur_pose,
                                                          str_pose,
                                                          key_pose,
                                                          flow,
                                                          key_flow_seq,
                                                          pose,
                                                          cur_img,
                                                          obd_speed,
                                                          heading_angle,
                                                          neighbors_st=neighbors_st,
                                                          adjacency=adjacency,
                                                          cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )

            # compute loss
            loss = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld']
            loss_goal_val += loss_dict['loss_goal'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            # loss_KLD_val += loss_dict['loss_kld'].item()
    loss_goal_val /= (iters + 1)
    loss_traj_val /= (iters + 1)
    loss_KLD_val /= (iters + 1)
    loss_val = loss_goal_val + loss_traj_val #+ loss_KLD_val

    info = "loss_val:{:.4f}, \
            loss_goal_val:{:.4f}, \
            loss_traj_val:{:.4f}, \
            loss_kld_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val, loss_KLD_val)

    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values({'loss_val': loss_val,
                           'loss_goal_val': loss_goal_val,
                           'loss_traj_val': loss_traj_val,
                           'loss_kld_val': loss_KLD_val})  # , step=epoch)
    else:
        print(info)
    return loss_val


def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
    model.eval()
    all_img_paths = []
    all_X_globals = []
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_distributions = []
    all_timesteps = []
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')

    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            ##############################################
            pose = batch['pose']
            flow = batch['flow']
            key_flow_seq = batch['key_flow_seq']
            cur_pose = batch['cur_pose']
            str_pose = batch['str_pose']
            key_pose = batch['key_pose']

            image_features = batch['image_features']
            cur_img = image_features[:, -1]
            if cfg.DATASET.NAME == 'PIE':
                obd_speed = batch['obd_speed'] / 54
                # print(obd_speed)
                heading_angle = batch['heading_angle'] / 360
            else:
                obd_speed = batch['obd_speed'] / 5
                heading_angle = None
            if cfg.DATASET.NAME == 'PIE' and 'obs_ped_flow' in batch:
                obs_ped_flow = batch['obs_ped_flow'].to(device)
                obs_ped_flow = torch.mean(obs_ped_flow, dim=-2)
                obs_cam_flow = batch['obs_ego_flow'].to(device)
                obs_cam_flow = torch.mean(obs_cam_flow, dim=-2)
                pred_ped_flow = batch['pred_ped_flow'].to(device)
                pred_ped_flow = torch.mean(pred_ped_flow, dim=-2)
                pred_cam_flow = batch['pred_ego_flow'].to(device)
                pred_cam_flow = torch.mean(pred_cam_flow, dim=-2)
            else:
                obs_ped_flow = None
                obs_cam_flow = None
                pred_ped_flow = None
                pred_cam_flow = None
            # batchsize = str_pose.shape[0]
            # img_sq = np.zeros((batchsize, 128, 128, 3))
            # id = 0
            # for path in img_path:
            #     img = cv2.imread(path)
            #     resize_img = cv2.resize(img, (128, 128))
            #     img_sq[id] = resize_img
            #     id = id + 1

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            y_global_fase = None
            if cfg.DATASET.NAME == 'PIE':
                pred_goal, pred_traj, _, dist_goal, dist_traj = model(
                    input_x,
                    y_global_fase,
                    cur_pose,
                    str_pose,
                    key_pose,
                    flow,
                    key_flow_seq,
                    pose,
                    cur_img,
                    obd_speed,
                    heading_angle,
                    obs_cam_flow=obs_cam_flow,
                    obs_ped_flow=obs_ped_flow,
                    neighbors_st=neighbors_st,
                    adjacency=adjacency,
                    z_mode=False,
                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )
            else:
                pred_goal, pred_traj, _, dist_goal, dist_traj = model(
                    input_x,
                                                                  y_global_fase,
                                                                  cur_pose,
                                                                  str_pose,
                                                                  key_pose,
                                                                  flow,
                                                                  key_flow_seq,
                                                                  pose,
                                                                  cur_img,
                                                                  obd_speed,
                                                                  heading_angle,
                                                                  neighbors_st=neighbors_st,
                                                                  adjacency=adjacency,
                                                                  z_mode=False,
                                                                  cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )

            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=dist_traj,
                               dist_goal=dist_goal)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            all_img_paths.extend(img_path)
            all_X_globals.append(X_global)
            all_pred_goals.append(pred_goal)
            all_pred_trajs.append(pred_traj)
            all_gt_goals.append(y_global[:, -1])
            all_gt_trajs.append(y_global)
            all_timesteps.append(batch['timestep'].numpy())
            if dist_traj is not None:
                all_distributions.append(dist_traj)
            else:
                all_distributions.append(dist_goal)
            if cfg.VISUALIZE and iters % max(int(len(dataloader) / 5), 1) == 0:
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')

        # Evaluate
        all_X_globals = np.concatenate(all_X_globals, axis=0)
        # all_pred_goals = np.concatenate(all_pred_goals, axis=0)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        # all_gt_goals = np.concatenate(all_gt_goals, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)
        if hasattr(all_distributions[0], 'mus'):
            distribution = model.GMM(torch.cat([d.input_log_pis for d in all_distributions], axis=0),
                                     torch.cat([d.mus for d in all_distributions], axis=0),
                                     torch.cat([d.log_sigmas for d in all_distributions], axis=0),
                                     torch.cat([d.corrs for d in all_distributions], axis=0))
        else:
            distribution = None
            # eval_pred_results = evaluate(all_pred_goals, all_gt_goals)
        mode = 'bbox' if all_gt_trajs.shape[-1] == 4 else 'point'
        eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=distribution,
                                           bbox_type=cfg.DATASET.BBOX_TYPE)
        for key, value in eval_results.items():
            info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)

        if hasattr(logger, 'log_values'):
            logger.log_values(eval_results)

        if test_mode:
            # save inputs, redictions and targets for test mode
            outputs = {'img_path': all_img_paths, 'X_global': all_X_globals, 'timestep': all_timesteps,
                       'pred_trajs': all_pred_trajs, 'gt_trajs': all_gt_trajs, 'distributions': distribution}

            if not os.path.exists(cfg.OUT_DIR):
                os.makedirs(cfg.OUT_DIR)
            output_file = os.path.join(cfg.OUT_DIR, '{}_{}.pkl'.format(cfg.MODEL.LATENT_DIST, cfg.DATASET.NAME))
            print("Writing outputs to: ", output_file)
            pkl.dump(outputs, open(output_file, 'wb'))

    # Mevaluate KDE NLL, since we sample 2000, need to use a smaller batchsize
    if eval_kde_nll:
        dataloader_params = {
            "batch_size": cfg.TEST.KDE_BATCH_SIZE,
            "shuffle": False,
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "collate_fn": dataloader.collate_fn,
        }
        kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
        inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)


def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_kde_nll = []
    all_per_step_kde_nll = []
    num_samples = model.K
    model.K = 2000
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            pose = batch['pose']
            flow = batch['flow']
            key_flow_seq = batch['key_flow_seq']
            cur_pose = batch['cur_pose']
            str_pose = batch['str_pose']
            key_pose = batch['key_pose']
            image_features = batch['image_features']
            cur_img = image_features[:, -1]
            if cfg.DATASET.NAME == 'PIE':
                obd_speed = batch['obd_speed'] / 54
                # print(obd_speed)
                heading_angle = batch['heading_angle'] / 360
            else:
                obd_speed = batch['obd_speed'] / 5
                heading_angle = None
            # batchsize = str_pose.shape[0]
            # img_sq = np.zeros((batchsize, 128, 128, 3))
            # id = 0
            # for path in img_path:
            #     img = cv2.imread(path)
            #     resize_img = cv2.resize(img, (128, 128))
            #     img_sq[id] = resize_img
            #     id = id + 1

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            _ = None
            if cfg.DATASET.NAME == 'PIE' and 'obs_ped_flow' in batch:
                obs_ped_flow = batch['obs_ped_flow'].to(device)
                obs_ped_flow = torch.mean(obs_ped_flow, dim=-2)
                obs_cam_flow = batch['obs_ego_flow'].to(device)
                obs_cam_flow = torch.mean(obs_cam_flow, dim=-2)
            else:
                obs_ped_flow = None
                obs_cam_flow = None
            if cfg.DATASET.NAME == 'PIE':
                pred_goal, pred_traj, _, _, _ = model(
                    input_x,
                    _,
                    cur_pose,
                    str_pose,
                    key_pose,
                    flow,
                    key_flow_seq,
                    pose,
                    cur_img,
                    obd_speed,
                    heading_angle,
                    obs_cam_flow=obs_cam_flow,
                    obs_ped_flow=obs_ped_flow,
                    neighbors_st=neighbors_st,
                    adjacency=adjacency,
                    z_mode=False,
                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )
            else:
                pred_goal, pred_traj, _, _, _ = model(
                    input_x,
                                                  _,
                                                  cur_pose,
                                                  str_pose,
                                                  key_pose,
                                                  flow,
                                                  key_flow_seq,
                                                  pose,
                                                  cur_img,
                                                  obd_speed,
                                                  heading_angle,
                                                  neighbors_st=neighbors_st,
                                                  adjacency=adjacency,
                                                  z_mode=False,
                                                  cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                    first_history_indices=first_history_indices,
                )

            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=None, dist_goal=None)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            for i in range(len(pred_traj)):
                KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(pred_traj[i:i + 1], y_global[i:i + 1])
                all_kde_nll.append(KDE_NLL)
                all_per_step_kde_nll.append(KDE_NLL_PER_STEP)
        KDE_NLL = np.array(all_kde_nll).mean()
        KDE_NLL_PER_STEP = np.stack(all_per_step_kde_nll, axis=0).mean(axis=0)
        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.MODEL.LATENT_DIST == 'categorical' else {'KDE_NLL': KDE_NLL,
                                                                                        'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset model.K back to 20
    model.K = num_samples
    return KDE_NLL
