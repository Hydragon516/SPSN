import os
import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from data import get_loader, get_testloader
from loss import IoU_loss
from model import MyModel, weights_init

import config
from logger import *

os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN['GPU']

valid_list = (config.DATA['val_dataset']).replace(",", "").split(" ")
test_list = (config.DATA['test_dataset']).replace(",", "").split(" ")

best_mae = {i:[] for i in list(range(len(valid_list)))}
for i in range(len(best_mae)):
    best_mae[i] = 1


def train(epoch, trainloader, optimizer, model, device, scheduler):
    model.train()
    avg_loss1 = 0
    avg_loss2 = 0
    avg_loss3 = 0
    avg_loss4 = 0

    loss_ss1 = nn.BCELoss()
    loss_ss2 = nn.BCELoss()
    loss_rely = nn.L1Loss()

    total_com = 0
    total_ss_acc = 0

    depth_total_com = 0
    depth_total_ss_acc = 0

    PTTM = pttm()

    for idx, batch in enumerate(trainloader):
        PTTM.print_status(epoch, idx, trainloader)
        images, gts, depths, ss_map, ss_label, depth_ss_map, depth_ss_label = batch
        
        if images.size(0) == 1:
            continue
        
        images, gts, depths, ss_map, ss_label, depth_ss_map, depth_ss_label = images.to(device), gts.to(device), depths.to(device), ss_map.to(device), ss_label.to(device), depth_ss_map.to(device), depth_ss_label.to(device)

        optimizer.zero_grad()
        pred, pred_ss_label, depth_pred_ss_label, pred_rely, label_rely = model(images, depths, ss_map, depth_ss_map)

        ss_gt = ss_map * ss_label.unsqueeze(-1).unsqueeze(-1)
        ss_gt = torch.sum(ss_gt, dim=1).unsqueeze(1).float()
        ss_pred = ss_map * pred_ss_label.unsqueeze(-1).unsqueeze(-1)
        ss_pred = torch.sum(ss_pred, dim=1).unsqueeze(1).float()

        depth_ss_gt = depth_ss_map * depth_ss_label.unsqueeze(-1).unsqueeze(-1)
        depth_ss_gt = torch.sum(depth_ss_gt, dim=1).unsqueeze(1).float()
        depth_ss_pred = depth_ss_map * depth_pred_ss_label.unsqueeze(-1).unsqueeze(-1)
        depth_ss_pred = torch.sum(depth_ss_pred, dim=1).unsqueeze(1).float()

        loss1 = IoU_loss(pred, gts)
        loss2 = loss_ss1(ss_pred, ss_gt)
        loss3 = loss_ss2(depth_ss_pred, depth_ss_gt)
        loss4 = loss_rely(pred_rely, label_rely)

        ######
        pred_ss_label = pred_ss_label.cpu().detach().numpy()
        ss_label = ss_label.cpu().detach().numpy()

        pred_ss_label[pred_ss_label > 0.5] = 1
        pred_ss_label[pred_ss_label <= 0.5] = 0

        total_com += ss_label.shape[0] * ss_label.shape[1]
        total_ss_acc += np.sum(pred_ss_label == ss_label)

        ######
        depth_pred_ss_label = depth_pred_ss_label.cpu().detach().numpy()
        depth_ss_label = depth_ss_label.cpu().detach().numpy()

        depth_pred_ss_label[depth_pred_ss_label > 0.5] = 1
        depth_pred_ss_label[depth_pred_ss_label <= 0.5] = 0

        depth_total_com += depth_ss_label.shape[0] * depth_ss_label.shape[1]
        depth_total_ss_acc += np.sum(depth_pred_ss_label == depth_ss_label)

        ######
        total_loss = loss1 + loss2 + loss3 + 10 * loss4
        
        avg_loss1 += loss1.item()
        avg_loss2 += loss2.item()
        avg_loss3 += loss3.item()
        avg_loss4 += loss4.item()

        total_loss.backward()
        optimizer.step()
        scheduler.step()
    
    print("")

    avg_loss1 = avg_loss1 / (idx + 1)
    avg_loss2 = avg_loss2 / (idx + 1)
    avg_loss3 = avg_loss3 / (idx + 1)
    avg_loss4 = avg_loss4 / (idx + 1)

    ss_acc = total_ss_acc / total_com
    depth_ss_acc = depth_total_ss_acc / depth_total_com

    print(
        "Epoch: #{0} Batch: {1}\t"
        "Lr: {lr:.7f}\n"
        "LOSS pred: {loss1:.4f}\t"
        "LOSS ss: {loss3:.4f}\t"
        "SS ACC: {acc:.4f}\n"
        "LOSS Depth ss: {loss4:.4f}\t"
        "Depth SS ACC: {depth_acc:.4f}\t"
        "Reliance loss: {loss5:.4f}\t"
        .format(epoch, idx, lr=optimizer.param_groups[-1]['lr'], loss1=avg_loss1, loss3=avg_loss3, acc=ss_acc, loss4=avg_loss4, depth_acc=depth_ss_acc, loss5 = 10*avg_loss4)
    )

    avg_loss1 = 0
    avg_loss2 = 0
    avg_loss3 = 0
    avg_loss4 = 0

    total_com = 0
    total_ss_acc = 0

    depth_total_com = 0
    depth_total_ss_acc = 0


def valid(epoch, model, device, work_dir):
    print("Evaluating model...")
    val_image_root = os.path.join(config.DATA['data_root'], 'TestDataset')
    test_loader = get_testloader(val_image_root, valid_list, config.TRAIN['batch_size'], 352)

    model.eval()
    with torch.no_grad():
        mae_buffer = {i:0 for i in valid_list}
        cnt_buffer = {i:0 for i in valid_list}

        PTTM = pttm()
        
        for idx, batch in enumerate(test_loader):
            PTTM.print_status(epoch, idx, test_loader)
            image, gt, depth, info, _, ss_map, _, depth_ss_map, _ = batch
            B = image.shape[0]

            ori_H = info[0][0]
            ori_W = info[0][1]

            image = image.to(device)
            ss_map = ss_map.to(device)
            depth_ss_map = depth_ss_map.to(device)
            depth = depth.to(device)
            gt = gt.to(device)

            preds, _, _, _, _ = model(image, depth, ss_map, depth_ss_map)

            res = preds[3]

            for b in range(B):
                res_slice = res[b, :, :, :].unsqueeze(0).float()
                gt_slice = gt[b, :, :, :].unsqueeze(0).float()

                res_slice = F.upsample(res_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                res_slice = (res_slice - res_slice.min()) / (res_slice.max() - res_slice.min() + 1e-8)

                gt_slice = F.upsample(gt_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                gt_slice /= (torch.max(gt_slice) + 1e-8)

                valid_name = info[1][b]

                mae_buffer[valid_name] += torch.sum(torch.abs(res_slice - gt_slice)).item() / (ori_H[b].item() * ori_W[b].item())
                cnt_buffer[valid_name] += 1
        
        print("")

        for x, n in enumerate(valid_list):
            mae = mae_buffer[n] / cnt_buffer[n]
            print('DATA: {} MAE: {} ---> bestMAE: {}'.format(n, mae, best_mae[x]))

            if mae < best_mae[x]:
                best_mae[x] = mae

                save_model(work_dir, epoch, model, n)
                print("Saved best model! :", n)


def visual(device, work_dir, valid_name, valid_list_len, target):
    model = MyModel()
    model.encoder_depth.vgg[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    model_dir = os.path.join(work_dir, "model")

    if valid_list_len == 1:
        checkpoint = torch.load(model_dir + "/{}_best_model.pth".format(target))
    else:
        checkpoint = torch.load(model_dir + "/{}_best_model.pth".format(valid_name))
    
    model.load_state_dict(checkpoint['model_state_dict'])

    with Loader("Saving {} dataset images...".format(valid_name)):
        valid_list = [valid_name]

        val_image_root = os.path.join(config.DATA['data_root'], 'TestDataset')
        test_loader = get_testloader(val_image_root, valid_list, config.TRAIN['batch_size'], 352)

        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                image, gt, depth, info, img_for_post, ss_map, ss_label, depth_ss_map, depth_ss_label = batch
                B = image.shape[0]

                ori_H = info[0][0]
                ori_W = info[0][1]

                image = image.to(device)
                ss_map = ss_map.to(device)
                ss_label = ss_label.to(device)
                depth_ss_map = depth_ss_map.to(device)
                depth_ss_label = depth_ss_label.to(device)
                depth = depth.to(device)

                preds, ss_maps_pred, depth_ss_maps_pred, _, _ = model(image, depth, ss_map, depth_ss_map)

                copy_ss_map = ss_map.clone()
                copy_depth_ss_map = depth_ss_map.clone()

                ss_gt = ss_map * ss_label.unsqueeze(-1).unsqueeze(-1)
                ss_gt = torch.sum(ss_gt, dim=1).unsqueeze(-1).float()

                depth_ss_gt = depth_ss_map * depth_ss_label.unsqueeze(-1).unsqueeze(-1)
                depth_ss_gt = torch.sum(depth_ss_gt, dim=1).unsqueeze(-1).float()

                depth_ss_pred = depth_ss_map * depth_ss_maps_pred.unsqueeze(-1).unsqueeze(-1)
                depth_ss_pred = torch.sum(depth_ss_pred, dim=1).unsqueeze(-1).float()

                ss_pred = ss_map * ss_maps_pred.unsqueeze(-1).unsqueeze(-1)
                ss_pred = torch.sum(ss_pred, dim=1).unsqueeze(-1).float()

                ss_gt = ss_gt.cpu().detach().numpy()
                depth_ss_gt = depth_ss_gt.cpu().detach().numpy()

                ss_pred = ss_pred.cpu().detach().numpy()
                depth_ss_pred = depth_ss_pred.cpu().detach().numpy()

                depth = depth.squeeze(-1).cpu().detach().numpy()

                res = preds[3]

                for b in range(B):
                    res_slice = res[b, :, :, :].unsqueeze(0)
                    gt_slice = gt[b, :, :, :].squeeze(0)
                    depth_slice = depth[b, :, :].squeeze(0)
                    ori_image_slice = img_for_post[b, :, :, :].squeeze(0)
                    copy_ss_map_slice = copy_ss_map[b, :, :, :]
                    copy_depth_ss_map_slice = copy_depth_ss_map[b, :, :, :]

                    blank_image = np.zeros((copy_ss_map_slice.shape[1], copy_ss_map_slice.shape[2]), np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

                    for i in range(copy_ss_map_slice.shape[0]):
                        a_copy_ss_map_slice = copy_ss_map_slice[i, :, :]
                        a_numpy_ss = a_copy_ss_map_slice.cpu().detach().numpy()
                        erosion = cv2.erode(a_numpy_ss, kernel)
                        a_edge_ss = a_numpy_ss - erosion
                        a_edge_ss = (a_edge_ss * 255).astype(np.uint8)

                        blank_image = blank_image + a_edge_ss
                    
                    rgb_edge_gt = blank_image
                    rgb_edge_gt = cv2.resize(rgb_edge_gt, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    rgb_edge_gt = rgb_edge_gt.astype(np.uint8)

                    blank_image = np.zeros((copy_depth_ss_map_slice.shape[1], copy_depth_ss_map_slice.shape[2]), np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

                    for i in range(copy_depth_ss_map_slice.shape[0]):
                        a_copy_depth_ss_map_slice = copy_depth_ss_map_slice[i, :, :]
                        a_numpy_depth_ss = a_copy_depth_ss_map_slice.cpu().detach().numpy()
                        erosion = cv2.erode(a_numpy_depth_ss, kernel)
                        a_edge_ss = a_numpy_depth_ss - erosion
                        a_edge_ss = (a_edge_ss * 255).astype(np.uint8)

                        blank_image = blank_image + a_edge_ss
                    
                    depth_edge_gt = blank_image
                    depth_edge_gt = cv2.resize(depth_edge_gt, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    depth_edge_gt = depth_edge_gt.astype(np.uint8)

                    gt_slice = np.asarray(gt_slice, np.float32)
                    gt_slice /= (gt_slice.max() + 1e-8)
                    
                    res_slice = F.upsample(res_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                    res_slice = res_slice.permute(0, 2, 3, 1).cpu().detach().squeeze(0).squeeze(-1).numpy()
                    res_slice = (res_slice - res_slice.min()) / (res_slice.max() - res_slice.min() + 1e-8)

                    cat_res = cv2.cvtColor(np.array(res_slice * 255), cv2.COLOR_GRAY2BGR)
                    cat_res = cv2.resize(cat_res, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    cat_res = cat_res.astype(np.uint8)
                    
                    cat_gt = cv2.cvtColor(np.array(gt_slice * 255), cv2.COLOR_GRAY2BGR)
                    cat_gt = cv2.resize(cat_gt, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    cat_gt = cat_gt.astype(np.uint8)

                    cat_depth = cv2.cvtColor(np.array(depth_slice * 255), cv2.COLOR_GRAY2BGR)
                    cat_depth = cv2.resize(cat_depth, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)

                    cat_ori = cv2.cvtColor(np.array(ori_image_slice), cv2.COLOR_RGB2BGR)
                    cat_ori = cv2.resize(cat_ori, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)

                    ######
                    ss_mask = ss_gt[b, :, :, :].squeeze(-1)
                    ss_mask = cv2.cvtColor(np.array(ss_mask * 255), cv2.COLOR_GRAY2BGR)
                    ss_mask = cv2.resize(ss_mask, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    ss_mask = ss_mask.astype(np.uint8)

                    ss_mask_pred = ss_pred[b, :, :, :].squeeze(-1)
                    ss_mask_pred = cv2.cvtColor(np.array(ss_mask_pred * 255), cv2.COLOR_GRAY2BGR)
                    ss_mask_pred = cv2.resize(ss_mask_pred, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    ss_mask_pred = ss_mask_pred.astype(np.uint8)

                    depth_ss_mask = depth_ss_gt[b, :, :, :].squeeze(-1)
                    depth_ss_mask = cv2.cvtColor(np.array(depth_ss_mask * 255), cv2.COLOR_GRAY2BGR)
                    depth_ss_mask = cv2.resize(depth_ss_mask, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    depth_ss_mask = depth_ss_mask.astype(np.uint8)

                    depth_ss_mask_pred = depth_ss_pred[b, :, :, :].squeeze(-1)
                    depth_ss_mask_pred = cv2.cvtColor(np.array(depth_ss_mask_pred * 255), cv2.COLOR_GRAY2BGR)
                    depth_ss_mask_pred = cv2.resize(depth_ss_mask_pred, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    depth_ss_mask_pred = depth_ss_mask_pred.astype(np.uint8)
                    ######

                    cat_ori = cat_ori.astype(np.uint8)
                    cat_depth = cat_depth.astype(np.uint8)

                    ss_mask_pred[rgb_edge_gt == 255] = [0, 255, 255]
                    ss_mask[rgb_edge_gt == 255] = [0, 255, 255]
                    depth_ss_mask_pred[depth_edge_gt == 255] = [0, 255, 255]
                    depth_ss_mask[depth_edge_gt == 255] = [0, 255, 255]

                    result = cv2.hconcat([cat_ori, cat_depth, cat_res, cat_gt, ss_mask_pred, ss_mask, depth_ss_mask_pred, depth_ss_mask])

                    valid_name = info[1][b]
                    name = info[2][b]

                    total_dir = os.path.join(work_dir, "result", "total", valid_name)
                    if not os.path.exists(total_dir):
                        os.makedirs(total_dir)
                    
                    pred_dir = os.path.join(work_dir, "result", "pred", valid_name)
                    if not os.path.exists(pred_dir):
                        os.makedirs(pred_dir)
                    
                    gt_dir = os.path.join(work_dir, "result", "gt", valid_name)
                    if not os.path.exists(gt_dir):
                        os.makedirs(gt_dir)

                    cv2.imwrite(os.path.join(total_dir, name), result)
                    cv2.imwrite(os.path.join(pred_dir, name), cat_res)
                    cv2.imwrite(os.path.join(gt_dir, name), cat_gt)


def main():
    work_dir = make_new_work_space()
    save_config_file(work_dir)

    with Loader("Load dataset..."):
        train_image_root = os.path.join(config.DATA['data_root'], 'TrainDataset', 'RGB') + '/'
        train_gt_root = os.path.join(config.DATA['data_root'], 'TrainDataset', 'GT') + '/'
        train_depth_root = os.path.join(config.DATA['data_root'], 'TrainDataset', 'depth') + '/'

        train_loader = get_loader(train_image_root, train_gt_root, train_depth_root, batchsize=config.TRAIN['batch_size'], trainsize=352)
    
    with Loader("Check device..."):
        device = torch.device("cuda")

    with Loader("Load model..."):
        model = MyModel()
        model.apply(weights_init)
        model.encoder_rgb.vgg.load_state_dict(torch.load("./pretrain/vgg16_feat.pth"))
        model.encoder_depth.vgg.load_state_dict(torch.load("./pretrain/vgg16_feat.pth"))
        
        model.encoder_depth.vgg[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        model = model.to(device)
        model = torch.nn.DataParallel(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    
    with Loader("Load optimizer..."):
        params = model.parameters()
        optimizer = torch.optim.Adam(params, config.TRAIN['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*config.TRAIN['epoch'], eta_min=config.TRAIN['learning_rate']/10)

    print("")
    print("Training start!")
    for epoch in range(config.TRAIN['epoch']):
        train(epoch, train_loader, optimizer, model, device, scheduler)
        print("")
        valid(epoch, model, device, work_dir)
        print("")

    print("Training finish!")

    valid_list_len = len(valid_list)

    print("Saving result images...")
    if valid_list_len == 1:
        target = valid_list[0]
        for test_name in test_list:
            visual(device, work_dir, test_name, valid_list_len, target)


if __name__ == '__main__':
    main()