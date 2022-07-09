import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import config

np.set_printoptions(suppress=True, threshold=1e5)

def resize(input, target_size=(352, 352)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers)
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats)
        
        return feats


class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        
        return pred


class Res(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1), 
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, in_channel, 3, 1, 1)
                                  )
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), 
                                  nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
                                  )

    def forward(self, feats):
        feats = feats + self.conv1(feats)
        feats = F.relu(feats, inplace=True)
        feats = self.conv2(feats)

        return feats


class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 64, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                                        nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, cosal_map, old_feats):
        _, _, H, W = low_level_feats.shape

        cosal_map = resize(cosal_map, [H, W])
        old_feats = resize(old_feats, [H, W])

        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map, old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        
        return new_feats, new_cosal_map


class GET_Prototype(nn.Module):
    def __init__(self):
        super(GET_Prototype, self).__init__()
    
    def Weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        
        return supp_feat

    def forward(self, x, ss_map):
        COM = ss_map.size(1)
        _, H, W = x.size(0), x.size(2), x.size(3)

        prototype_list = []

        for i in range(COM):
            ss_map_slice = ss_map[:, i, :, :].unsqueeze(1).float()
            ss_map_slice = F.interpolate(ss_map_slice, (H, W), mode='bilinear', align_corners=True)
            ss_mask = ss_map_slice

            prototype = self.Weighted_GAP(x, ss_mask)
            prototype = prototype.squeeze(-1).squeeze(-1).unsqueeze(1)

            prototype_list.append(prototype)
        
        prototype_block = torch.cat(prototype_list, dim=1)

        return prototype_block


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)

        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)

        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)

        N, num, c = feature.size()

        x_theta = self.theta(feature)
        x_phi = self.phi(feature)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

        f_div_C = F.softmax(attention, dim=-1)

        g_x = self.g(feature)

        y = torch.matmul(f_div_C, g_x)

        W_y = self.W(y).contiguous().view(N, num, c)

        att_fea = ori_feature.permute(0, 2, 1) + W_y

        return att_fea


class Prototype_Sampler_Network(nn.Module):
    def __init__(self, in_channels):
        super(Prototype_Sampler_Network, self).__init__()
        self.TF = Transformer(in_channels)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.fc = nn.Linear(in_channels, 1)
       
    def forward(self, prototype_block):
        att_prototype_block = self.TF(prototype_block)

        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        
        graph_prototype_block = graph_prototype.permute(0, 2, 1)
        graph_sampler = self.fc(graph_prototype_block)
        graph_sampler = graph_sampler.squeeze(-1)

        total_samler_block = graph_sampler
        total_samler_block = F.sigmoid(total_samler_block)

        sampled_prototype_block = total_samler_block.unsqueeze(-1) * att_prototype_block

        return total_samler_block, sampled_prototype_block


class Fusion_Module(nn.Module):
    def __init__(self, channel):
        super(Fusion_Module, self).__init__()
        self.conv_28_1 = nn.Conv2d(channel, channel, 1)
        self.conv_14_1 = nn.Conv2d(channel, channel, 1)
        self.conv_14_2 = nn.Conv2d(channel, channel, 1)
        self.conv_7_1 = nn.Conv2d(channel, channel, 1)
        self.conv_out = nn.Conv2d(channel, channel, 1)

    def forward(self, f_28, f_14, f_7):
        f_7 = self.conv_7_1(f_7)
        up_f_14 = F.interpolate(f_7, scale_factor=2, mode='bilinear')

        f_14 = self.conv_14_1(f_14)
        f_14 = f_14 + up_f_14
        f_14 = self.conv_14_2(f_14)
        up_f_28 = F.interpolate(f_14, scale_factor=2, mode='bilinear')

        f_28 =self.conv_28_1(f_28)
        f_28 = f_28 + up_f_28
        f_28 = self.conv_out(f_28)

        return f_28


class Get_Correlation_Map(nn.Module):
    def __init__(self):
        super(Get_Correlation_Map, self).__init__()
    
    def forward(self, x, prototype_block):
        B, _, _ = prototype_block.size()

        CM = []

        for b in range(B):
            prototypes = prototype_block[b, :].unsqueeze(-1).unsqueeze(-1)
            x_batch = x[b, :, :, :].unsqueeze(0)
            Correlation_Map = F.conv2d(x_batch, prototypes)
            CM.append(Correlation_Map)

        Correlation_Maps = torch.cat(CM, dim=0)

        return Correlation_Maps


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 

        return out


class ReliabilitySelector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReliabilitySelector, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1 = nn.BatchNorm2d(out_channels)
        
        self.conv_3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_conv_3x3 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.fc = nn.Linear(3872, 2)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = F.relu(self.bn_conv_1x1(self.conv_1x1(x)))
        x = F.relu(self.bn_conv_3x3(self.conv_3x3(x)))
        x = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(x)))
        
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        x = self.dropout(x)
        x = F.sigmoid(x)

        return x


class ReliabilityCorrelationGT(nn.Module):
    def __init__(self):
        super(ReliabilityCorrelationGT, self).__init__()
        self.NS = config.TRAIN['num_superpixels']

    def forward(self, gt_guide, reliance_input):
        rgb_distance = []
        depth_distance = []
        sum_iou = []
        
        for i in range(gt_guide.shape[0]):
            rgb_distance.append(abs(gt_guide[i][0] - reliance_input[i][0:self.NS]))
            depth_distance.append(abs(gt_guide[i][0] - reliance_input[i][self.NS:(self.NS * 2)]))
            sum_iou.append([torch.mean(rgb_distance[i]), torch.mean(depth_distance[i])])
        
        sum_iou = torch.tensor(sum_iou).cuda()
        
        return sum_iou

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        NS = config.TRAIN['num_superpixels']
        self.encoder_rgb = VGG16()
        self.encoder_depth = VGG16()

        self.aspp_rgb = ASPP(128, 128)
        self.aspp_depth = ASPP(128, 128)

        self.rgb_COMP6 = nn.Sequential(nn.MaxPool2d(2, 2), 
                                    nn.Conv2d(512, 128, 1),
                                    nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
                                )
        self.rgb_COMP5 = nn.Conv2d(512, 128, 1)
        self.rgb_COMP4 = nn.Conv2d(512, 128, 1)

        self.depth_COMP6 = nn.Sequential(nn.MaxPool2d(2, 2), 
                                    nn.Conv2d(512, 128, 1),
                                    nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                )
        self.depth_COMP5 = nn.Conv2d(512, 128, 1)
        self.depth_COMP4 = nn.Conv2d(512, 128, 1)

        self.rgb_FM = Fusion_Module(128)
        self.depth_FM = Fusion_Module(128)

        self.rgb_GETP = GET_Prototype()
        self.depth_GETP = GET_Prototype()
        
        self.rgb_PSN = Prototype_Sampler_Network(128)
        self.depth_PSN = Prototype_Sampler_Network(128)

        self.rgb_GCM6 = Get_Correlation_Map()
        self.rgb_GCM5 = Get_Correlation_Map()
        self.rgb_GCM4 = Get_Correlation_Map()

        self.depth_GCM6 = Get_Correlation_Map()
        self.depth_GCM5 = Get_Correlation_Map()
        self.depth_GCM4 = Get_Correlation_Map()

        self.guide_selector = ReliabilitySelector(NS * 2, 2)

        self.RSM_FM = Fusion_Module(NS * 2)
        self.get_reliablity_GT = ReliabilityCorrelationGT()

        self.total_fusion6 = Res(NS * 2, NS)
        self.total_fusion5 = Res(NS * 2, NS)
        self.total_fusion4 = Res(NS * 2, NS)

        self.rgb_merge_co_56 = Res(NS * 2, NS)
        self.rgb_merge_co_45 = nn.Sequential(Res(NS * 2, NS), nn.Conv2d(NS, 32, 1))
        self.rgb_get_pred_4 = Prediction(32)
        self.rgb_refine_3 = Decoder_Block(256)
        self.rgb_refine_2 = Decoder_Block(128)
        self.rgb_refine_1 = Decoder_Block(64)

    def forward(self, image, depth, rgb_ss_map, depth_ss_map):
        com = rgb_ss_map.size(1)

        ## Encoder for RGB ###
        rgb_f1 = self.encoder_rgb(image, 'conv1_1', 'conv1_2_mp')
        rgb_f2 = self.encoder_rgb(rgb_f1, 'conv1_2_mp', 'conv2_2_mp')
        rgb_f3 = self.encoder_rgb(rgb_f2, 'conv2_2_mp', 'conv3_3_mp')
        rgb_f4 = self.encoder_rgb(rgb_f3, 'conv3_3_mp', 'conv4_3_mp')
        rgb_f5 = self.encoder_rgb(rgb_f4, 'conv4_3_mp', 'conv5_3_mp')
        #######################

        ## Encoder for Depth ###
        depth_f1 = self.encoder_depth(depth, 'conv1_1', 'conv1_2_mp')
        depth_f2 = self.encoder_depth(depth_f1, 'conv1_2_mp', 'conv2_2_mp')
        depth_f3 = self.encoder_depth(depth_f2, 'conv2_2_mp', 'conv3_3_mp')
        depth_f4 = self.encoder_depth(depth_f3, 'conv3_3_mp', 'conv4_3_mp')
        depth_f5 = self.encoder_depth(depth_f4, 'conv4_3_mp', 'conv5_3_mp')
        #######################

        ### Feature Fusion Module ###
        rgb_cf6 = self.rgb_COMP6(rgb_f5)
        rgb_cf6 = self.aspp_rgb(rgb_cf6)
        rgb_cf5 = self.rgb_COMP5(rgb_f5)
        rgb_cf4 = self.rgb_COMP4(rgb_f4)

        depth_cf6 = self.depth_COMP6(depth_f5)
        depth_cf6 = self.aspp_depth(depth_cf6)
        depth_cf5 = self.depth_COMP5(depth_f5)
        depth_cf4 = self.depth_COMP4(depth_f4)

        rgb_fusion_feature = self.rgb_FM(rgb_cf4, rgb_cf5, rgb_cf6)                                 # rgb_fusion_feature : (N, 128, H/8, W/8)
        depth_fusion_feature = self.depth_FM(depth_cf4, depth_cf5, depth_cf6)                       # depth_fusion_feature : (N, 128, H/8, W/8)
        #############################
        
        ### Prototype Generating Module ###
        rgb_prototype_block = self.rgb_GETP(rgb_fusion_feature, rgb_ss_map)                         # rgb_prototype_block : (N, NS, 128)
        depth_prototype_block = self.depth_GETP(depth_fusion_feature, depth_ss_map)                 # depth_prototype_block : (N, NS, 128)
        ###################################

        ### Prototype Sampling Network Module ###
        # Part A & B #
        rgb_sampler_block, rgb_sampled_prototype_block = self.rgb_PSN(rgb_prototype_block)          # rgb_sampler_block : (N, NS)
                                                                                                    # rgb_sampled_prototype_block : (N, NS, 128)
        depth_sampler_block, depth_sampled_prototype_block = self.depth_PSN(depth_prototype_block)  # depth_sampler_block : (N, NS)
                                                                                                    # depth_sampled_prototype_block : (N, NS, 128)
        
        # Part D (Part C is included in the train.py) #
        rgb_sampled_feature6 = self.rgb_GCM6(rgb_cf6, rgb_sampled_prototype_block)                  # rgb_sampled_feature6 : (N, NS, H/32, W/32)
        rgb_sampled_feature5 = self.rgb_GCM5(rgb_cf5, rgb_sampled_prototype_block)                  # rgb_sampled_feature5 : (N, NS, H/16, W/16)
        rgb_sampled_feature4 = self.rgb_GCM4(rgb_cf4, rgb_sampled_prototype_block)                  # rgb_sampled_feature4 : (N, NS, H/8, W/8)

        depth_sampled_feature6 = self.depth_GCM6(depth_cf6, depth_sampled_prototype_block)          # depth_sampled_feature6 : (N, NS, H/32, W/32)
        depth_sampled_feature5 = self.depth_GCM5(depth_cf5, depth_sampled_prototype_block)          # depth_sampled_feature5 : (N, NS, H/16, W/16)
        depth_sampled_feature4 = self.depth_GCM4(depth_cf4, depth_sampled_prototype_block)          # depth_sampled_feature4 : (N, NS, H/8, W/8)

        tortal_sampled_feature6 = torch.cat([rgb_sampled_feature6, depth_sampled_feature6], dim=1)  # tortal_sampled_feature6 : (N, 2*NS, H/32, W/32)                                                                               # rgb_sampled_feature6 torch.Size([16, 100, 11, 11])
        tortal_sampled_feature5 = torch.cat([rgb_sampled_feature5, depth_sampled_feature5], dim=1)  # tortal_sampled_feature5 : (N, 2*NS, H/16, W/16)
        tortal_sampled_feature4 = torch.cat([rgb_sampled_feature4, depth_sampled_feature4], dim=1)  # tortal_sampled_feature4 : (N, 2*NS, H/8, W/8)
        ##########################################

        ### Reliance Selection Module ###
        # RSM
        reliance_fused_feature = self.RSM_FM(tortal_sampled_feature4, tortal_sampled_feature5, tortal_sampled_feature6)     # reliance_fused_feature : (N, 2*NS, H/8, W/8)
        reliance_4 = self.guide_selector(reliance_fused_feature)        # reliance_4 : (N, 2)

        reliance4_r = reliance_4[:,0].unsqueeze(-1).repeat(1, com).unsqueeze(-1).unsqueeze(-1)  # reliance4_r : (N, NS, 1, 1) 
        reliance4_d = reliance_4[:,1].unsqueeze(-1).repeat(1, com).unsqueeze(-1).unsqueeze(-1)  # reliance4_d : (N, NS, 1, 1)
        reliance4_total = torch.cat([reliance4_r, reliance4_d], dim=1)                          # reliance4_total : (N, 2*NS, 1, 1)

        tortal_sampled_feature6 = tortal_sampled_feature6 * reliance4_total     # tortal_sampled_feature6 : (N, 2*NS, H/32, W/32)
        tortal_sampled_feature5 = tortal_sampled_feature5 * reliance4_total     # tortal_sampled_feature5 : (N, 2*NS, H/16, W/16)
        tortal_sampled_feature4 = tortal_sampled_feature4 * reliance4_total     # tortal_sampled_feature4 : (N, 2*NS, H/8, W/8)

        # Ground truth for RSM
        sum_RFF = torch.sum(reliance_fused_feature.detach(), dim=1, keepdim=True)   # sum_RFF : (N, 1, H/8, W/8)
        norm_RFF = reliance_fused_feature.clone().detach()                          # norm_RFF : (N, 2*NS, H/8, W/8)
        
        for i in range(sum_RFF.shape[0]):
            sum_RFF[i] -= torch.min(sum_RFF[i])
            sum_RFF[i] /= (torch.max(sum_RFF[i]) + 1e-8)

        reliability_GT = self.get_reliablity_GT(sum_RFF, norm_RFF)               # reliability_GT : (N, 2)

        for i in range(reliability_GT.shape[0]):
            reliability_GT[i] *= 1 / torch.sum(reliability_GT[i])
        #################################

        ### Decoder ###
        tortal_sampled_feature6 = self.total_fusion6(tortal_sampled_feature6)
        tortal_sampled_feature5 = self.total_fusion5(tortal_sampled_feature5)
        tortal_sampled_feature4 = self.total_fusion4(tortal_sampled_feature4)

        feat_56 = self.rgb_merge_co_56(torch.cat([tortal_sampled_feature5, resize(tortal_sampled_feature6, [22, 22])], dim=1))                # feat_56 : [N, 128, 14, 14]
        feat_45 = self.rgb_merge_co_45(torch.cat([tortal_sampled_feature4, resize(feat_56, [44, 44])], dim=1))                         # feat_45 : [N, 128, 28, 28]
        cosal_map_4 = self.rgb_get_pred_4(feat_45)                                                                                  # cosal_map_4 : [N, 1, 28, 28]

        feat_34, cosal_map_3 = self.rgb_refine_3(rgb_f3, cosal_map_4, feat_45)
        feat_23, cosal_map_2 = self.rgb_refine_2(rgb_f2, cosal_map_4, feat_34)
        _, cosal_map_1 = self.rgb_refine_1(rgb_f1, cosal_map_4, feat_23)                                                        # cosal_map_1 : [N, 1, 224, 224]

        preds_list = [resize(cosal_map_4), resize(cosal_map_3), resize(cosal_map_2), cosal_map_1]
        ###############
            
        return preds_list, rgb_sampler_block, depth_sampler_block, reliance_4, reliability_GT

if __name__ == '__main__':
    model = MyModel().cuda()
    rgb = torch.rand(5, 3, 352, 352).cuda()
    depth = torch.rand(5, 3, 352, 352).cuda()
    rgb_ssmap = torch.rand(5, 100, 352, 352).cuda()
    depth_ssmap = torch.rand(5, 100, 352, 352).cuda()

    out = model(rgb, depth, rgb_ssmap, depth_ssmap)