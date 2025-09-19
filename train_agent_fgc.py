from dataset import VTKG
from agent_MMKGC import MMKGCAgent
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random
import distutils
import logging

from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens

OMP_NUM_THREADS = 8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(2024)
random.seed(2024)
np.random.seed(2024)
# 随机数种子，用以保证结果可复现性

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 用户运行脚本时可以通过命令行传递参数
    parser.add_argument('--data', default="MKG-W", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--valid_epoch', default=50, type=int)
    parser.add_argument('--exp', default='MMKGCAgent')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--emb_dropout', default=0.9, type=float)
    parser.add_argument('--vis_dropout', default=0.4, type=float)
    parser.add_argument('--txt_dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--max_img_num', default=3, type=int)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--max_vis_token', default=8, type=int)
    parser.add_argument('--max_txt_token', default=8, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)
    parser.add_argument('--mu', default=0, type=float)
    args = parser.parse_args()

    file_format = ""
    # 创建一个空字符串 file_format，将用于构建文件名格式
    for arg_name in vars(args).keys():
        if arg_name in ["lr", "hidden_dim", "batch_size", "num_epoch", "max_vis_token", "max_txt_token", "num_head",
                        "mu"]:
            file_format += f"{arg_name}_{vars(args)[arg_name]}"
        elif arg_name in ["score_function", "emb_dropout", "vis_dropout", "txt_dropout"]:
            file_format += f"{vars(args)[arg_name]}"

    if not args.no_write:
        os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./logs/{args.exp}/{args.data}", exist_ok=True)
        if not os.path.isfile(f"ckpt/{args.exp}/args.txt"):
            with open(f"ckpt/{args.exp}/args.txt", "w") as f:
                for arg_name in vars(args).keys():
                    if arg_name not in ["data", "exp", "no_write", "num_epoch", "cont", "early_stop"]:
                        f.write(f"{arg_name}\t{type(vars(args)[arg_name])}\n")
    else:
        file_format = None

    file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.data}/{file_format}.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"{os.getpid()}")
    logger.info(args)

    KG = VTKG(args.data, logger, max_vis_len=args.max_img_num)  # 每个实体对应的图片数量最大为3；args.data为数据集

    KG_Loader = torch.utils.data.DataLoader(KG, batch_size=args.batch_size, shuffle=True)

    '''
    初始化edge_index用来作为GATv2Conv参数
    '''
    # KG_Edge_Index=KG.build_edge_index()
    # KG_Edge_Index=KG_Edge_Index.cuda()

    visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    visual_token_index = visual_token_index.cuda()
    text_token_index, text_key_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)
    text_token_index = text_token_index.cuda()
    # logger.info(visual_token_index, text_token_index)
    logger.info("Visual tokens: %s, Text tokens: %s", visual_token_index, text_token_index)
    logger.info("Visual key mask: %s, Text key mask: %s", visual_key_mask, text_key_mask)
    model = MMKGCAgent(
        KG=KG,
        num_ent=KG.num_ent,  # 实体数量
        num_rel=KG.num_rel,  # 关系数量
        ent_vis_mask=visual_key_mask,  # 实体视觉token掩码
        ent_txt_mask=text_key_mask,  # 实体文本token掩码
        dim_str=args.dim,  # 嵌入维度
        num_head=args.num_head,  # 多头注意力头的个数，默认为2
        dim_hid=args.hidden_dim,  # 隐藏层维度
        num_layer_enc_ent=args.num_layer_enc_ent,
        # 实体编码器ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)的层数默认为1
        num_layer_enc_rel=args.num_layer_enc_rel,
        # 关系编码器rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)的层数默认为1
        num_layer_dec=args.num_layer_dec,
        # 解码器decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)的层数
        dropout=args.dropout,  # dropout层dropout率默认0.01，用来减少过拟合
        emb_dropout=args.emb_dropout,  # 默认0.9
        vis_dropout=args.vis_dropout,  # 默认0.4
        txt_dropout=args.txt_dropout,  # 默认0.1
        visual_token_index=visual_token_index,  # 视觉token索引
        text_token_index=text_token_index,  # 文本token索引
        score_function=args.score_function  # 评分函数
    ).cuda()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)  # 平滑标签的超参数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  # 正则化参数

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)  # 学习率调度器

    last_epoch = 0
    start = time.time()
    logger.info("EPOCH\tLOSS\tTOTAL TIME")

    '''
    生成所有实体的id和所有关系的id（索引）
    '''
    all_ents = torch.arange(KG.num_ent).cuda()
    all_rels = torch.arange(KG.num_rel).cuda()

    best_mrr = 0.0

    for epoch in range(last_epoch + 1, args.num_epoch + 1):
        total_loss = 0.0
        for batch, label in KG_Loader:  # 每次取出一个批次的的数据batch和标签label        return torch.tensor(masked_triplet), torch.tensor(label)
            ent_embs, rel_embs, ent_contrast = model()  # 分别是MMKGCAgent中的forward返回的return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1)
            scores = model.score(ent_embs, rel_embs, batch.cuda())
            loss = loss_fn(scores, label.cuda())

            if args.mu != 0:
                loss += model.contrastive_loss_BarlowTwinsLoss(ent_contrast) * args.mu  # GLF
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 对梯度进行裁剪，防止梯度爆炸，最大梯度范数为 0.1
            optimizer.step()  # 更新模型参数

        scheduler.step()  # 更新学习率调度器，按照预设的计划调整学习率

        logger.info(f"{epoch} \t {total_loss:.6f} \t {time.time() - start:.6f} s")
        if (epoch) % args.valid_epoch == 0:
            model.eval()  # 评估阶段
            with torch.no_grad():
                ent_embs, rel_embs, ent_contrast = model()
                lp_list_rank = []  # 初始化一个空的列表 lp_list_rank，用来保存链接预测的排名
                for triplet in tqdm(KG.valid):  # tqdm：显示进度条   验证集
                    h, r, t = triplet
                    '''
                    score函数参数中第三部分：对应model_MMKGCAgent.py文件中score函数中对h_seq/r_seq/t_seq的处理，进行ID偏移
                    '''
                    head_score = model.score(ent_embs, rel_embs, torch.tensor(
                        [[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[
                        (-1, r, t)])  # KG.filter_dict[(-1, r, t)] 是一个过滤字典，用于从三元组中过滤掉无关的实体。
                    # (-1, r, t) 表示过滤掉除当前头实体外的所有可能头实体
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor(
                        [[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])

                    lp_list_rank.append(head_rank)
                    lp_list_rank.append(tail_rank)
                    # lp_list_rank 用来存储所有三元组（头实体、关系、尾实体）的排名。
                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Validation Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

                lp_list_rank = []
                for triplet in tqdm(KG.test):  # 测试集
                    h, r, t = triplet
                    head_score = model.score(ent_embs, rel_embs, torch.tensor(
                        [[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor(
                        [[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])

                    lp_list_rank.append(head_rank)
                    lp_list_rank.append(tail_rank)

                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Test Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

            if best_mrr < mrr:
                best_mrr = mrr
                best_result = (mr, mrr, hit10, hit3, hit1)

            model.train()
            if (epoch) % 500 == 0:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    },
                    f"./ckpt/{args.exp}/{args.data}/{file_format}_{epoch}.ckpt"
                )

            model.train()

    logger.info("Done! {}. The best results are shown below:".format(args.data))
    logger.info(best_result)

