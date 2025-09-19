import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from model_new import *
from dataset import VTKG


class MMKGCAgent(nn.Module):
    '''
    1:KG
    '''

    def __init__(
            self,
            KG,  # 知识图谱，用来给gatv2conv提供参数
            num_ent,  # 实体数量
            num_rel,  # 关系数量
            ent_vis_mask,  # 实体视觉掩码
            ent_txt_mask,  # 实体文本掩码
            dim_str,  # 维度
            num_head,
            dim_hid,
            num_layer_enc_ent,  # 实体编码器中encoder layer层数
            num_layer_enc_rel,  # 关系编码器中encoder layer层数
            num_layer_dec,  # 解码器中encoder layer层数
            # 实体编码层/关系编码层/解码层各自的层数
            dropout=0.1,
            emb_dropout=0.6,
            vis_dropout=0.1,
            txt_dropout=0.1,
            visual_token_index=None,
            text_token_index=None,
            score_function="tucker"  # 评分函数

    ):
        super(MMKGCAgent, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        '''
        1
        '''
        self.KG = KG

        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/textual.pth")
        # 加载预训练的token嵌入
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        # 视觉token和文本token嵌入，创建一个embedding层，requires_grad_(False)表示嵌入层权重在训练过程中不被更新
        self.score_function = score_function

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        '''
        创建了一个形状为[self.num_ent, 1]的张量,这个张量中的所有元素都是布尔值False
        torch.cat(tensors, dim) 将多个张量沿着指定的维度dim连接起来。
        最终得到实体掩码和关系掩码
        '''
        false_ents = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim=1)
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim=1)

        '''
        创建形状为 [1, 1, dim_str] 的ent_token和rel_token
        创建了形状为 [num_ent, 1, dim_str] 的参数张量
        ent_embedding对应4.2.1中的se，是一个可学习的嵌入，表示实体的结构信息
        '''
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))

        '''
        ???
        '''
        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))

        '''
        实体嵌入/关系嵌入/视觉特征/文本特征的归一化层，dim_str表示该层维度
        '''
        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)
        '''
        嵌入层/视觉特征/文本特征的dropout层
        '''
        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        '''
        字符串实体/视觉实体/文本实体位置编码
        '''
        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        '''
        字符串关系/视觉关系/文本关系位置编码
        '''
        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        '''
        头实体/关系/尾实体的位置编码
        '''
        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim_str))
        '''
        投影层
        将视觉特征维度32和文本特征维度768映射到dim_str
        nn.Linear：全连接层（线性层）：对输入进行线性变换
        对应论文4.2.1部分(4)
        '''
        self.proj_ent_vis = nn.Linear(32, dim_str)
        self.proj_ent_txt = nn.Linear(768, dim_str)

        # self.proj_rel_vis = nn.Linear(dim_vis * 3, dim_str)

        '''
        dim_str:嵌入维度
        num_head:多头注意力头数
        dim_hid:隐藏层层数
        drop_out:dropout比率
        batch_first:指定输入数据和输出数据的维度顺序，为true，则第一个维度是批次大小
        '''
        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        '''
        实体编码器
        将上面的encoderlayer堆叠形成的transformer encoder实例，参数ent_encoder_layer, num_layer_enc_ent分别表示
        上面的encoder layer层和层数
        '''
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        # 关系编码器，同上
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        # 解码器，同上
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        '''
        12
        '''
        # 这里暂时先1层试试效果
        self.gat_conv = GATv2Conv(dim_str, dim_str // num_head, heads=num_head, dropout=dropout)
        self.gat_conv1 = GATConv(dim_str, dim_str // num_head, heads=num_head, dropout=dropout)
        self.gcn_conv = GCNConv(dim_str, dim_str)
        self.sage_conv = SAGEConv(dim_str, dim_str)
        mlp = nn.Sequential(
            nn.Linear(dim_str, dim_str // num_head),
            nn.ReLU(),
            nn.Linear(dim_str // num_head, dim_str)
        )
        self.gin_conv = GINConv(mlp)

        # self.gat_conv = GATv2Conv(dim_str, dim_str)
        self.res_linear = CrossInteractLayer(dim_str)

        self.contrastive = ContrastiveLoss(temp=0.5)  # 原代码
        self.contrastive_BarlowTwinsLoss = BarlowTwinsLoss(lambda_param=5e-3)
        # self.contrastive_BarlowTwinsContrastiveLoss=BarlowTwinsContrastiveLoss(temp=0.5,lambda_param=5e-3)

        self.num_con = 256  # 对比样本的数量
        self.num_vis = ent_vis_mask.shape[1]  # 视觉token数量
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        else:
            pass

        self.init_weights()
        torch.save(self.visual_token_embedding, open("visual_token.pth", "wb"))
        torch.save(self.text_token_embedding, open("textual_token.pth", "wb"))

    def init_weights(self):
        # Xavier 均匀初始化方法
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self):
        
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)     
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))
        ent_embs_ENT = torch.cat([ent_embs, self.lp_token], dim=0)
        ent_embs1 = self.gat_conv(ent_embs_ENT, self.KG.edge_index.cuda())
        #         ent_embs1 = self.gat_conv1(ent_embs_ENT, self.KG.edge_index.cuda())
        #         ent_embs1 = self.gcn_conv(ent_embs_ENT, self.KG.edge_index.cuda())
        #         ent_embs1 = self.sage_conv(ent_embs_ENT, self.KG.edge_index.cuda())
        #         ent_embs1 = self.gin_conv(ent_embs_ENT, self.KG.edge_index.cuda())
        return ent_embs1, rep_rel_str.squeeze(dim=1), ent_embs_ENT
    def contrastive_loss(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]
        emb_ent2 = torch.cat([ent_embs, self.lp_token], dim=0)
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]

        contrastive_loss = self.contrastive(emb_ent1[select_ents], emb_ent2[select_ents])
        # print(contrastive_loss)
        return contrastive_loss

    def contrastive_loss_finegrained(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        # 形成完整的实体序列

        # ent_embs: [ent_num, seq_len, embed_dim]
        # 这里与forward中不同，forward中做了[:,0]处理
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)

        # 将[ENT]（每个实体前的标记，汇聚了一个实体的信息）与特殊标记lp_token拼接
        emb_ent2 = torch.cat([ent_embs[:, 0], self.lp_token], dim=0)

        # 所有实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb3 = torch.cat([torch.mean(ent_embs, dim=1), self.lp_token], dim=0)

        # 计算视觉实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb4 = torch.cat([torch.mean(ent_embs[:, 2: 2 + self.num_vis, :], dim=1), self.lp_token], dim=0)

        # 计算非视觉实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb5 = torch.cat([torch.mean(ent_embs[:, 2 + self.num_vis: -1, :], dim=1), self.lp_token], dim=0)

        # ent_embs1= self.gat_conv(emb_ent2, self.KG.edge_index.cuda())

        # emb_ent2 = self.gat_conv(emb_ent2, self.KG.edge_index.cuda())
        # ent_emb3 = self.gat_conv(ent_emb3, self.KG.edge_index.cuda())
        # ent_emb4 = self.gat_conv(ent_emb4, self.KG.edge_index.cuda())
        # ent_emb5 = self.gat_conv(ent_emb5, self.KG.edge_index.cuda())

        # emb_ent1:形参，从一个批次的实体嵌入随机选择num_con个实体进行对比学习
        # select_ents是一个包含self.num_con个随机索引的数组
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]
        contrastive_loss = 0

        for emb in [emb_ent2, ent_emb3, ent_emb4, ent_emb5]:
            contrastive_loss += self.contrastive(emb_ent1[select_ents], emb[select_ents])
        contrastive_loss /= 4
        return contrastive_loss



    def contrastive_loss_BarlowTwinsLoss(self, emb_ent1):
        # emb_ent1
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        # 形成完整的实体序列

        # ent_embs: [ent_num, seq_len, embed_dim]
        # 这里与forward中不同，forward中做了[:,0]处理
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)

        # 将[ENT]（每个实体前的标记，汇聚了一个实体的信息）与特殊标记lp_token拼接
        emb_ent2 = torch.cat([ent_embs[:, 0], self.lp_token], dim=0)
        # 所有实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb3 = torch.cat([torch.mean(ent_embs, dim=1), self.lp_token], dim=0)
        # 计算视觉实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb4 = torch.cat([torch.mean(ent_embs[:, 2: 2 + self.num_vis, :], dim=1), self.lp_token], dim=0)
        # 计算非视觉实体嵌入的平均值与特殊标记lp_token拼接
        ent_emb5 = torch.cat([torch.mean(ent_embs[:, 2 + self.num_vis: -1, :], dim=1), self.lp_token], dim=0)

        # emb_ent1:形参，从一个批次的实体嵌入随机选择num_con个实体进行对比学习
        # select_ents是一个包含self.num_con个随机索引的数组
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]

        contrastive_loss = 0

        for emb in [emb_ent2, ent_emb3, ent_emb4, ent_emb5]:
            contrastive_loss += self.contrastive_BarlowTwinsLoss(emb_ent1[select_ents], emb[select_ents])

        contrastive_loss /= 4
        return contrastive_loss

    def score(self, emb_ent, emb_rel, triplets):
        # args:
        #   emb_ent: [num_ent, emb_dim]
        #   emb_rel: [num_rel, emb_dim]
        #   triples: [batch_size, 3]（头实体，关系，尾实体）triplets[:,0]：头实体；triplets[:,1]：关系；triplets[:,2]：尾实体
        # return:
        #   scores: [batch_size, num_ent]

        # 在三元组中，为便于索引和避免冲突，通常将关系的索引设置在 [0, num_rel) 范围，而将实体索引设置在 [num_rel, num_rel + num_ent) 范围。
        # emb_ent[triplets[:,0] - self.num_rel]是为了将三元组中头实体的索引从 [num_rel, num_rel + num_ent) 范围映射回 [0, num_ent)，从而从emb_ent正确提取实体嵌入，这里完成
        # 通过索引从num_ent个实体中选择该实体
        # unsqueeze(dim = 1):在该选定实体维度增加一个维度-》[1,1,emb_dim]
        # pos_head/pos_rel/pos_tail:[1,1,dim_str]
        h_seq = emb_ent[triplets[:, 0] - self.num_rel].unsqueeze(dim=1) + self.pos_head
        r_seq = emb_rel[triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel
        t_seq = emb_ent[triplets[:, 2] - self.num_rel].unsqueeze(dim=1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)
        output_dec = self.decoder(dec_seq)

        rel_emb = output_dec[:, 1, :]  # 提取关系嵌入
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]  # 对应论文4.2.2部分上下文嵌入
        # indexs = triplets != self.num_ent + self.num_rel
        # indexs[:, 1] = False
        # ent_emb = output_dec[indexs]
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)
            # emb_ent[:-1] 表示对 emb_ent 的切片操作，去除了最后一个实体嵌入，得到 [num_ent - 1, emb_dim] 的张量。
            # transpose(1, 0) 是一个转置操作，将张量的第 1 维（emb_dim）和第 0 维（num_ent - 1）交换。将 emb_ent[:-1] 从形状 [num_ent - 1, emb_dim] 转置为 [emb_dim, num_ent - 1]。
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
        else:
            # output_dec = self.decoder(dec_seq)
            score = torch.inner(ctx_emb, emb_ent[:-1])
        return score
        # 每个 score[i, j] 表示第 i 个三元组与第 j 个实体之间的得分。

