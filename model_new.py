import torch
import torch.nn as nn


class CrossInteractLayer(nn.Module):
    def __init__(self, dim_str):
        super(CrossInteractLayer, self).__init__()
        self.dim_str = dim_str
        self.dropout = nn.Dropout(0.5)
        self.fc_linear = nn.Linear(self.dim_str * 5, self.dim_str)

    def forward(self, x1, x2):
        res = torch.cat([x1, x2, x1 + x2, x1 - x2, x1 * x2], dim=1)
        res = self.fc_linear(res)
        res = self.dropout(res)
        return res


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)
        self.cross_interact = CrossInteractLayer(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_act = nn.ReLU()
        self.output_linear = nn.Linear(dim, dim)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x_residual = x  # 保存输入
        # 残差交互
        x = self.cross_interact(x, r_embed)
        x = x + x_residual  # 这里加上残差连接
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        x = x.view(-1, 1, x.size(1))
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        # 默认在最后一个维度计算余弦相似度

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
        # self.temp温度参数通常用于调整相似度的尺度


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor):
        # 沿batch维度对嵌入向量进行标准化
        emb1_norm = (emb1 - emb1.mean(0)) / emb1.std(0)
        emb2_norm = (emb2 - emb2.mean(0)) / emb2.std(0)
        batch_size = emb1.size(0)  # 256
        features = emb1.size(1)
        # 计算交叉相关矩阵
        cross_corr = torch.mm(emb1_norm.T, emb2_norm) / batch_size
        # 创建单位矩阵并计算差异
        corr_diff = (cross_corr - torch.eye(features, device=self.device)).pow(2)
        # 对非对角线元素应用缩放系数（lambda_param），从而减少不同特征之间的相似性，促进特征的独立性。
        corr_diff[~torch.eye(features, dtype=bool, device=self.device)] *= self.lambda_param
        # 计算并返回损失
        loss = corr_diff.mean()
        return loss
class VISTATucker(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            rel_vis,
            dim_vis,
            rel_txt,
            dim_txt,
            ent_vis_mask,
            ent_txt_mask,
            rel_vis_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout=0.1,
            emb_dropout=0.6,
            vis_dropout=0.1,
            txt_dropout=0.1,
            visual_token_index=None,
            text_token_index=None,
            score_function="tucker"
    ):
        super(VISTATucker, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.rel_vis = rel_vis
        self.rel_txt = None

        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding(num_embeddings=8193, embedding_dim=self.dim_str)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding(num_embeddings=15000, embedding_dim=self.dim_str)
        self.score_function = score_function

        false_ents = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim=1)
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim=1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim_str))

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.contrastive = ContrastiveLoss(temp=0.5)
        self.num_con = 512

        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        else:
            pass

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        # nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        # nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        # nn.init.xavier_uniform_(self.proj_txt.weight)
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

        nn.init.xavier_uniform_(self.visual_token_embedding.weight)
        nn.init.xavier_uniform_(self.text_token_embedding.weight)

        # self.proj_ent_vis.bias.data.zero_()
        # self.proj_rel_vis.bias.data.zero_()
        # self.proj_txt.bias.data.zero_()

    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent

        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(entity_visual_tokens)) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(entity_text_tokens)) + self.pos_txt_ent

        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]
        # rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))  # + self.pos_str_rel
        # rel_seq = torch.cat([rel_tkn, rep_rel_str], dim = 1)
        # rel_embs = self.rel_encoder(rel_seq, src_key_padding_mask = self.rel_mask)[:,0]
        return torch.cat([ent_embs, self.lp_token], dim=0), rep_rel_str.squeeze(dim=1)

    def contrastive_loss(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(entity_visual_tokens)) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(entity_text_tokens)) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)[:, 0]
        emb_ent2 = torch.cat([ent_embs, self.lp_token], dim=0)
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]

        contrastive_loss = self.contrastive(emb_ent1[select_ents], emb_ent2[select_ents])
        return contrastive_loss

    def score(self, emb_ent, emb_rel, triplets):
        # args:
        #   emb_ent: [num_ent, emb_dim]
        #   emb_rel: [num_rel, emb_dim]
        #   triples: [batch_size, 3]
        # return:
        #   scores: [batch_size, num_ent]
        h_seq = emb_ent[triplets[:, 0] - self.num_rel].unsqueeze(dim=1) + self.pos_head
        r_seq = emb_rel[triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel
        t_seq = emb_ent[triplets[:, 2] - self.num_rel].unsqueeze(dim=1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ent_emb = output_dec[triplets != self.num_ent + self.num_rel]
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ent_emb, rel_emb)
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
        else:
            output_dec = self.decoder(dec_seq)
            score = torch.inner(ent_emb, emb_ent[:-1])
        return score
