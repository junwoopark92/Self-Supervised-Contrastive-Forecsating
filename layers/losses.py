import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def temporal_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss#, a_loss


def instance_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    if reduction:
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    else:
        loss = (logits[:, i, B + i - 1].mean(dim=0) + logits[:, B + i, i].mean(dim=0)) / 2

    return loss#, a_loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, reduction=True):
    B = z1.size(0)
    if reduction:
        loss = torch.tensor(0., device=z1.device)
    else:
        loss = torch.zeros(B, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, reduction)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        d += 1
    return loss / d


def relative_mask(distance_matrix):
    same_label_mask = (distance_matrix == 0.0)
    relative_matrix = distance_matrix.masked_fill(same_label_mask, np.inf) # remove same label
    min_vals, _ = torch.min(relative_matrix, dim=1, keepdim=True)
    pos_mask = (relative_matrix == min_vals).float()
    neg_mask = torch.ones_like(relative_matrix) - same_label_mask.float()
    return pos_mask, neg_mask


def get_circle_embedding(N):
    index = np.arange(N)
    interval = 2 * np.pi / N
    theta = index * interval
    x = np.cos(theta)
    y = np.sin(theta)
    embeds = np.stack([x, y], axis=1)
    return embeds


def autocorr_mask_with_CI(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask

def autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=1, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def local_autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


class AutoCon(nn.Module):
    def __init__(self, batch_size, seq_len, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AutoCon, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.acf_values = torch.from_numpy(acf_values)
        self.seq_len = seq_len

    def local_contrastive_loss(self, features, labels):
        B, T, D = features.shape
        local_features = features.clone()  # Feature shape: (B, T, C)

        anchor_dot_contrast = torch.div(
            torch.bmm(local_features, local_features.transpose(2, 1)),  # Shape: (B, T, C) X (C, T, B) > (B, T, T)
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        local_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # Local Distance map shape: (B, T, T)
        local_distmap = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()
        local_distmap = self.acf_values[local_distmap.abs().long()].float().to(features.get_device())

        neg_mask = torch.scatter(
            torch.ones_like(local_distmap).to(features.get_device()),
            2,
            torch.arange(T).reshape(1, -1, 1).repeat(B,1,1).to(features.get_device()),
            0
        )

        self_mask = (local_distmap == 1.0)
        pos_mask = local_autocorr_mask(local_distmap, self_mask) + (neg_mask * self_mask)

        exp_local_logits = torch.exp(local_logits) * neg_mask  # denominator

        log_local_prob = local_logits - torch.log(exp_local_logits.sum(2, keepdim=True))  # (B, T, T) > (B, T ,1)

        mean_log_local_prob_pos = (local_distmap * pos_mask * log_local_prob).sum(2) / pos_mask.sum(2)

        local_loss = -(self.temperature / self.base_temperature) * mean_log_local_prob_pos

        return local_loss


    def avg_global_contrastive_loss(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        B, T, C = features.shape

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1)

        global_features = pooled_features.clone()

        anchor_dot_contrast = torch.div(
            torch.matmul(global_features, global_features.transpose(1, 0)),  # (B, C) X (C, B) > (B, B)
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # Global Distance map shape: (B, B)
        global_distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
        global_distmap = self.acf_values[global_distmap.abs().long()].float().to(device)

        neg_mask = torch.scatter(
            torch.ones_like(global_distmap),
            1,
            torch.arange(B).view(-1, 1).to(device),
            0
        )

        self_mask = (global_distmap == 1.0)
        pos_mask = autocorr_mask(global_distmap, self_mask) + (neg_mask * self_mask)

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        log_global_prob = global_logits - torch.log(exp_global_logits.sum(1, keepdim=True))  # (B, B) > (B ,1)

        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(1) \
                                   / pos_mask.sum(1)

        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, features, labels=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        B, I, D = features.shape
        feature_idxs = torch.rand(B, self.seq_len).argsort(-1)[:, :self.seq_len//3].to(features.get_device())
        selected_features = torch.gather(features, 1, feature_idxs.unsqueeze(-1).repeat(1, 1, D))

        local_loss = self.local_contrastive_loss(selected_features, feature_idxs)
        global_loss = self.avg_global_contrastive_loss(features, labels)

        return local_loss, global_loss


class AutoConCI(nn.Module):  # AutoCon Channel Independence (CI) version for Multivariate
    def __init__(self, batch_size, seq_len, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AutoConCI, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.acf_values = torch.from_numpy(acf_values)
        self.seq_len = seq_len

    def local_contrastive_loss(self, features, labels):
        BC, T, D = features.shape

        # Compute local representation similarities
        local_features = features.clone()  # BC, T, D

        anchor_dot_contrast = torch.div(
            torch.bmm(local_features, local_features.transpose(1, 2)),  # (BC, T, D) X (BC, D, T) > (BC, T, T)
            self.temperature)
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        local_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # tile mask
        local_distmap = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()  # (BC, T, T) ,  (C, L)
        local_distmap = local_distmap.reshape(self.batch_size, -1, T*T).cpu()
        acf_values = self.acf_values.unsqueeze(0).repeat(self.batch_size,1, 1)
        local_distmap = torch.gather(acf_values, 2, local_distmap).float().to(features.get_device())
        local_distmap = local_distmap.reshape(-1, T, T)

        neg_mask = torch.scatter(
            torch.ones_like(local_distmap).to(features.get_device()),
            2,
            torch.arange(T).reshape(1, -1, 1).repeat(BC, 1, 1).to(features.get_device()),
            0
        )

        self_mask = (local_distmap == 1.0)
        pos_mask = local_autocorr_mask(local_distmap, self_mask) + (neg_mask * self_mask)

        exp_local_logits = torch.exp(local_logits) * neg_mask  # denominator

        log_local_prob = local_logits - torch.log(exp_local_logits.sum(2, keepdim=True))  # (B, T, T) > (B, T ,1)

        mean_log_local_prob_pos = (local_distmap * pos_mask * log_local_prob).sum(2) / pos_mask.sum(2)

        local_loss = - (self.temperature / self.base_temperature) * mean_log_local_prob_pos

        return local_loss

    def avg_global_contrastive_loss(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        BC, T, D = features.shape

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1) # (BC, D)

        # Compute global representation similarities
        global_features = pooled_features.reshape(self.batch_size, -1, D).permute(1, 0, 2).clone()
        C, B, D = global_features.shape

        anchor_dot_contrast = torch.div(
            torch.bmm(global_features, global_features.transpose(1, 2)),  # (C, B, D) X (C, D, B) > (C, B, B)
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # tile mask
        global_distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()  #(B, B)
        global_distmap = global_distmap.reshape(1, B * B).repeat(C, 1).cpu()
        global_distmap = torch.gather(self.acf_values,1, global_distmap).float().to(device)
        global_distmap = global_distmap.reshape(C, B, B)

        neg_mask = torch.scatter(
            torch.ones_like(global_distmap),
            2,
            torch.arange(B).reshape(1, -1, 1).repeat(C, 1, 1).to(device),
            0
        )

        self_mask = (global_distmap == 1.0)

        pos_mask = autocorr_mask_with_CI(global_distmap, self_mask) + (neg_mask * self_mask)

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        log_global_prob = global_logits - torch.log(exp_global_logits.sum(2, keepdim=True))  # (C, B, B) > (C, B ,1)


        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(2) \
                                   / pos_mask.sum(2)


        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, features, labels=None):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        B, I, D = features.shape
        feature_idxs = torch.rand(B, self.seq_len).argsort(-1)[:, :self.seq_len//3].to(features.get_device())
        selected_features = torch.gather(features, 1, feature_idxs.unsqueeze(-1).repeat(1, 1, D))

        local_loss = self.local_contrastive_loss(selected_features, feature_idxs)
        global_loss = self.avg_global_contrastive_loss(features, labels)

        return local_loss, global_loss


if __name__ == '__main__':
    batch_size = 5
    n_view = 2
    seq_len = 4
    dim = 3
    # features = torch.rand((batch_size, n_view, seq_len, dim)) # (Batch_size, N_view, T, Dim)
    features = torch.rand((batch_size, seq_len, dim)) # (Batch_size, T, Dim)
    labels = torch.tensor([0, 2, 3, 4, 6])
    distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()

    acf_values = np.array([1.0, 0.3, 0.1, -0.3, 0.2, 0.4, 0.1, 0.0])
    print(acf_values[distmap])
    supcreg = AutoCon(acf_values, contrast_mode='all')#.cuda()
    local_loss, global_loss = supcreg(features.cuda(), labels.cuda())
    print(local_loss.shape, global_loss.shape)