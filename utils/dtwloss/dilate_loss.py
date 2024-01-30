import torch
from utils.dtwloss import soft_dtw
from utils.dtwloss import path_soft_dtw


def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal



if __name__ == '__main__':
    gt_seq = torch.arange(10).reshape(2, 5, 1)
    pred_seq = torch.arange(10).reshape(2, 5, 1)
    pred_seq2 = torch.arange(1, 11).reshape(2, 5, 1)

    exact_match = dilate_loss(pred_seq, gt_seq, 0.5, 0.01, 'cpu')
    shift_match = dilate_loss(pred_seq2, gt_seq, 0.5, 0.01, 'cpu')
    print(exact_match[0].item())
    print(exact_match, shift_match)

