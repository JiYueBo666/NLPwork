import torch
def get_mask(batch_seq,batch_length):
    batch=batch_seq.shape[0]
    max_length=torch.max(batch_length)
    mask=torch.ones((batch,max_length),dtype=torch.float32)
    mask[batch_seq[:,:max_length]==0]=0.0
    return mask
def sort_by_seq_lens(batch_seq,batch_length,descending=True):

    sorted_length,sorting_idx=\
        batch_length.sort(0,descending=descending)
    if sorting_idx.ndim>1:
        sorting_idx=sorting_idx.reshape(-1)
    sorted_seq=batch_seq.index_select(0,sorting_idx)

    arrange_idx=\
        batch_length.new_tensor(torch.arange(0,len(batch_length)),device=batch_length.device)
    _,revert_idx=sorting_idx.sort(0,descending=False)
    if revert_idx.ndim>1:
        revert_idx=revert_idx.reshape(-1)
    restoration_idx=arrange_idx.index_select(0,revert_idx)

    return sorted_seq,sorted_length,sorting_idx,restoration_idx
def masked_softmax(tensor,mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = torch.nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def weighted_sum(value,attention_matrix,mask):
    weighted_sum = attention_matrix.bmm(value)
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    #mask=[batchsize,seqlength,1]
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


def replace_masked(tensor,mask,value):
    #[batch,seq_length]------>[batch,seq_length,1]
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask


    values_to_add = value * reverse_mask
    #[batch_size,hidden_size]*
    return tensor * mask + values_to_add












