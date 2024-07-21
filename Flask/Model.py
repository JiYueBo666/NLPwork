import torch.nn as nn
import torch
from untils import *

class RNNdropout(nn.Dropout):
    def __init__(self,p):
        super().__init__()
        self.p=p
    def forward(self,batch_seq):
        #[batch_size,hidden_size]
        ones =batch_seq.data.new_ones(batch_seq.shape[0],batch_seq.shape[-1])
        mask=nn.functional.dropout(ones,self.p,training=self.training,inplace=False)
        mask=mask.unsqueeze(1)
        return mask*batch_seq

class Seq_2_Seq_encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Seq_2_Seq_encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.encoder=nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=True,num_layers=1)

    def forward(self,batch_seq,batch_lengths):
        sorted_batch,sorted_length,_,reverse_index=sort_by_seq_lens(batch_seq,batch_lengths)
        packed_batch=nn.utils.rnn.pack_padded_sequence(sorted_batch,sorted_length.cpu(),batch_first=True)
        output,_=self.encoder(packed_batch)
        unpacked_output,_=nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        restored_output=unpacked_output.index_select(0,reverse_index)
        return restored_output



class mask_Attention():

    def forward(self,premises_encoded,premises_mask,hypotheses_encoded,hypotheses_mask):
        similarity_matrix=premises_encoded.bmm(hypotheses_encoded.transpose(1,2).contiguous())

        prem_hyp_attn= masked_softmax(similarity_matrix, hypotheses_mask)
        hyp_prem_attn=masked_softmax(similarity_matrix.transpose(1,2).contiguous(),premises_mask)

        attended_premises = weighted_sum(hypotheses_encoded,
                                         prem_hyp_attn,
                                         premises_mask)
        attended_hypotheses = weighted_sum(premises_encoded,
                                           hyp_prem_attn,
                                           hypotheses_mask)

        return attended_premises, attended_hypotheses

class ESIM(nn.Module):
    def __init__(self,embedding_size,hidden_size,padding_idx=0,dropout=0.5):
        super(ESIM, self).__init__()
        self.hidden_size=hidden_size
        self.embedding = nn.Embedding(embedding_size, hidden_size)
        self.padding_idx=padding_idx
        self.dropout=dropout
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dropout is not None:
            self._dropout = RNNdropout(self.dropout)
        self.seq_encoder=Seq_2_Seq_encoder(self.hidden_size,self.hidden_size)
        self.composition=Seq_2_Seq_encoder(self.hidden_size,self.hidden_size)


        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       1),
                                             )
        self.mask_attention=mask_Attention()

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())
        self.probility_proj=nn.Sigmoid()

        self.loss_function = nn.BCELoss()
    def forward(self,premises,premises_lengths,hypotheses,hypotheses_lengths,label=None):
        seq_length=premises.shape[1]

        #获取掩码
        premises_mask=get_mask(premises,premises_lengths).to(self.device)
        hypotheses_mask=get_mask(hypotheses,hypotheses_lengths).to(self.device)

        #词嵌入
        embedd_premises=self.embedding(premises)
        embedd_hypotheses=self.embedding(hypotheses)

        #dropout
        if self.dropout is not None:
            embedd_premises=self._dropout(embedd_premises)
            embedd_hypotheses=self._dropout(embedd_hypotheses)

        #[batch,seq_len,hidden*2]
        encoded_premises=self.seq_encoder(embedd_premises,premises_lengths)
        encoded_hypotheses=self.seq_encoder(embedd_hypotheses,hypotheses_lengths)

        #attention
        attend_premises,attend_hypotheses=self.mask_attention.forward(encoded_premises,premises_mask,encoded_hypotheses,hypotheses_mask)

        #batch_size,seq_len,hidden*8
        enhanced_premises = torch.cat([encoded_premises,
                                       attend_premises,
                                       encoded_premises - attend_premises,
                                       encoded_premises * attend_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attend_hypotheses,
                                         encoded_hypotheses -
                                         attend_hypotheses,
                                         encoded_hypotheses *
                                         attend_hypotheses],
                                        dim=-1)
        #[batch_size,seq_len,hidden]
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._dropout(projected_premises)
            projected_hypotheses = self._dropout(projected_hypotheses)

        v_ai = self.composition(projected_premises, premises_lengths)
        v_bj = self.composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg=torch.nn.functional.avg_pool1d(v_ai.transpose(-1, -2),kernel_size=v_ai.shape[1]).squeeze()
        v_a_max=torch.nn.functional.max_pool1d(v_ai.transpose(-1, -2),kernel_size=v_ai.shape[1]).squeeze()

        v_b_avg=torch.nn.functional.avg_pool1d(v_bj.transpose(-1, -2),kernel_size=v_bj.shape[1]).squeeze()
        v_b_max=torch.nn.functional.max_pool1d(v_bj.transpose(-1, -2),kernel_size=v_bj.shape[1]).squeeze()


        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)

        probabilities =self.probility_proj(logits)
        if label is not None:
            loss=self.loss_function(probabilities,label.float())
            return loss

        return logits, probabilities