from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from transformers import BertForMaskedLM
from torch.nn import CrossEntropyLoss

class ScaledDotAttention(nn.Module):
    def __init__(self, d_k):
        """d_k: attention 的维度"""
        super(ScaledDotAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, mask):

        score = torch.einsum("nqd,nkd->nqk", [q, k]) / np.sqrt(self.d_k)

        if mask is not None:
            # 将mask为0的值，填充为负无穷，则在softmax时权重为0（被屏蔽的值不考虑）
            score.masked_fill_(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(score, -1)
        return attn

class BilinearSeqAttn(nn.Module):

    def __init__(self, d_model=768):
        super(BilinearSeqAttn, self).__init__()
        self.d_model = d_model  # 等于embedding_dim
        self.d_k = d_model
        self.scaled_dot_attn = ScaledDotAttention(self.d_k)
        self.W_Q = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_k, self.d_k, bias=False)

    def forward(self, query, key, mask):

        query = self.W_Q(query)
        key = self.W_K(key)

        attn = self.scaled_dot_attn(query, key, mask)  # nhqk
        return attn

class BaselineBert(nn.Module):
    def __init__(self,tokenizer, pretrained_dir):
        super().__init__()
        self.text_encoder = BertForMaskedLM.from_pretrained(pretrained_dir)
        hs = self.text_encoder.config.hidden_size
        self.start_attn = BilinearSeqAttn(hs)
        self.end_attn = BilinearSeqAttn(hs)
        self.tokenizer = tokenizer

    def forward(self, batch,train=True):

        # text_inputs = batch['text_inputs']
        output = self.text_encoder(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        token_type_ids=batch['token_type_ids'],
                                        output_hidden_states=True,
                                        return_dict=True)
        hidden_state = output.hidden_states[-1]
        # loss = 0

        # Table prompt task
        # prompt = batch['prompt']
        
        # # if train:
        # loss_fn=CrossEntropyLoss()
        # loss_start=loss_fn(start_span.transpose(-2,-1),tgt_start)
        # loss_end=loss_fn(end_span.transpose(-2,-1),tgt_end)
        # loss=loss_start+loss_end
        if train:
            # prompt_masked_positions_ = batch['prompt_masked_positions_']
            # prompt_attention_masks_ = batch['prompt_attention_masks_']
            loss=0
            predictions=[]
            # prompt = batch['prompt']
            prompt_masked_positions = batch['prompt_masked_positions_']
            prompt_label_positions = batch['label_positions']
            prompt_attention_mask = batch['prompt_attention_mask_']
            span = torch.stack([hidden_state[bs,index] for bs, index in enumerate(prompt_masked_positions)], dim=0)
            start_span = self.start_attn(span, hidden_state, prompt_attention_mask!=1)
            end_span = self.end_attn(span, hidden_state, prompt_attention_mask!=1)
            prediction = torch.cat((start_span.unsqueeze(2),end_span.unsqueeze(2)), dim=2)
            bs, _, _, seq = prediction.size()
            loss_prompt = F.cross_entropy(prediction.view(-1, seq), prompt_label_positions.flatten(), ignore_index=-100)
            loss += loss_prompt
            # for i in range(prompt_masked_positions_.shape[1]):
            #     span = torch.stack([hidden_state[bs,index] for bs, index in enumerate(prompt_masked_positions_[:,i])], dim=0)
            #     prompt_attention_mask_=prompt_attention_masks_[:,i,:]
            #     # for prompt_mask in prompt_attention_mask_:
            #     #     if sum(prompt_mask==0)==prompt_attention_masks_[:,i,:][0].shape[0]:
            #     #         print('debug')
            #     # hidden_state=hidden_state.unsqueeze(1).repeat(1,prompt_attention_masks_.shape[1],1,1)
            #     start_span = self.start_attn(span.unsqueeze(1), hidden_state, prompt_attention_mask_!=1)
            #     end_span = self.end_attn(span.unsqueeze(1), hidden_state, prompt_attention_mask_!=1)
            #     tgt_start=batch['tgt_starts'][:,i].unsqueeze(1)
            #     tgt_end=batch['tgt_ends'][:,i].unsqueeze(1)
            #     predictions.append(torch.cat((start_span.unsqueeze(2),end_span.unsqueeze(2)), dim=2))
            #     loss_fn=CrossEntropyLoss(ignore_index=-100)
            #     loss_start=loss_fn(start_span.transpose(-2,-1),tgt_start)
            #     loss_end=loss_fn(end_span.transpose(-2,-1),tgt_end)
            #     loss+=loss_start+loss_end
            output = {}
            output['predictions'] = prediction
            output['loss'] = loss
            # output['number'] = (batch['tgt_starts']!=0).sum().tolist()
            return output
        else:
            prompt_masked_positions_ = batch['prompt_masked_positions_']
            prompt_attention_mask_ = batch['prompt_attention_mask_']
            span = torch.stack([hidden_state[bs,index] for bs, index in enumerate(prompt_masked_positions_)], dim=0)
            start_span = self.start_attn(span, hidden_state, prompt_attention_mask_!=1)
            end_span = self.end_attn(span, hidden_state, prompt_attention_mask_!=1)
            prediction = torch.cat((start_span.unsqueeze(2),end_span.unsqueeze(2)), dim=2)
            output = {}
            output['predictions'] = prediction
            # output['loss'] = loss
            return output
            # prompt_masked_positions = batch['prompt_masked_positions']
            # prompt_offset = batch['offsets']
            # prompt_attention_mask = batch['prompt_attention_mask']
            # prompt_hidden_states=torch.gather(hidden_state,1,prompt_offset.unsqueeze(2).repeat(1, 1, hidden_state.size()[2]))
            # span=torch.gather(prompt_hidden_states,1,prompt_masked_positions.view(-1, 1).unsqueeze(2).repeat(1, 1, prompt_hidden_states.size()[2]))
            # # input_token_tensor = torch.gather(hidden_states, 1, input_id.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()
            # start_span = self.start_attn(span, prompt_hidden_states, prompt_attention_mask!=1)
            # end_span = self.end_attn(span, prompt_hidden_states, prompt_attention_mask!=1)
            # prediction = torch.cat((start_span.unsqueeze(2),end_span.unsqueeze(2)), dim=2)
    
    
