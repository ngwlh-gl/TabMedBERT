"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0
"""

import argparse, json, logging, numpy, os, random, sys, torch, math
from collections import Counter
from transformers import BertTokenizer,AdamW,get_linear_schedule_with_warmup
from ours_model import BaselineBert
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
import random
from tensorboardX import SummaryWriter
import numpy as np
import re
import shutil,glob
logger = logging.getLogger(__name__)

"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0

The code in this file is partly based on the FLAIR library,
(https://github.com/flairNLP/flair), licensed under the MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""
random.seed(42)

class dataset(torch.utils.data.Dataset):
    def __init__(self, args,file,mode='train'):
        # self.name = name
        self.sentences = []
        self.ner_label_list=args.ner_label_list
        self.task=args.task
        self.max_len=args.max_position_embeddings
        # revise_file=args.revise_filepath
        self.tokenizer=args.tokenizer
        self.sos_id=self.tokenizer.convert_tokens_to_ids(['[sos]'])[0]
        count_gold_single=0
        total_sentences_num=0
        self.max_prompt_times=args.max_prompt_times
        self.pos_spans_num=0
        self.neg_spans_num=0
        self.pos_first_spans_num=0
        self.pos_no_first_spans_num=0
        self.first_neg_num=0
        self.not_first_neg_num=0
        self.mode=mode
        self.use_neg=args.use_neg
        self.use_pos_prompt=args.use_pos_prompt
        self.use_only_one_pos=args.use_only_one_pos
        self.neg_ratio=args.neg_ratio
        self.only_use_first=args.only_use_first
        self.pos_samples=[]
        self.neg_samples=[]
        self.first_pos_samples=[]
        self.not_first_pos_samples=[]
        self.first_neg_samples=[]
        self.not_first_neg_samples=[]
        # if args.task=='cdr':
        prompt=['is','associated','with']
        
        if file is not None:
            with open(file, encoding="utf-8") as f:
                for line in f:
                    con=json.loads(line)
                    sentences=con['sentences']
                    gold_ner=con['ner']
                    if args.task=='tbga':
                        self.get_tbga_samples(gold_ner,con,sentences,prompt)
                    elif args.task in ['ade','cdr','biored','pubmed','chr','drugprot']:
                        self.get_samples(con,prompt)
            if self.only_use_first:
                one_sample_pos_num=len(self.first_pos_samples)
                if self.mode=='train':
                    one_sample_neg_num=math.floor(one_sample_pos_num*self.neg_ratio)
                    random.shuffle(self.first_neg_samples)
                    self.first_neg_samples=self.first_neg_samples[:one_sample_neg_num]
                samples=self.first_pos_samples+self.first_neg_samples
                for sample in samples:
                    mode=sample['mode']
                    item=sample['item']
                    self.append_sample(mode,item)
            else:
                one_sample_pos_num=len(self.pos_samples)
                if self.mode=='train':
                    one_sample_neg_num=math.floor(one_sample_pos_num*self.neg_ratio)
                    random.shuffle(neg_samples)
                    neg_samples=self.neg_samples[:one_sample_neg_num]
                samples=self.pos_samples+self.neg_samples
                for sample in samples:
                    mode=sample['mode']
                    item=sample['item']
                    self.append_sample(mode,item)
                    
            logger.info("Load %s sentences from %s" % (len(self.sentences), file))
            print("Load %s sentences from %s" % (len(self.sentences), file))
            if args.task!='tbga':
                ratio = count_gold_single/total_sentences_num if total_sentences_num!=0 else 0
                print("Only one entity in sentence : {}, ratio is {}, Total sentence number : {}".format(count_gold_single,ratio,total_sentences_num))
    def get_tbga_samples(self,gold_ner,con,sentences,prompt):
        number=random.random()
        head=gold_ner[0][0]
        tail=gold_ner[0][1]
        sent=sentences[0]
        flag=True
        if con['relations'][0][0][-1]=='NA':
            flag=False
        if number<0.5:
            prompt_=['Gene']+sent[head[0]:head[1]+1]+prompt+['Disease']+['[sos]']+['.']
            label='Disease'
            start=tail[0]
            end=tail[1]
        else:
            prompt_=['Disease']+sent[tail[0]:tail[1]+1]+prompt+['Gene']+['[sos]']+['.']
            label='Gene'
            start=head[0]
            end=head[1]
        i=0
        if flag:
            self.sentences.append({'sent':sent,'start':start,'end':end,'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
    
    def append_sample(self,mode,item):
        if mode=='first-pos':
            self.pos_first_spans_num+=1
            self.pos_spans_num+=1
        elif mode=='not-first-pos':
            self.pos_no_first_spans_num+=1
            self.pos_spans_num+=1
        elif mode=='neg-after-pos':
            self.neg_spans_num+=1
            self.not_first_neg_num+=1
        else:
            self.neg_spans_num+=1
            self.first_neg_num+=1
        
        self.sentences.append({'sent':item['sent'],'starts':item['starts'],'ends':item['ends'],'label':item['label'],'doc_key':item['doc_key'],'idx':item['idx'],'prompt':item['prompt_'],'mask_starts':item['mask_starts'],'mask_ends':item['mask_ends'],'mode':mode})
    def check(self,exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item):
        prompt_=item['prompt_']
        mask_starts=item['mask_starts']
        mask_ends=item['mask_ends']
        starts=item['starts']
        ends=item['ends']
        if (prompt_ in exist_sentences and mask_starts in exist_mask_starts and mask_ends in exist_mask_ends and starts in exist_starts and ends in exist_ends) or len(prompt_)>150:
            return False
        else:
            exist_sentences.append(prompt_[:])
            exist_mask_starts.append(mask_starts[:])
            exist_mask_ends.append(mask_ends[:])
            exist_starts.append(starts[:])
            exist_ends.append(ends[:])
            return True
    def get_samples(self,con,prompt):
        total_len=0
        if self.mode!='test':
            for i,(tags,sent,relation) in enumerate(zip(con['ner'],con['sentences'],con['relations'])):
                self.get_one_samples(con['doc_key'],tags,sent,relation,prompt,total_len,i)
                total_len+=len(sent)
        else:
            for i,(tags,sent,relation) in enumerate(zip(con['predicted_ner'],con['sentences'],con['relations'])):
                self.get_one_samples(con['doc_key'],tags,sent,relation,prompt,0,i)
                total_len+=len(sent)
    def get_one_samples(self,doc_key,ner,sent,relation,prompt,length,idx):
        # tags=gold_ner
        # sent=sentences
        # relations=con['relations']
        tags=[]
        for tag in ner:
            tags.append([tag[0]-length,tag[1]-length,tag[-1]])
        relations=[]
        for rel in relation:
            relations.append([rel[0]-length,rel[1]-length,rel[2]-length,rel[3]-length,rel[-1]])
        neg_samples=[]
        pos_samples=[]
        first_pos_samples=[]
        first_neg_samples=[]
        # if con['doc_key']=='3750012':
        #     print('debug')
        rels=dict()
        rels_dic={}
        rels_to_labels={}
        
        for tag in tags:
            if (tag[0],tag[1]) not in rels_to_labels:
                rels_to_labels[(tag[0],tag[1])]=tag[2]
        entity_not_in_relations={}
        entity_in_relations=[]
        for rel in relations:
            subj=(rel[0],rel[1])
            obj=(rel[2],rel[3])
            entity_in_relations.append((rel[0],rel[1]))
            entity_in_relations.append((rel[2],rel[3]))
            rels[(subj,obj)]=rel[4]
            if subj not in rels_dic:
                rels_dic[subj]=[]
            if obj not in rels_dic:
                rels_dic[obj]=[]
            rels_dic[subj].append(obj)
            rels_dic[obj].append(subj)
        if self.task in ['cdr','biored','pubmed','chr','drugprot']:
            final_rels_dic={}
            for key,values in rels_dic.items():
                values.sort()
                text_to_spans={}
                for span in values:
                    text=' '.join(sent[span[0]:span[1]+1]).lower()
                    if text not in text_to_spans:
                        text_to_spans[text]=[]
                    text_to_spans[text].append(span)
                final_rels_dic[key]=[v[0] for k,v in text_to_spans.items()]
            rels_dic=final_rels_dic

        for entity in tags:
            if (entity[0],entity[1]) not in entity_in_relations:
                entity_not_in_relations[(entity[0],entity[1])]=entity[2]
        text_to_spans={}
        exist_samples=[]
        exist_sentences=[]
        exist_mask_starts=[]
        exist_mask_ends=[]
        exist_starts=[]
        exist_ends=[]
        exist_entities=[]
        pos_prompts=[]
        neg_prompts=[]
        # 正例
        for subj in tags:
            for obj in tags:
                if subj[-1]=='ADE':
                    subj[-1]='Adverse-Effect'
                if obj[-1]=='ADE':
                    obj[-1]='Adverse-Effect'
                if subj==obj:
                    continue
                subj_label=subj[-1]
                obj_label=obj[-1]
                tuple_subj=(subj[0],subj[1])
                tuple_obj=(obj[0],obj[1])
                
                # if (tuple_subj,tuple_obj) in exist_samples or (tuple_subj,tuple_obj) in exist_samples:
                #     continue
                if (tuple_subj,tuple_obj) in rels or (tuple_obj,tuple_subj) in rels:
                    exist_samples.append((tuple_subj,tuple_obj))
                    exist_samples.append((tuple_obj,tuple_subj))
        
                    head=tuple_subj
                    tail=tuple_obj
                    # if number<0.5:
                    objs=rels_dic[tuple_subj]
                    prompt_=[]
                    mask_starts=[]
                    mask_ends=[]
                    last_start=-1
                    last_end=-1
                    # if ' '.join(sent[head[0]:head[1]+1]).lower() in exist_entities:
                    #     continue
                    # else:
                    #     exist_entities.append(' '.join(sent[head[0]:head[1]+1]).lower())
                    for i,o in enumerate(objs):
                        starts=[]
                        ends=[]
                        labels=[]
                        if o not in rels_to_labels:
                            continue
                        o_label=rels_to_labels[o]
                        prompt_=[subj_label]+sent[head[0]:head[1]+1]+prompt+[o_label]+['[sos]']+['.']+['[SEP]']
                        labels.append(o_label)
                        starts.append(o[0])
                        ends.append(o[1])
                        mask_starts.append(last_start)
                        mask_ends.append(last_end)
                        # mask_starts.append(-1)
                        # mask_ends.append(-1)
                        last_start=o[0]
                        last_end=o[1]
                        item={'sent':sent,'doc_key':doc_key,'starts':starts[:],'ends':ends[:],'label':labels[:],'idx':idx,'prompt_':prompt_,'mask_starts':mask_starts[:],'mask_ends':mask_ends[:]}
                        if i==0:
                            mode='first-pos'
                        else:
                            mode='not-first-pos'
                        if self.check(exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item):
                            pos_samples.append({'mode':mode,'item':item})
                            self.pos_samples.append({'mode':mode,'item':item})
                            if mode=='first-pos':
                                first_pos_samples.append({'mode':mode,'item':item})
                                self.first_pos_samples.append({'mode':mode,'item':item})
                            else:
                                self.not_first_pos_samples.append({'mode':mode,'item':item})
                            pos_prompts.append(prompt_)

                        # self.append_sample(mode,exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item)
                    # 这个针对 "一对多" 的情况，忽略
                    if self.use_pos_prompt:
                    # while len(mask_starts)<self.max_prompt_times:
                        starts=[]
                        ends=[]
                        labels=[]
                        prompt_=[subj_label]+sent[head[0]:head[1]+1]+prompt+[o_label]+['[sos]']+['.']+['[SEP]']
                        labels.append(o_label)
                        starts.append(len(sent))
                        ends.append(len(sent))
                        mask_starts.append(last_start)
                        mask_ends.append(last_end)
                        # mask_starts.append(-1)
                        # mask_ends.append(-1)
                        last_start=o[0]
                        last_end=o[1]
                        item={'sent':sent,'doc_key':doc_key,'starts':starts[:],'ends':ends[:],'label':labels[:],'idx':idx,'prompt_':prompt_,'mask_starts':mask_starts[:],'mask_ends':mask_ends[:]}
                        mode='neg-after-pos'
                        if self.check(exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item):
                            self.neg_samples.append({'mode':mode,'item':item})
                            neg_samples.append({'mode':mode,'item':item})
                            self.not_first_neg_samples.append({'mode':mode,'item':item})
                        # self.append_sample(mode,exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item)

        # 负例
        if self.use_neg:
            for k,v in entity_not_in_relations.items():
                # v=rels_to_labels[k]
                if self.task=='biored':
                    s_labels=self.ner_label_list
                else:
                    s_labels=list(set(self.ner_label_list)-set([v]))
                random.shuffle(s_labels)
                if v=='ADE':
                    v='Adverse-Effect'
                max_prompt_times=self.max_prompt_times if len(s_labels)>self.max_prompt_times else len(s_labels)
                s_labels=s_labels[:max_prompt_times]
                mask_starts=[]
                mask_ends=[]
                prompt_=[]
                last_start=-1
                last_end=-1
                # if ' '.join(sent[k[0]:k[1]+1]).lower() in exist_entities:
                #     continue
                # else:
                #     exist_entities.append(' '.join(sent[k[0]:k[1]+1]).lower())
                for i,s_label in enumerate(s_labels):
                    starts=[]
                    ends=[]
                    labels=[]
                    prompt_=[v]+sent[k[0]:k[1]+1]+prompt+[s_label]+['[sos]']+['.']+['[SEP]']
                    labels.append(s_label)
                    starts.append(len(sent))
                    ends.append(len(sent))
                    mask_starts.append(last_start)
                    mask_ends.append(last_end)
                    # mask_starts.append(-1)
                    # mask_ends.append(-1)
                    last_start=-1
                    last_end=-1
                    item={'sent':sent,'doc_key':doc_key,'starts':starts[:],'ends':ends[:],'label':labels[:],'idx':idx,'prompt_':prompt_,'mask_starts':mask_starts[:],'mask_ends':mask_ends[:]}
                    mode='other'
                    if self.check(exist_sentences,exist_mask_starts,exist_mask_ends,exist_starts,exist_ends,item):
                        neg_samples.append({'mode':mode,'item':item})
                        self.neg_samples.append({'mode':mode,'item':item})
                        first_neg_samples.append({'mode':mode,'item':item})
                        self.first_neg_samples.append({'mode':mode,'item':item})
                        neg_prompts.append(prompt_)
                   

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        data=self.sentences[idx]
        sent=data["sent"]
        doc_key=data['doc_key']
        # if doc_key=='11250767':
        #     print('debug')
        index=data['idx']
        prompt=data['prompt']
        starts=data['starts']
        mode=data['mode']
        ends=data['ends']
        pairs_to_idxs={}
        starts_dict={}
        ends_dict={}
        for i,(start,end) in enumerate(zip(starts,ends)):
            pairs_to_idxs[i]=[start,end]
            starts_dict[start]=i
            ends_dict[end]=i
        total_sentences=['[CLS]']+sent+['[SEP]']+prompt+['[SEP]']
        prompt_len=0
        for token in prompt:
            sub_tokens=self.tokenizer.tokenize(token)
            prompt_len+=len(sub_tokens)
        start_idxs={}
        # end_idxs={}
        count=0
        count_word=0
        prompt_masked_positions=[]
        prompt_masked_positions_=[]
        # tgt_starts=[None for _ in range(self.max_prompt_times)]
        # tgt_ends=[None for _ in range(self.max_prompt_times)]
        tgt_starts=[]
        tgt_ends=[]
        token_ids=[]
        attention_mask=[]
        token_type_ids=[]
        prompt_attention_mask=[]
        prompt_attention_mask_=[]
        sep_pos=[]
        token_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(0)
        attention_mask.append(1)
        offset={}
        offset[count_word]=count
        prompt_attention_mask.append(1)
        prompt_attention_mask_.append(1)
        # prompt_attention_mask.append(0)
        # prompt_attention_mask_.append(0)
        count+=1
        count_word+=1
        tokens=[]
        tokens+=['[CLS]']
        # count_tokens=0
        for i,token in enumerate(sent):
            sub_tokens=self.tokenizer.tokenize(token)
            if count+len(sub_tokens)>=self.max_len-prompt_len-2:
                break
            # count_tokens+=1
            tokens+=sub_tokens
            sub_tokens_ids=self.tokenizer.convert_tokens_to_ids(sub_tokens)
            token_ids+=sub_tokens_ids
            attention_mask+=[1]*len(sub_tokens_ids)
            prompt_attention_mask+=[1]
            prompt_attention_mask_+=[1]*len(sub_tokens_ids)
            token_type_ids+=[0]*len(sub_tokens_ids)
            offset[count_word]=count
            if i in starts_dict:
                pair_index=starts_dict[i]
                pairs_to_idxs[pair_index][0]=count
            # if i in data['starts']:
            #     idx=data['starts'].index(i)
            #     tgt_starts[idx]=count
            for _ in range(len(sub_tokens_ids)):
                start_idxs[count]=i
                count+=1
            # if i in data['ends']:
            #     idx=data['ends'].index(i)
            #     tgt_ends[idx]=count
            if i in ends_dict:
                pair_index=ends_dict[i]
                pairs_to_idxs[pair_index][1]=count
            count_word+=1
        token_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        prompt_attention_mask.append(0)
        prompt_attention_mask_.append(0)
        # prompt_attention_mask.append(1)
        # prompt_attention_mask_.append(1)
        token_type_ids.append(0)
        offset[count_word]=count
        tokens+=['[SEP]']
        # start_idxs[count]=start_idxs[count-1]+1
        # sep_pos.append(count)
        count+=1
        count_word+=1
        prompt_tokens_len=0
        prompt_tokens=[]
        prompt_tokens_ids=[]
        token_type_id=1
        for i,token in enumerate(prompt):
            sub_tokens=self.tokenizer.tokenize(token)
            tokens+=sub_tokens
            prompt_tokens+=sub_tokens
            sub_tokens_ids=self.tokenizer.convert_tokens_to_ids(sub_tokens)
            token_ids+=sub_tokens_ids
            prompt_tokens_ids+=sub_tokens_ids
            attention_mask+=[1]*len(sub_tokens_ids)
            prompt_attention_mask+=[0]
            prompt_attention_mask_+=[0]*len(sub_tokens_ids)
            token_type_ids+=[token_type_id]*len(sub_tokens_ids)
            offset[count_word]=count
            for token_id in sub_tokens_ids:
                if token_id==self.sos_id:
                    prompt_masked_positions.append(count_word)
                    prompt_masked_positions_.append(count)
                count+=1
            count_word+=1
            
        self.sentences[idx]['start_idxs']=start_idxs
        # self.sentences[idx]['end_idxs']=end_idxs
        assert len(pairs_to_idxs)==len(prompt_masked_positions_)
        # if len(pairs_to_idxs)>1:
        #     print('debug')
        for k,v in pairs_to_idxs.items():
            if v==[len(sent),len(sent)]:
                # v=[sep_pos[0],sep_pos[0]]
                v=[0,0]
                pairs_to_idxs[k]=v
            elif v[0] not in start_idxs:
                # v=[sep_pos[0],sep_pos[0]]
                v=[0,0]
                pairs_to_idxs[k]=v
        
            tgt_starts.append(v[0])
            tgt_ends.append(v[1])
        prompt_attention_masks_=[]
        # accumulate_mask_pos=[]
        # prompt_attention_masks_.append(prompt_attention_mask_[:])
        mask_starts=data['mask_starts']
        mask_ends=data['mask_ends']
        sub_attention_mask_=prompt_attention_mask_[:]
        for i,(mask_start,mask_end) in enumerate(zip(mask_starts,mask_ends)):
            if mask_start!=-1 and mask_end!=-1 and mask_end<len(start_idxs):
                sub_attention_mask_[mask_start:mask_end+1]=[0]*(mask_end+1-mask_start)
            
        prompt_attention_masks_.append(sub_attention_mask_)
        prompt_attention_mask_=sub_attention_mask_[:]
        
        assert len(token_ids)==len(attention_mask)==len(token_type_ids)==len(prompt_attention_mask_)==len(prompt_attention_masks_[0])
        assert len(prompt_attention_mask)==len(offset)
        label_positions=[]
        for start,end in zip(tgt_starts,tgt_ends):
            label_positions.append([start,end])
        # convert tokens to ids
        item={}
        item["input_ids"]=token_ids
        # item["labels"]=torch.tensor(labels).unsqueeze(0)
        item["attention_mask"] = attention_mask
        item["token_type_ids"] = token_type_ids
        item["prompt_attention_mask"]=prompt_attention_mask
        item["prompt_attention_mask_"]=prompt_attention_mask_
        item["prompt_masked_positions"]=prompt_masked_positions
        item["prompt_masked_positions_"]=prompt_masked_positions_
        item["prompt_attention_masks_"]=prompt_attention_masks_
        item['offset']=list(offset.values())
        item['tgt_starts']=tgt_starts
        item['tgt_ends']=tgt_ends
        item['label_positions']=label_positions
        # item['sep_pos']=sep_pos[0]
        return item


def collate_dataset(inputs):
    max_len=0
    max_prompt_len=0
    max_final_labels=0
    for input in inputs:
        if len(input['input_ids'])>max_len:
            max_len=len(input['input_ids'])
        if len(input['offset'])>max_prompt_len:
            max_prompt_len=len(input['offset'])
        if len(input['prompt_masked_positions'])>max_final_labels:
            max_final_labels=len(input['prompt_masked_positions'])
    inputs_tensor=[]
    for input in inputs:
        # pad input
        item={}
        token_ids=input['input_ids']
        attention_mask=input['attention_mask']
        token_type_ids=input['token_type_ids']
        prompt_attention_mask=input['prompt_attention_mask']
        prompt_masked_positions=input['prompt_masked_positions']
        prompt_attention_mask_=input['prompt_attention_mask_']
        prompt_masked_positions_=input['prompt_masked_positions_']
        prompt_attention_masks_=input['prompt_attention_masks_']
        label_positions=input['label_positions']
        offset=input['offset']
        tgt_starts=input['tgt_starts']
        tgt_ends=input['tgt_ends']
        # sep_pos=input['sep_pos']
        pad_length=max_len-len(token_ids)
        token_ids+=[0]*pad_length
        attention_mask+=[0]*pad_length
        token_type_ids+=[0]*pad_length
        new_prompt_attention_masks_=[]
        for prompt_mask in prompt_attention_masks_:
            prompt_mask+=[0]*pad_length
            new_prompt_attention_masks_.append(prompt_mask)
        prompt_attention_mask_+=[0]*pad_length
        prompt_pad_length=max_prompt_len-len(prompt_attention_mask)
        assert len(prompt_attention_mask)==len(offset)
        prompt_attention_mask+=[0]*prompt_pad_length
        offset+=[0]*prompt_pad_length
        pad_final_labels=max_final_labels-len(prompt_masked_positions)
        # prompt_masked_positions+=[0]*pad_final_labels
        # prompt_masked_positions_+=[0]*pad_final_labels
        prompt_masked_positions+=[0]*pad_final_labels
        prompt_masked_positions_+=[0]*pad_final_labels
        tgt_starts+=[-100]*pad_final_labels
        tgt_ends+=[-100]*pad_final_labels
        label_positions+=[[-100,-100]]*pad_final_labels
        for _ in range(pad_final_labels):
            sub=[0]*max_len
            sub[0]=1
            new_prompt_attention_masks_.append(sub)
        item["input_ids"]=torch.tensor(token_ids).unsqueeze(0)
        # item["labels"]=torch.tensor(labels).unsqueeze(0)
        item["attention_mask"] = torch.tensor(attention_mask).unsqueeze(0)
        item["token_type_ids"] = torch.tensor(token_type_ids).unsqueeze(0)
        item["prompt_attention_mask"]=torch.tensor(prompt_attention_mask).unsqueeze(0)
        item["prompt_masked_positions"]=torch.tensor(prompt_masked_positions).unsqueeze(0)
        item["prompt_attention_mask_"]=torch.tensor(prompt_attention_mask_).unsqueeze(0)
        item["prompt_masked_positions_"]=torch.tensor(prompt_masked_positions_).unsqueeze(0)
        item['offset']=torch.tensor(offset).unsqueeze(0)
        item['tgt_starts']=torch.tensor(tgt_starts).unsqueeze(0)
        item['tgt_ends']=torch.tensor(tgt_ends).unsqueeze(0)
        item['prompt_attention_masks_']=torch.tensor(new_prompt_attention_masks_).unsqueeze(0)
        item['label_positions']=torch.tensor(label_positions).unsqueeze(0)
        inputs_tensor.append(item)

    # update final outputs
    final_input_ids=None
    final_labels=None
    final_attention_mask=None
    final_token_type_ids=None
    final_prompt_attention_mask=None
    final_prompt_attention_mask_=None
    final_labels_=None
    offsets=None
    tgt_starts=None
    tgt_ends=None
    prompt_attention_masks_=None
    label_positions=None
    for input in inputs_tensor:

        if final_input_ids is None:
            final_input_ids = input['input_ids']
            final_labels = input['prompt_masked_positions']
            final_labels_ = input['prompt_masked_positions_']
            final_attention_mask = input['attention_mask']
            final_token_type_ids = input['token_type_ids']
            final_prompt_attention_mask=input['prompt_attention_mask']
            final_prompt_attention_mask_=input['prompt_attention_mask_']
            offsets=input['offset']
            tgt_starts=input['tgt_starts']
            tgt_ends=input['tgt_ends']
            prompt_attention_masks_=input['prompt_attention_masks_']
            label_positions=input['label_positions']

        else:
            final_input_ids = torch.cat((final_input_ids, input['input_ids']), dim=0)
            final_labels = torch.cat((final_labels, input['prompt_masked_positions']), dim=0)
            final_labels_ = torch.cat((final_labels_, input['prompt_masked_positions_']), dim=0)
            final_attention_mask = torch.cat((final_attention_mask, input['attention_mask']), dim=0)
            final_token_type_ids = torch.cat((final_token_type_ids, input['token_type_ids']), dim=0)
            final_prompt_attention_mask=torch.cat((final_prompt_attention_mask, input['prompt_attention_mask']), dim=0)
            final_prompt_attention_mask_=torch.cat((final_prompt_attention_mask_, input['prompt_attention_mask_']), dim=0)
            offsets=torch.cat((offsets, input['offset']), dim=0)
            tgt_starts=torch.cat((tgt_starts, input['tgt_starts']), dim=0)
            tgt_ends=torch.cat((tgt_ends, input['tgt_ends']), dim=0)
            prompt_attention_masks_=torch.cat((prompt_attention_masks_,input['prompt_attention_masks_']),dim=0)
            label_positions=torch.cat((label_positions,input['label_positions']),dim=0)

    batch={}
    batch['input_ids']=final_input_ids
    batch['prompt_masked_positions']=final_labels
    batch['prompt_masked_positions_']=final_labels_
    batch['attention_mask']=final_attention_mask
    batch['token_type_ids']=final_token_type_ids
    batch['prompt_attention_mask']=final_prompt_attention_mask
    batch['prompt_attention_mask_']=final_prompt_attention_mask_
    batch['offsets']=offsets
    batch['tgt_starts']=tgt_starts
    batch['tgt_ends']=tgt_ends
    batch['prompt_attention_masks_']=prompt_attention_masks_
    batch['label_positions']=label_positions
    return batch


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # input
    # parser.add_argument("--data_folder", default="/data/dai031/Corpora")
    # parser.add_argument("--task_name", default="development", type=str)
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    # parser.add_argument("--context_train_filepath", default="train.json", type=str)
    # parser.add_argument("--context_dev_filepath", default="dev.json", type=str)
    # parser.add_argument("--context_test_filepath", default="test.json", type=str)
    # parser.add_argument("--context_data_folder", default='/data1/gl/project/medical-rubiks-cube/process_CDR_CHR/PURE_DATA/CDR/final/txt', type=str)
    parser.add_argument('--task', type=str, default='pubmed', required=True, choices=['pubmed-met','pubmed','met','cail','cdr','chr','ade','ncbi','bc2gm','jnlpba','bc5cdr-disease','bc5cdr-chem','ncbi-disease','biored','tbga','drugprot'])
    # parser.add_argument("--use_features", action='store_true', help='whether use drug and endpoint features.')
    # output
    parser.add_argument("--output_filepath", default="development", type=str)
    # parser.add_argument("--last_step_test_filepath", default="development.json", type=str)
    parser.add_argument("--log_filepath", default="development.log")
    parser.add_argument("--logs", default="summary writer dir")

    # train
    # parser.add_argument("--lr", default=3e-5, type=float)
    # parser.add_argument("--min_lr", default=1e-8, type=float)
    # parser.add_argument("--train_bs", default=16, type=int)
    # parser.add_argument("--eval_bs", default=16, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")
    parser.add_argument("--use_neg", action='store_true',
                        help="Whether to use negative samples.")
    parser.add_argument("--use_pos_prompt", action='store_true',
                        help="Whether to cat negative samples after positive samples.")
    parser.add_argument("--use_only_one_pos", action='store_true',
                        help="Whether to use only one positive sample.")
    parser.add_argument("--only_use_first", action='store_true',
                        help="Whether to cat negative samples after positive samples.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="evaluate all checkpoints.")
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--max_prompt_times', type=int, default=1,
                        help='Limit the total amount of prompt of repeat.')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_lower_case", action='store_true', help='whether use multi-task model.')
    # parser.add_argument("--anneal_factor", default=0.5, type=float)
    # parser.add_argument("--anneal_patience", default=20, type=int)
    # parser.add_argument("--early_stop_patience", default=10, type=int)
    # parser.add_argument("--optimizer", default="adam", type=str)

    # environment
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=0, type=int)

    # embeddings
    parser.add_argument("--embedding_type", default=None, type=str)
    parser.add_argument("--pretrained_dir", default=None, type=str)
    parser.add_argument("--re_pretrained_dir", default=None, type=str)
    # parser.add_argument("--load_pretrain_ckpt_dir", default=None, type=str)
    # parser.add_argument("--load_pretrain_ckpt", action='store_true', help='whether load pretrained model parameters.')
    parser.add_argument("--max_position_embeddings", default=512, type=int)
    
    # dropout
    # parser.add_argument("--dropout", default=0.4, type=float)
    # parser.add_argument("--word_dropout", default=0.05, type=float)
    # parser.add_argument("--variational_dropout", default=0.5, type=float)

    # augmentation
    # parser.add_argument("--augmentation", type=str, nargs="+", default=[])
    # parser.add_argument("--p_power", default=1.0, type=float,
    #                     help="the exponent in p^x, used to smooth the distribution, "
    #                          "if it is 1, the original distribution is used; "
    #                          "if it is 0, it becomes uniform distribution")
    parser.add_argument("--neg_ratio", default=0.3, type=float)
    # parser.add_argument("--num_generated_samples", default=1, type=int)
    # parser.add_argument("--do_train", action="store_true")

    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", default=128, type=int)


    args, _ = parser.parse_known_args()

    # args.train_file = os.path.join(args.data_folder, args.train_filepath)
    # args.dev_file = os.path.join(args.data_folder, args.dev_filepath)
    # args.test_file = os.path.join(args.data_folder, args.test_filepath)
    # args.context_train_filepath=os.path.join(args.context_data_folder, args.context_train_filepath)
    # args.context_dev_filepath=os.path.join(args.context_data_folder, args.context_dev_filepath)
    # args.context_test_filepath=os.path.join(args.context_data_folder, args.context_test_filepath)

    return args


def random_seed(seed=42):
    if seed > 0:
        random.seed(seed)
        numpy.random.seed(int(seed / 2))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(int(seed / 4))
        torch.cuda.manual_seed(int(seed / 8))
        torch.cuda.manual_seed_all(int(seed / 8))

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

task_ner_labels = {
    # 'cail':['Nh', 'NDR', 'NT', 'NW', 'Ns'],
    'cdr':['Disease','Chemical'],
    'tbga':['Gene','Disease'],
    'chr':['ChemMet'],
    'ade':['ADE','Drug'],
    'ncbi':['DiseaseClass','SpecificDisease','CompositeMention','Modifier'],
    'bc2gm':['GENE'],
    'jnlpba':['DNA','cell_type','protein','RNA','cell_line'],
    'bc5cdr-chem':['Chemical'],
    'bc5cdr-disease':['Disease'],
    'ncbi-disease':['Disease'],
    'biored':['GeneOrGeneProduct','DiseaseOrPhenotypicFeature','ChemicalEntity','OrganismTaxon','SequenceVariant','CellLine'],
    'pubmed':['metric','result','experiment','number'],
    'drugprot':['CHEMICAL','GENE']
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter(args.logs)

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = args.batch_size
    train_dataset=args.train_dataset
    train_dataloader=args.train_dataloader
    # train_dataset = ACEDatasetNER(tokenizer=tokenizer, args=args)
                            
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
        # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #                args.train_batch_size if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1
    best_accu = -1
    best_soft_f1=-1

    for _ in train_iterator:
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        numbers=0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            batch={k:inputs.to(device) for k,inputs in batch.items()}
            # sub_corpus=corpus.sentences[step*args.batch_size:(step+1)*args.batch_size]
            outputs=model(batch)
            loss = outputs['loss'] 
            # numbers+=outputs['number']
            loss.backward()
            tr_loss += loss.item()
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm > 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
                logging_loss = tr_loss
                # numbers=0


            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                update = True
                # Save model checkpoint
                if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model)
                    accu=results['accu']
                    pos_accu=results['pos_accu']
                    neg_accu=results['neg_accu']
                    first_pos_accu=results['first_pos_accu']
                    no_first_pos_accu=results['no_first_pos_accu']
                    first_neg_accu=results['first_neg_accu']
                    not_first_neg_accu=results['not_first_neg_accu']
                    # original_f1=results['original_f1']
                    # after_revise_f1=results['after_revise_f1']
                    # revise_f1=results['revise_f1']
                    soft_f1 = results['pos_soft_f1']
                    # accu=results['pos_accu']
                    
                    tb_writer.add_scalar('accu/accu', accu, global_step)
                    tb_writer.add_scalar('accu/pos_accu', pos_accu, global_step)
                    tb_writer.add_scalar('accu/neg_accu', neg_accu, global_step)
                    tb_writer.add_scalar('accu/first_pos_accu', first_pos_accu, global_step)
                    tb_writer.add_scalar('accu/no_first_pos_accu', no_first_pos_accu, global_step)
                    tb_writer.add_scalar('accu/first_neg_accu', first_neg_accu, global_step)
                    tb_writer.add_scalar('accu/not_first_neg_accu', not_first_neg_accu, global_step)
                    # tb_writer.add_scalar('f1/original_f1', original_f1, global_step)
                    # tb_writer.add_scalar('f1/after_revise_f1', after_revise_f1, global_step)
                    # tb_writer.add_scalar('f1/revise_f1', revise_f1, global_step)
                    tb_writer.add_scalar('f1/soft_f1', soft_f1, global_step)


                    # if revise_f1 >= best_f1:
                    #     best_f1 = revise_f1
                    #     print ('Best F1', best_f1)
                    if soft_f1 >= best_soft_f1:
                        best_soft_f1 = soft_f1
                        print ('Best Soft F1', best_soft_f1)
                    # else:
                    #     update = False
                    if accu >= best_accu:
                        best_accu = accu
                        print ('Best Accu', best_accu)
                    else:
                        update = False

                if update:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)
                    torch.save(model_to_save,os.path.join(output_dir, 'pytorch_model.bin'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)


    if args.local_rank in [-1, 0]:
        tb_writer.close()


    return global_step, tr_loss / global_step, best_f1 ,best_accu

def get_metrics(file,revise_dict):
    revise_tp=0
    revise_fp=0
    revise_tn=0
    revise_fn=0
    sub_revise_tp=0
    sub_revise_fp=0
    sub_revise_tn=0
    sub_revise_fn=0
    tp=0
    fp=0
    tn=0
    fn=0
    f1s=[]
    with open(file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            con=json.loads(line)
            doc_key=con['doc_key']
            if doc_key in revise_dict.keys():
                revise_labels=revise_dict[doc_key]
            else:
                revise_labels=[]
            preds=con['predicted_ner']
            golds=con['ner']
            sentences=con['sentences']
            final_preds=preds
            pred_ner=[]
            for tags in final_preds:
                for tag in tags:
                    if tag not in pred_ner:
                        pred_ner.append(tag)
            gold_ner=[]
            # if args.task in ['cdr']:
            #     total_len=0
            #     for tags,sent in zip(golds,sentences):
            #         for tag in tags:
            #             if tag[-1]=='Adverse-Effect':
            #                 tag[-1]='ADE'
            #             tag[0]+=total_len
            #             tag[1]+=total_len
            #             if tag not in gold_ner:
            #                 gold_ner.append(tag)
            #         total_len+=len(sent)
            # else:
            for tags in golds:
                for tag in tags:
                    # if tag[-1]=='Adverse-Effect':
                    #     tag[-1]='ADE'
                    if tag not in gold_ner:
                        gold_ner.append(tag)

            # gold_ner=list(set(gold_ner))
            # pred_ner=list(set(pred_ner))
            for p in pred_ner:
                if p in gold_ner:
                    # print(doc_key,p)
                    tp+=1
                else:
                    fp+=1
            for g in gold_ner:
                if g in pred_ner:
                    tn+=1
                else:
                    fn+=1

            total_len=0
            for i,(sent,revise) in enumerate(zip(sentences,revise_labels)):
                for tag in revise:
                    final_preds[i].append([tag[0]+total_len,tag[1]+total_len,tag[2]])
                total_len+=len(sent)
            pred_ner=[]
            for tags in final_preds:
                for tag in tags:
                    if tag not in pred_ner:
                        pred_ner.append(tag)
            for p in pred_ner:
                if p in gold_ner:
                    # print(doc_key,p)
                    revise_tp+=1
                else:
                    revise_fp+=1
            for g in gold_ner:
                if g in pred_ner:
                    revise_tn+=1
                else:
                    revise_fn+=1
            final_preds=[[] for _ in range(len(sentences))]
            total_len=0
            for i,(sent,revise) in enumerate(zip(sentences,revise_labels)):
                for tag in revise:
                    final_preds[i].append([tag[0]+total_len,tag[1]+total_len,tag[2]])
                total_len+=len(sent)
            pred_ner=[]
            for tags in final_preds:
                for tag in tags:
                    if tag not in pred_ner:
                        pred_ner.append(tag)
            for p in pred_ner:
                if p in gold_ner:
                    # print(doc_key,p)
                    sub_revise_tp+=1
                else:
                    sub_revise_fp+=1
            for g in gold_ner:
                if g in pred_ner:
                    sub_revise_tn+=1
                else:
                    sub_revise_fn+=1
    P=tp/(tp+fp)
    R=tp/(tp+fn)   
    f1=2*P*R/(P+R)
    print('**************************before revision****************************')
    print('p:{}, r:{}, f1:{}'.format(P,R,f1)) 
    print('TP:{}, FP:{}, TN:{}, FN:{}'.format(tp,fp,tn,fn))
    f1s.append(f1)
    P=revise_tp/(revise_tp+revise_fp)
    R=revise_tp/(revise_tp+revise_fn)   
    f1=2*P*R/(P+R)
    print('**************************after revision****************************')
    print('p:{}, r:{}, f1:{}'.format(P,R,f1)) 
    print('TP:{}, FP:{}, TN:{}, FN:{}'.format(revise_tp,revise_fp,revise_tn,revise_fn))
    f1s.append(f1)
    P=sub_revise_tp/(sub_revise_tp+sub_revise_fp) if sub_revise_tp+sub_revise_fp!=0 else 0
    R=sub_revise_tp/(sub_revise_tp+sub_revise_fn)   if sub_revise_tp+sub_revise_fn!=0 else 0
    f1=2*P*R/(P+R) if P+R!=0 else 0
    print('**************************after revision****************************')
    print('p:{}, r:{}, f1:{}'.format(P,R,f1)) 
    print('TP:{}, FP:{}, TN:{}, FN:{}'.format(sub_revise_tp,sub_revise_fp,sub_revise_tn,sub_revise_fn))
    f1s.append(f1)

    return f1s

def f1_score(prediction, ground_truth):
    prediction_tokens = prediction
    ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(args, model,dataset_mode='eval'):
    if dataset_mode=='eval':
        eval_dataloader=args.eval_dataloader
        corpus=args.eval_dataset
    else:
        eval_dataloader=args.test_dataloader
        corpus=args.test_dataset
    model.eval()
    # revise_dict={}
    soft_f1=0
    # soft_count=0
    with torch.no_grad():
        right_num=0
        total_num=0
        right_neg_num=0
        right_first_pos_num=0
        right_no_first_pos_num=0
        not_first_neg_num=0
        first_neg_num=0
        neg_count=0
        pos_count=0
        count_pred_neg=0
        for step,batch in enumerate(tqdm(eval_dataloader)):
            batch={k:inputs.to(device) for k,inputs in batch.items()}
            sub_corpus=corpus.sentences[step*args.batch_size:(step+1)*args.batch_size]
            output=model(batch)
            # preds=[[] for _ in range(args.batch_size)]
            # for prediction in output['predictions']:
            #     predicts=torch.argmax(prediction,dim=-1).tolist()
            #     for i,predict in enumerate(predicts):
            #         preds[i].append(predict[0])
            prediction=output['predictions']
            preds=torch.argmax(prediction,dim=-1).tolist()
            # print(output)
            for predicts,data in zip(preds,sub_corpus):
                start_idxs=data['start_idxs']
                # end_idxs=data['end_idxs']
                labels=data['label']
                # doc_len=data['doc_len']
                mode=data['mode']
                # revise_preds=[]
                for j,pred in enumerate(predicts):
                    flag=False
                    if j<len(labels):
                        total_num+=1
                        
                        if mode in ['first-pos','not-first-pos']:
                            pos_count+=1
                            # if pred[0]==pred[1]==len(start_idxs):
                            if pred[0]==pred[1]==0:
                                count_pred_neg+=1
                            if pred[0]==0 or pred[1]==0:
                                # soft_count+=1
                                continue
                            start=start_idxs[pred[0]]
                            end=start_idxs[pred[1]]-1
                            if start>end:
                                temp=start
                                start=end
                                end=temp
                            # if start<=end:
                            # start=pred[0]-1
                            # end=pred[1]-1
                            if start==data['starts'][j] and end==data['ends'][j]:
                                right_num+=1
                                if mode=='first-pos':
                                    right_first_pos_num+=1
                                else:
                                    right_no_first_pos_num+=1
                            # else:
                            #     print(data['doc_key'],[start,end],[data['starts'][j],data['ends'][j]])
                            label=labels[j]
                            # revise_preds.append([start,end,label])
                            soft_f1+=f1_score(data['sent'][start:end+1], data['sent'][data['starts'][j]:data['ends'][j]+1])
                        else:
                            neg_count+=1
                            # start=start_idxs[pred[0]]
                            # end=start_idxs[pred[1]]
                            # if start==data['starts'][j] and end==data['ends'][j]:
                            if pred[0]==pred[1]==0:
                            # if pred[0]==pred[1]==len(start_idxs):
                                if mode=='neg-after-pos':
                                    not_first_neg_num+=1
                                else:
                                    first_neg_num+=1
                                right_neg_num+=1
                                count_pred_neg+=1
                        
                # if data['doc_key'] not in revise_dict.keys():
                #     revise_dict[data['doc_key']]=[[] for _ in range(doc_len)]
                # revise_dict[data['doc_key']][data['idx']]+=revise_preds
    assert corpus.pos_spans_num==pos_count
    assert corpus.neg_spans_num==neg_count
    assert total_num==corpus.pos_spans_num+corpus.neg_spans_num
    pos_accu=right_num/corpus.pos_spans_num if corpus.pos_spans_num!=0 else 0
    neg_accu=right_neg_num/corpus.neg_spans_num if corpus.neg_spans_num!=0 else 0
    if neg_accu > 1:
        print('Wrong!!!')
    accu=(right_num+right_neg_num)/(total_num)
    first_pos_accu=right_first_pos_num/corpus.pos_first_spans_num if corpus.pos_first_spans_num!=0 else 0
    no_first_pos_accu=right_no_first_pos_num/corpus.pos_no_first_spans_num if corpus.pos_no_first_spans_num!=0 else 0
    first_neg_accu=first_neg_num/corpus.first_neg_num if corpus.first_neg_num!=0 else 0
    not_first_neg_accu=not_first_neg_num/corpus.not_first_neg_num if corpus.not_first_neg_num!=0 else 0
    if corpus.pos_spans_num+corpus.neg_spans_num!=total_num:
        print('Some thing wrong!')
    print('-'*30)
    print('test positive accuracy:{}, right positive number:{}, total positive number:{}'.format(pos_accu,right_num,corpus.pos_spans_num))
    print('test negative accuracy:{}, right negative number:{}, total negative number:{}'.format(neg_accu,right_neg_num,corpus.neg_spans_num))
    print('total accuracy:{}, total number:{}'.format(accu,corpus.pos_spans_num+corpus.neg_spans_num))
    print('test first positive accuracy:{}, right first pos number:{}, total first pos number:{}'.format(first_pos_accu,right_first_pos_num,corpus.pos_first_spans_num))
    print('test not first positive accuracy:{}, right not first pos number:{}, total not first pos number:{}'.format(no_first_pos_accu,right_no_first_pos_num,corpus.pos_no_first_spans_num))
    print('test first negative accuracy:{}, right first neg number:{}, total first neg number:{}'.format(first_neg_accu,first_neg_num,corpus.first_neg_num))
    print('test not first negative accuracy:{}, right not first neg number:{}, total not first neg number:{}'.format(not_first_neg_accu,not_first_neg_num,corpus.not_first_neg_num))
    print('test positive soft f1:{}'.format(soft_f1/pos_count))
    print('pred negative count:{}, pred negative ratio:{}'.format(count_pred_neg,count_pred_neg/total_num))
    # if dataset_mode=='eval':
    #     original_f1,after_revise_f1,revise_f1=get_metrics(args.dev_file,revise_dict)
    # else:
    #     original_f1,after_revise_f1,revise_f1=get_metrics(args.test_file,revise_dict)
    results={}
    results['accu']=round(accu,2)
    results['pos_accu']=round(pos_accu,2)
    results['neg_accu']=round(neg_accu,2)
    results['first_pos_accu']=round(first_pos_accu,2)
    results['no_first_pos_accu']=round(no_first_pos_accu,2)
    results['first_neg_accu']=round(first_neg_accu,2)
    results['not_first_neg_accu']=round(not_first_neg_accu,2)
    # results['original_f1']=original_f1
    # results['after_revise_f1']=after_revise_f1
    # results['revise_f1']=revise_f1
    results['pos_soft_f1']=round(soft_f1/pos_count,2)
    return results

if __name__ == "__main__":
    args = parse_parameters()
    
    device = torch.device("cuda:%d" % args.device)
    args.result = {}

    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    # if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    random_seed(args.seed)
    logger.info(f'CONFIG: "{args}"')
    tokenizer=BertTokenizer.from_pretrained(args.pretrained_dir,do_lower_case=args.do_lower_case)
    special_tokens_dict = {'additional_special_tokens': ['[sos]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    args.tokenizer=tokenizer
    ner_label_list=task_ner_labels[args.task]
    args.ner_label_list=ner_label_list
    # load model
    logger.info("- - - - - - - - - - - - - Creating model- - - - - - - - - - - - - ")
    model = BaselineBert(tokenizer=tokenizer, pretrained_dir=args.pretrained_dir)
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    if args.re_pretrained_dir:
        checkpoint = torch.load(args.re_pretrained_dir, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info('load checkpoint from %s' % args.re_pretrained_dir)
        logger.info(msg)
        print(msg)
    model=model.to(device)
    
    revise_dict={}
    if args.do_train:
        train_corpus = dataset(args,args.train_file,'train')
        train_dataloader=DataLoader(train_corpus,batch_size=args.batch_size,shuffle=True,collate_fn=collate_dataset)
        args.train_dataset=train_corpus
        args.train_dataloader=train_dataloader
    if args.do_eval:
        eval_corpus = dataset(args,args.dev_file,'train')
        eval_dataloader=DataLoader(eval_corpus,batch_size=args.batch_size,shuffle=False,collate_fn=collate_dataset)
        args.eval_dataset=eval_corpus
        args.eval_dataloader=eval_dataloader
    if args.do_test:
        test_corpus = dataset(args,args.test_file,'train')
        test_dataloader=DataLoader(test_corpus,batch_size=args.batch_size,shuffle=False,collate_fn=collate_dataset)
        args.test_dataset=test_corpus
        args.test_dataloader=test_dataloader
    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    best_accu = 0
    best_soft_f1=0
    best_pos_accu=0
    # accu_to_pos={}
    # Training
    if args.do_train:
        global_step, tr_loss, best_f1,best_accu = train(args, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model)
            accu=results['accu']
            pos_accu=results['pos_accu']
            neg_accu=results['neg_accu']
            first_pos_accu=results['first_pos_accu']
            no_first_pos_accu=results['no_first_pos_accu']
            first_neg_accu=results['first_neg_accu']
            not_first_neg_accu=results['not_first_neg_accu']
            # original_f1=results['original_f1']
            # after_revise_f1=results['after_revise_f1']
            # revise_f1=results['revise_f1']
            soft_f1=results['pos_soft_f1']
            # if revise_f1 >= best_f1:
            #     best_f1 = revise_f1
            #     print ('Best F1', best_f1)
            if soft_f1 >= best_soft_f1:
                best_soft_f1 = soft_f1
                print ('Best F1', best_soft_f1)
            # else:
            #     update = False
            if accu >= best_accu:
                # if accu==best_accu:
                #     if pos_accu>accu_to_pos[accu]:
                #         accu_to_pos[accu]=pos_accu
                #         update=True
                #     else:
                #         update=False
                # else:
                #     update=True
                #     accu_to_pos[accu]=pos_accu
                best_accu = accu
                print ('Best Accu', best_accu)
            else:
                update = False
            test_result = evaluate(args, model, 'test')
            print('Test Accu:{}'.format(test_result['accu']))

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            # model_to_save.save_pretrained(output_dir)
            torch.save(model_to_save,os.path.join(output_dir, 'pytorch_model.bin'))

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        # tokenizer.save_pretrained(args.output_dir)
        # torch.save(model_to_save,os.path.join(output_dir, 'pytorch_model.bin'))

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    # Evaluation
    results = {'dev_best_f1': best_f1,'dev_best_accu':best_accu}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]


        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate on test set")

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""


            # model = model_class.from_pretrained(checkpoint, config=config)
            model = torch.load(checkpoint+'/pytorch_model.bin', map_location='cpu')
            # state_dict = checkpoint['model']
            # msg = model.load_state_dict(checkpoint, strict=False)
            model.to(args.device)
            if args.no_test:
                mode='eval'
            else:
                mode='test'
            result = evaluate(args, model, mode)

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))
        logger.info("Result: %s", json.dumps(results))


                    
        