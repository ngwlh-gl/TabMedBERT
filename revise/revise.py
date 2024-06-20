"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0
"""

import argparse, json, logging, numpy, os, random, sys, torch

from transformers import BertTokenizer
from ours_model import BaselineBert
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

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
    def __init__(self, args):
        # self.name = name
        self.sentences = []
        self.ner_label_list=args.ner_label_list
        self.max_len=args.max_position_embeddings
        revise_file=args.revise_filepath
        self.tokenizer=args.tokenizer
        self.sos_id=self.tokenizer.convert_tokens_to_ids(['[sos]'])[0]
        count_gold_single=0
        total_sentences_num=0
        # if args.task=='cdr':
        prompt=['is','associated','with']
        
        if revise_file is not None:
            with open(revise_file, encoding="utf-8") as f:
                for line in f:
                    con=json.loads(line)
                    sentences=con['sentences']
                    gold_ner=con['ner']
                    
                    if args.task=='tbga':
                        number=random.random()
                        head=gold_ner[0][0]
                        tail=gold_ner[0][1]
                        sent=sentences[0]
                        if con['relations'][0][0][-1]=='NA':
                            continue
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
                        self.sentences.append({'sent':sent,'start':start,'end':end,'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
                    else:
                        pred_ner=con['predicted_ner']
                        for tags in gold_ner:
                            exist_labels=[]
                            for tag in tags:
                                if tag [-1] not in exist_labels:
                                    exist_labels.append(tag[-1])
                            if len(exist_labels)!=len(self.ner_label_list) and len(exist_labels)==1:
                                count_gold_single+=1
                            total_sentences_num+=1
                        total_len=0
                        for i,(sent,gold_tags,pred_tags) in enumerate(zip(sentences,gold_ner,pred_ner)):
                            # exist_labels=[]
                            # for tag in gold_tags:
                            #     if tag [-1] not in exist_labels:
                            #         exist_labels.append(tag[-1])
                            # if len(exist_labels)!=len(self.ner_label_list) and len(exist_labels)==1:
                            #     total_len+=len(sent)
                            #     continue
                            exist_labels=[]
                            for tag in pred_tags:
                                if tag[-1] not in exist_labels:
                                    exist_labels.append(tag[-1])
                            if args.task in ['cdr','ade','ncbi','jnlpba']:
                                if len(exist_labels)!=len(self.ner_label_list):
                                    other_tags=list(set(self.ner_label_list)-set(exist_labels))
                                    for tag in pred_tags:
                                        if args.task=='cdr':
                                            if tag[2]=='Chemical':
                                                prompt_=[tag[-1]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+['Disease']+['[sos]']+['.']
                                                label='Disease'
                                            else:
                                                prompt_=[tag[-1]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+['Chemical']+['[sos]']+['.']
                                                label='Chemical'
                                            self.sentences.append({'sent':sent,'start':tag[0],'end':tag[1],'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
                                        elif args.task=='ade':
                                            if tag[2]=='ADE':
                                                prompt_=['Adverse-Effect']+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+['Drug']+['[sos]']+['.']
                                                label='Drug'
                                            else:
                                                prompt_=[tag[-1]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+['Adverse-Effect']+['[sos]']+['.']
                                                label='ADE'
                                            self.sentences.append({'sent':sent,'start':tag[0],'end':tag[1],'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
                                        elif args.task=='ncbi' or args.task=='jnlpba':
                                            for obj_tag in other_tags:
                                                prompt_=[tag[-1]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+[obj_tag]+['[sos]']+['.']
                                                label=obj_tag
                                                self.sentences.append({'sent':sent,'start':tag[0],'end':tag[1],'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
                                        
                            else:
                                if args.task in ['chr','bc5cdr-chem','bc5cdr-disease','ncbi-disease','bc2gm']:
                                    for tag in pred_tags:
                                        prompt_=[tag[2]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+[tag[2]]+['[sos]']+['.']
                                        label=tag[2]
                                        self.sentences.append({'sent':sent,'start':tag[0],'end':tag[1],'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})
                                else:
                                    if args.task=='biored':
                                        other_tags=list(set(self.ner_label_list)-set(exist_labels))
                                        for tag in pred_tags:
                                            for obj_tag in other_tags:
                                                prompt_=[tag[-1]]+sent[tag[0]-total_len:tag[1]+1-total_len]+prompt+[obj_tag]+['[sos]']+['.']
                                                label=obj_tag
                                                self.sentences.append({'sent':sent,'start':tag[0],'end':tag[1],'label':label,'doc_key':con['doc_key'],'idx':i,'prompt':prompt_,'doc_len':len(sentences)})

                                
                            total_len+=len(sent)
            # assert idx == len(self.sentences)
            logger.info("Load %s sentences from %s" % (len(self.sentences), revise_file))
            print("Load %s sentences from %s" % (len(self.sentences), revise_file))
            if args.task!='tbga':
                ratio = count_gold_single/total_sentences_num if total_sentences_num!=0 else 0
                print("Only one entity in sentence : {}, ratio is {}, Total sentence number : {}".format(count_gold_single,ratio,total_sentences_num))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        data=self.sentences[idx]
        sent=data["sent"]
        doc_key=data['doc_key']
        # if doc_key=='24802403':
        #     print('debug')
        index=data['idx']
        prompt=data['prompt']
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
        token_ids=[]
        attention_mask=[]
        token_type_ids=[]
        prompt_attention_mask=[]
        prompt_attention_mask_=[]
        token_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(0)
        attention_mask.append(1)
        offset={}
        offset[count_word]=count
        prompt_attention_mask.append(0)
        prompt_attention_mask_.append(0)
        count+=1
        count_word+=1
        tokens=[]
        tokens+=['[CLS]']
        for i,token in enumerate(sent):
            if count>=self.max_len-prompt_len-2:
                break
            sub_tokens=self.tokenizer.tokenize(token)
            tokens+=sub_tokens
            sub_tokens_ids=self.tokenizer.convert_tokens_to_ids(sub_tokens)
            token_ids+=sub_tokens_ids
            attention_mask+=[1]*len(sub_tokens_ids)
            prompt_attention_mask+=[1]
            prompt_attention_mask_+=[1]*len(sub_tokens_ids)
            token_type_ids+=[0]*len(sub_tokens_ids)
            offset[count_word]=count
            for _ in range(len(sub_tokens_ids)):
                start_idxs[count]=i
                # end_idxs[count]=i
                count+=1
            count_word+=1
        token_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        prompt_attention_mask.append(0)
        prompt_attention_mask_.append(0)
        token_type_ids.append(0)
        offset[count_word]=count
        tokens+=['[SEP]']
        count+=1
        count_word+=1
        for i,token in enumerate(prompt):
            sub_tokens=self.tokenizer.tokenize(token)
            tokens+=sub_tokens
            sub_tokens_ids=self.tokenizer.convert_tokens_to_ids(sub_tokens)
            token_ids+=sub_tokens_ids
            attention_mask+=[1]*len(sub_tokens_ids)
            prompt_attention_mask+=[0]
            prompt_attention_mask_+=[0]*len(sub_tokens_ids)
            token_type_ids+=[1]*len(sub_tokens_ids)
            offset[count_word]=count
            for token_id in sub_tokens_ids:
                if token_id==self.sos_id:
                    prompt_masked_positions.append(count_word)
                    prompt_masked_positions_.append(count)
                count+=1
            count_word+=1
        token_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        prompt_attention_mask.append(0)
        prompt_attention_mask_.append(0)
        token_type_ids.append(1)
        offset[count_word]=count
        tokens+=['[SEP]']
        self.sentences[idx]['start_idxs']=start_idxs
        # self.sentences[idx]['end_idxs']=end_idxs

        # # cut or pad input
        if len(token_ids)>=self.max_len:
            # cut input
            token_ids=token_ids[:self.max_len]
            attention_mask=attention_mask[:self.max_len]
            token_type_ids=token_type_ids[:self.max_len]
            prompt_attention_mask=prompt_attention_mask[:self.max_len]
            prompt_attention_mask_=prompt_attention_mask_[:self.max_len]
        # else:
        #     # pad input
        #     pad_length=self.max_len-len(token_ids)
        #     token_ids+=[self.tokenizer.pad_token_id]*pad_length
        #     attention_mask+=[0]*pad_length
        #     token_type_ids+=[0]*pad_length
        assert len(token_ids)==len(attention_mask)==len(token_type_ids)==len(prompt_attention_mask_)
        assert len(prompt_attention_mask)==len(total_sentences)==len(offset)
        
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
        item['offset']=list(offset.values())
        return item


def collate_dataset(inputs):
    max_len=0
    max_prompt_len=0
    for input in inputs:
        if len(input['input_ids'])>max_len:
            max_len=len(input['input_ids'])
        if len(input['offset'])>max_prompt_len:
            max_prompt_len=len(input['offset'])
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
        offset=input['offset']
        pad_length=max_len-len(token_ids)
        token_ids+=[0]*pad_length
        attention_mask+=[0]*pad_length
        token_type_ids+=[0]*pad_length
        prompt_attention_mask_+=[0]*pad_length
        prompt_pad_length=max_prompt_len-len(prompt_attention_mask)
        assert len(prompt_attention_mask)==len(offset)
        prompt_attention_mask+=[0]*prompt_pad_length
        offset+=[0]*prompt_pad_length
        item["input_ids"]=torch.tensor(token_ids).unsqueeze(0)
        # item["labels"]=torch.tensor(labels).unsqueeze(0)
        item["attention_mask"] = torch.tensor(attention_mask).unsqueeze(0)
        item["token_type_ids"] = torch.tensor(token_type_ids).unsqueeze(0)
        item["prompt_attention_mask"]=torch.tensor(prompt_attention_mask).unsqueeze(0)
        item["prompt_masked_positions"]=torch.tensor(prompt_masked_positions).unsqueeze(0)
        item["prompt_attention_mask_"]=torch.tensor(prompt_attention_mask_).unsqueeze(0)
        item["prompt_masked_positions_"]=torch.tensor(prompt_masked_positions_).unsqueeze(0)
        item['offset']=torch.tensor(offset).unsqueeze(0)
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

        else:
            final_input_ids = torch.cat((final_input_ids, input['input_ids']), dim=0)
            final_labels = torch.cat((final_labels, input['prompt_masked_positions']), dim=0)
            final_labels_ = torch.cat((final_labels_, input['prompt_masked_positions_']), dim=0)
            final_attention_mask = torch.cat((final_attention_mask, input['attention_mask']), dim=0)
            final_token_type_ids = torch.cat((final_token_type_ids, input['token_type_ids']), dim=0)
            final_prompt_attention_mask=torch.cat((final_prompt_attention_mask, input['prompt_attention_mask']), dim=0)
            final_prompt_attention_mask_=torch.cat((final_prompt_attention_mask_, input['prompt_attention_mask_']), dim=0)
            offsets=torch.cat((offsets, input['offset']), dim=0)

    batch={}
    batch['input_ids']=final_input_ids
    batch['prompt_masked_positions']=final_labels
    batch['prompt_masked_positions_']=final_labels_
    batch['attention_mask']=final_attention_mask
    batch['token_type_ids']=final_token_type_ids
    batch['prompt_attention_mask']=final_prompt_attention_mask
    batch['prompt_attention_mask_']=final_prompt_attention_mask_
    batch['offsets']=offsets
    return batch


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # input
    # parser.add_argument("--data_folder", default="/data/dai031/Corpora")
    # parser.add_argument("--task_name", default="development", type=str)
    # parser.add_argument("--train_filepath", default="train.txt", type=str)
    # parser.add_argument("--dev_filepath", default="dev.txt", type=str)
    parser.add_argument("--revise_filepath", default="test.json", type=str)
    # parser.add_argument("--context_train_filepath", default="train.json", type=str)
    # parser.add_argument("--context_dev_filepath", default="dev.json", type=str)
    # parser.add_argument("--context_test_filepath", default="test.json", type=str)
    # parser.add_argument("--context_data_folder", default='/data1/gl/project/medical-rubiks-cube/process_CDR_CHR/PURE_DATA/CDR/final/txt', type=str)
    parser.add_argument('--task', type=str, default='pubmed', required=True, choices=['pubmed-met','pubmed','met','cail','cdr','chr','ade','ncbi','bc2gm','jnlpba','bc5cdr-disease','bc5cdr-chem','ncbi-disease','biored','tbga'])
    # parser.add_argument("--use_features", action='store_true', help='whether use drug and endpoint features.')
    # output
    parser.add_argument("--output_filepath", default="development", type=str)
    # parser.add_argument("--last_step_test_filepath", default="development.json", type=str)
    parser.add_argument("--log_filepath", default="development.log")

    # train
    # parser.add_argument("--lr", default=3e-5, type=float)
    # parser.add_argument("--min_lr", default=1e-8, type=float)
    # parser.add_argument("--train_bs", default=16, type=int)
    # parser.add_argument("--eval_bs", default=16, type=int)
    # parser.add_argument("--max_epochs", default=100, type=int)
    # parser.add_argument("--anneal_factor", default=0.5, type=float)
    # parser.add_argument("--anneal_patience", default=20, type=int)
    # parser.add_argument("--early_stop_patience", default=10, type=int)
    # parser.add_argument("--optimizer", default="adam", type=str)

    # environment
    parser.add_argument("--seed", default=52, type=int)
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
    # parser.add_argument("--replace_ratio", default=0.3, type=float)
    # parser.add_argument("--num_generated_samples", default=1, type=int)
    # parser.add_argument("--do_train", action="store_true")

    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", default=128, type=int)


    args, _ = parser.parse_known_args()

    # args.train_filepath = os.path.join(args.data_folder, args.train_filepath)
    # args.dev_filepath = os.path.join(args.data_folder, args.dev_filepath)
    # args.test_filepath = os.path.join(args.data_folder, args.test_filepath)
    # args.context_train_filepath=os.path.join(args.context_data_folder, args.context_train_filepath)
    # args.context_dev_filepath=os.path.join(args.context_data_folder, args.context_dev_filepath)
    # args.context_test_filepath=os.path.join(args.context_data_folder, args.context_test_filepath)

    return args


def random_seed(seed=52):
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
    'biored':['GeneOrGeneProduct','DiseaseOrPhenotypicFeature','ChemicalEntity','OrganismTaxon','SequenceVariant','CellLine']
}


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
    tokenizer=BertTokenizer.from_pretrained(args.pretrained_dir)
    special_tokens_dict = {'additional_special_tokens': ['[sos]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    args.tokenizer=tokenizer
    ner_label_list=task_ner_labels[args.task]
    args.ner_label_list=ner_label_list
    corpus = dataset(args)
    # load model
    logger.info("- - - - - - - - - - - - - Creating model- - - - - - - - - - - - - ")
    model = BaselineBert(tokenizer=tokenizer, pretrained_dir=args.pretrained_dir)
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(args.re_pretrained_dir, map_location='cpu')
    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict, strict=False)
    model=model.to(device)
    logger.info('load checkpoint from %s' % args.re_pretrained_dir)
    logger.info(msg)
    print(msg)
    revise_dict={}
    eval_dataloader=DataLoader(corpus,batch_size=args.batch_size,shuffle=False,collate_fn=collate_dataset)
    with torch.no_grad():
        if args.task=='tbga':
            right_num=0
            total_num=0
            for step,batch in enumerate(tqdm(eval_dataloader)):
                batch={k:inputs.to(device) for k,inputs in batch.items()}
                sub_corpus=corpus.sentences[step*args.batch_size:(step+1)*args.batch_size]
                output=model(batch)
                predictions=torch.argmax(output['prediction'],dim=-1).tolist()
                # print(output)
                for preds,data in zip(predictions,sub_corpus):
                    start_idxs=data['start_idxs']
                    # end_idxs=data['end_idxs']
                    label=data['label']
                    doc_len=data['doc_len']
                    revise_preds=[]
                    for pred in preds:
                        total_num+=1
                        start=start_idxs[pred[0]]
                        end=start_idxs[pred[1]]-1
                        if start<=end:
                            # start=pred[0]-1
                            # end=pred[1]-1
                            if start==data['start'] and end==data['end']:
                                right_num+=1
            print('TBGA test accuracy:{}, raght number:{}, total number:{}'.format(right_num/total_num,right_num,total_num))
        else:
            for step,batch in enumerate(tqdm(eval_dataloader)):
                batch={k:inputs.to(device) for k,inputs in batch.items()}
                sub_corpus=corpus.sentences[step*args.batch_size:(step+1)*args.batch_size]
                output=model(batch)
                predictions=torch.argmax(output['prediction'],dim=-1).tolist()
                # print(output)
                for preds,data in zip(predictions,sub_corpus):
                    start_idxs=data['start_idxs']
                    # end_idxs=data['end_idxs']
                    label=data['label']
                    doc_len=data['doc_len']
                    revise_preds=[]
                    for pred in preds:
                        start=start_idxs[pred[0]]
                        end=start_idxs[pred[1]]-1
                        if start<=end:
                            revise_preds.append([start,end,label])

                    if data['doc_key'] not in revise_dict.keys():
                        revise_dict[data['doc_key']]=[[] for _ in range(doc_len)]
                    revise_dict[data['doc_key']][data['idx']]+=revise_preds
    # w_f=open(args.output_filepath,'w',encoding='utf-8')
    if args.task!='tbga':
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
        with open(args.revise_filepath,'r',encoding='utf-8') as f:
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
                        if tag[-1]=='Adverse-Effect':
                            tag[-1]='ADE'
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
                        print(doc_key,p)
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

        P=revise_tp/(revise_tp+revise_fp)
        R=revise_tp/(revise_tp+revise_fn)   
        f1=2*P*R/(P+R)
        print('**************************after revision****************************')
        print('p:{}, r:{}, f1:{}'.format(P,R,f1)) 
        print('TP:{}, FP:{}, TN:{}, FN:{}'.format(revise_tp,revise_fp,revise_tn,revise_fn))

        P=sub_revise_tp/(sub_revise_tp+sub_revise_fp)
        R=sub_revise_tp/(sub_revise_tp+sub_revise_fn)   
        f1=2*P*R/(P+R)
        print('**************************after revision****************************')
        print('p:{}, r:{}, f1:{}'.format(P,R,f1)) 
        print('TP:{}, FP:{}, TN:{}, FN:{}'.format(sub_revise_tp,sub_revise_fp,sub_revise_tn,sub_revise_fn))

            



                    
        