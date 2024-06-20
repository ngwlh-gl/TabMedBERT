import json
from process_chr import check

def find_entity(gold_rel,gold_tags,pred_tags=None):
        golds={}
        for tag in gold_tags:
            golds[(tag[0],tag[1])]=tag[2]
        entity_in_rel=[]
        rel_dict={}
        for rel in gold_rel:
            if (rel[0],rel[1]) not in entity_in_rel:
                entity_in_rel.append((rel[0],rel[1]))
            if (rel[0],rel[1]) not in rel_dict:
                rel_dict[(rel[0],rel[1])]=[]
            rel_dict[(rel[0],rel[1])].append((rel[2],rel[3]))
            if (rel[2],rel[3]) not in entity_in_rel:
                entity_in_rel.append((rel[2],rel[3]))
            if (rel[2],rel[3]) not in rel_dict:
                rel_dict[(rel[2],rel[3])]=[]
            rel_dict[(rel[2],rel[3])].append((rel[0],rel[1]))
        # for k,v in rel_dict.items():
        #     if len(v)>1:
        #         # print('pass')
        #         return False,None
        for tag in entity_in_rel:
            if tag not in golds:
                return False,None
        new_gold_tags=[]
        pop_pred_tags=[]
        for tag,value in golds.items():
            if tag not in entity_in_rel:
                if pred_tags and [tag[0],tag[1],value] in pred_tags:
                    pop_tags=pred_tags.pop(pred_tags.index([tag[0],tag[1],value]))
                    pop_pred_tags.append([pop_tags[0],pop_tags[1]])
            else:
                new_gold_tags.append([tag[0],tag[1],value])
  
        return True,new_gold_tags,pop_pred_tags

def find_entity_2(gold_rel,gold_tags,pred_tags=None):
        golds={}
        for tag in gold_tags:
            golds[(tag[0],tag[1])]=tag[2]
        entity_in_rel=[]
        rel_dict={}
        for rel in gold_rel:
            if (rel[0],rel[1]) not in entity_in_rel:
                entity_in_rel.append((rel[0],rel[1]))
            if (rel[0],rel[1]) not in rel_dict:
                rel_dict[(rel[0],rel[1])]=[]
            rel_dict[(rel[0],rel[1])].append((rel[2],rel[3]))
            if (rel[2],rel[3]) not in entity_in_rel:
                entity_in_rel.append((rel[2],rel[3]))
            if (rel[2],rel[3]) not in rel_dict:
                rel_dict[(rel[2],rel[3])]=[]
            rel_dict[(rel[2],rel[3])].append((rel[0],rel[1]))
        for k,v in rel_dict.items():
            if len(v)>1:
                # print('pass')
                return False,None
        for tag in entity_in_rel:
            if tag not in golds:
                return False,None
        for tag in golds:
            if tag not in entity_in_rel:
                return False,None
        return True,gold_tags

def select_pair_data(file,out_file,pred_file=None):
    w_f=open(out_file,'w',encoding='utf-8')
    pred_results={}
    pmids=[]
    if pred_file:
        with open(pred_file,'r',encoding='utf-8') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                con=json.loads(line)
                if 'predicted_relations' not in con:
                    con['predicted_relations']=[]
                pred_results[con['doc_key']]={'predicted_ner':con['predicted_ner'],'predicted_relations':con['predicted_relations']}
        with open(file,'r',encoding='utf-8') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                con=json.loads(line)
                relations=con['relations']
                sentences=con['sentences']
                # if 'ade' in file:
                #     relations=relations[0]
                ner=con['ner']
                predicted_ner=pred_results[str(con['doc_key'])]['predicted_ner']
                # predicted_ner=con['predicted_ner']
                if len(predicted_ner)==1:
                    new_predicted_ner=[]
                    total_len=0
                    last_len=0
                    for sent in sentences:
                        last_len=total_len
                        total_len+=len(sent)
                        sub_tags=[]
                        for tag in predicted_ner[0]:
                            if tag[0]>=last_len and tag[1]<total_len:
                                sub_tags.append(tag)
                        new_predicted_ner.append(sub_tags)
                    # con['predicted_ner']=new_predicted_ner
                    predicted_ner=new_predicted_ner
                total_len=0
                pred_relations=pred_results[str(con['doc_key'])]['predicted_relations']
                if len(pred_relations)==1 and type(pred_relations)==list:
                    pred_relations=pred_relations[0]
                for i,(sent,gold_rel,gold_tags,pred_tags) in enumerate(zip(sentences,relations,ner,predicted_ner)):
                    if str(con['doc_key'])+'_'+str(i)=='23433219_13':
                        print('debug')
                    flag,gold_tags,pop_pred_tags=find_entity(gold_rel,gold_tags,pred_tags)
                    if flag and gold_rel:
                        if con['doc_key'] not in pmids:
                            pmids.append(con['doc_key'])
                        for tag in gold_tags:
                            if tag not in pred_tags:
                                print(con['doc_key'])
                        for tag in pred_tags:
                            if tag not in gold_tags:
                                # print(con['doc_key'])
                                pass
                        new_gold_rel=[]
                        for rel in gold_rel:
                            new_gold_rel.append([rel[0]-total_len,rel[1]-total_len,rel[2]-total_len,rel[3]-total_len,rel[4]])
                        new_gold_tags=[]
                        for tag in gold_tags:
                            new_gold_tags.append([tag[0]-total_len,tag[1]-total_len,tag[2]])
                        new_pred_tags=[]
                        for tag in pred_tags:
                            new_pred_tags.append([tag[0]-total_len,tag[1]-total_len,tag[2]])
                        pred_rels=[]
                        for rel in pred_relations:
                            if [rel[0],rel[1]] in pop_pred_tags or [rel[2],rel[3]] in pop_pred_tags:
                                continue
                            if total_len<=rel[0]<total_len+len(sent) and total_len<=rel[1]<total_len+len(sent) and total_len<=rel[2]<total_len+len(sent) and total_len<=rel[3]<total_len+len(sent):
                                pred_rels.append([rel[0]-total_len,rel[1]-total_len,rel[2]-total_len,rel[3]-total_len,rel[4]])
                        dic={'sentences':[sent],'ner':[new_gold_tags],'predicted_ner':[new_pred_tags],'clusters':[],'doc_key':str(con['doc_key'])+'_'+str(i),'relations':[new_gold_rel],'predicted_relations':[pred_rels],'absolute_idx':total_len}
                        json.dump(dic,w_f,ensure_ascii=False)
                        w_f.write('\n')
                    total_len+=len(sent)
    else:
        with open(file,'r',encoding='utf-8') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                con=json.loads(line)
                relations=con['relations']
                sentences=con['sentences']
                # if 'ade' in file:
                #     relations=relations[0]
                ner=con['ner']
                total_len=0
                for i,(sent,gold_rel,gold_tags) in enumerate(zip(sentences,relations,ner)):
                    # flag=find_entity(gold_rel,gold_tags)
                    flag,gold_tags=find_entity(gold_rel,gold_tags)
                    if flag and gold_rel:
                        if con['doc_key'] not in pmids:
                            pmids.append(con['doc_key'])
                        new_gold_rel=[]
                        for rel in gold_rel:
                            new_gold_rel.append([rel[0]-total_len,rel[1]-total_len,rel[2]-total_len,rel[3]-total_len,rel[4]])
                        new_gold_tags=[]
                        for tag in gold_tags:
                            new_gold_tags.append([tag[0]-total_len,tag[1]-total_len,tag[2]])
                        dic={'sentences':[sent],'ner':[new_gold_tags],'clusters':[],'doc_key':str(con['doc_key'])+'_'+str(i),'relations':[new_gold_rel],'absolute_idx':total_len}
                        json.dump(dic,w_f,ensure_ascii=False)
                        w_f.write('\n')
                    total_len+=len(sent)
    print(len(pmids))
file='/data1/gl/project/ner-relation/revise/revise_data/cdr/test.json'
# pred_file="/data1/gl/project/ner-relation/ner/data/ADE-10-folders/0/predicted_json/ours/drug-mlm-complete/test.json"
pred_file='/data1/gl/project/ner-relation/PURE-main/ckpts/relation/pred/ours/drug-complete/cdr/predictions.json'
# pred_file="/data1/gl/project/ner-relation/ner/data/official_CDR/predicted_json/scibert/test.json"
# "/data1/gl/project/ner-relation/PL-Marker/ckpts/pubmedbert/re_pred/pubmedbert_adere_models_pred_0/adere-pubmedbert-42/rel_pred_results.json"
# "/data1/gl/project/ner-relation/PL-Marker/ckpts/pubmedbert/re_gold/bioredre_models_gold/bioredre-pubmedbert-42/rel_pred_results.json"
# "/data1/gl/project/ner-relation/PL-Marker/ckpts/pubmedbert/ner/pubmedbert_pubmedner_models/PL-Marker-pubmed-ours-42/ent_pred_test.json"
# "/data1/gl/project/ner-relation/PL-Marker/ckpts/pubmedbert/re_pred/pubmedbert_pubmedre_models_pred/pubmedre-ours-42/rel_pred_results.json"
# pred_file='/data1/gl/project/ner-relation/PURE-main/ckpts/relation/chr/predictions.json'
out_file='/data1/gl/project/ner-relation/revise/revise_data/cdr/select/drug-mlm/test.json'
select_pair_data(file,out_file,pred_file)

file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/test.json'
out_file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/select/test.json'
pred_file=None
# select_pair_data(file,out_file,pred_file)

file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/train.json'
out_file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/select/train.json'
pred_file=None
# select_pair_data(file,out_file,pred_file)

file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/dev.json'
out_file='/data1/gl/project/ner-relation/revise/revise_data/drugprot/select/dev.json'
pred_file=None
# select_pair_data(file,out_file,pred_file)
