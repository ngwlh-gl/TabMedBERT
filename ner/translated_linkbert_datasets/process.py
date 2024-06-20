import random
random.seed(42)
def convert_json_to_txt(in_file,out_file):
    
    with open(in_file,'r',encoding='utf-8') as f:
        test_data=f.readlines()
    # w_f=open(out_file,'w',encoding='utf-8')
    count=0
    # samples=[]
    w_f=open(out_file,'w',encoding='utf-8')
    for data in test_data:
        # count+=1
        data = eval(data)
        sentences = data['sentences']
        ner = data['ner']
        # flag=True
        sub_count=0
        pmid = data['doc_key']
        # if pmid not in useful_pmids:
        #     continue
        sentence_start = 0
        w_f.write('-DOCSTART-'+'\t'+str(pmid)+'\n')
        w_f.write('\n')
        # count=0
        
        for sent_lst, ner_tag in zip(sentences, ner):
            dic = {}
            count+=1
            # if ner_tag:
            tags = ner_tag
            spans=[]
            labels=[]
            sub_count+=len(tags)
            for tag in tags:
                start,end,label=tag
                if label=='Adverse-Effect':
                    label='ADE'
                if label=='Chemical':
                    label='CHEM'
                if label=='Disease':
                    label='DIS'
                spans.append([start,end])
                labels.append(label)
            exist_labels=[]
            for i,word in enumerate(sent_lst):
                for j,span in enumerate(spans):
                    if i>=span[0] and i <=span[1]:
                        if i==span[0]:
                            # if exist_labels[i]!='O':

                            # else:
                                w_f.write(word+'\t'+'B-'+labels[j]+'\n')
                                exist_labels.append('B-'+labels[j])
                        else:
                            w_f.write(word+'\t'+'I-'+labels[j]+'\n')
                            exist_labels.append('I-'+labels[j])
                        break
                else:
                    w_f.write(word+'\t'+'O'+'\n')

            w_f.write('\n')

            sentence_start += len(sent_lst)
    print(count)

def split_file_en(in_file,out_file):
    abstracts=[]
    examples=[]
    example=[]
    words=[]
    tags=[]
    pmid=''
    pmids=[]
    with open(in_file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                if words and tags:
                    assert len(words)==len(tags)
                    
                    examples.append([words,tags])
                    # example=[]
                if examples:
                    abstracts.append(examples)
                break
            if line=='\n':
                if words and tags:
                    assert len(words)==len(tags)
                
                    examples.append([words,tags])
                    # example=[]
                    words=[]
                    tags=[]
                continue
            word,tag=line.strip().split('\t')
            if word=='-DOCSTART-':
                if examples:
                    abstracts.append(examples)
                    examples=[]
                pmids.append(tag)
                continue
            else:
                words.append(word)
                tags.append(tag)

    # with open('test.txt','w',encoding='utf-8') as w_f:
    w_f=open(out_file,'w',encoding='utf-8')
    # test_f=open('new_txt/test.txt','w',encoding='utf-8')
    for pmid,abs in zip(pmids,abstracts):
        w_f.write('-DOCSTART-'+'\t'+pmid+'\n')
        w_f.write('\n')
        for num,example in enumerate(abs):
            words,tags=example
            
            i=0
            last_idx=0
            indexs=[]
            single_sen=[]
            single_tags=[]
            count=0
            while i<len(words):
                single_sen.append(words[i])
                single_tags.append(tags[i])
                if i<len(words)-1:
                    if (words[i]=='.' and words[i+1][0].isupper() and not tags[i+1].startswith('I-')) or (words[i]=='。' and not tags[i+1].startswith('I-')):
                        indexs.append(i)
                        for word,tag in zip(single_sen,single_tags):
                            w_f.write(word+'\t'+tag+'\n')
                        w_f.write('\n')
                    
                        single_sen=[]
                        single_tags=[]  
                        count+=1
                i+=1
            if single_sen!=[]:
                for word,tag in zip(single_sen,single_tags):
                    w_f.write(word+'\t'+tag+'\n')
                w_f.write('\n')



def split_file_zh(in_file,out_file):
    abstracts=[]
    examples=[]
    example=[]
    words=[]
    tags=[]
    pmid=''
    pmids=[]
    with open(in_file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                if words and tags:
                    assert len(words)==len(tags)
                    
                    examples.append([words,tags])
                    # example=[]
                if examples:
                    abstracts.append(examples)
                break
            if line=='\n':
                if words and tags:
                    assert len(words)==len(tags)
                
                    examples.append([words,tags])
                    # example=[]
                    words=[]
                    tags=[]
                continue
            word,tag=line.strip().split('\t')
            if word=='-DOCSTART-':
                if examples:
                    abstracts.append(examples)
                    examples=[]
                pmids.append(tag)
                continue
            else:
                words.append(word)
                tags.append(tag)

    # with open('test.txt','w',encoding='utf-8') as w_f:
    w_f=open(out_file,'w',encoding='utf-8')
    # test_f=open('new_txt/test.txt','w',encoding='utf-8')
    for pmid,abs in zip(pmids,abstracts):
        w_f.write('-DOCSTART-'+'\t'+pmid+'\n')
        w_f.write('\n')
        for num,example in enumerate(abs):
            words,tags=example
            
            i=0
            last_idx=0
            indexs=[]
            single_sen=[]
            single_tags=[]
            count=0
            while i<len(words):
                single_sen.append(words[i])
                single_tags.append(tags[i])
                if i<len(words)-1:
                    if words[i]=='。' and not tags[i+1].startswith('I-'):
                        indexs.append(i)
                        for word,tag in zip(single_sen,single_tags):
                            w_f.write(word+'\t'+tag+'\n')
                        w_f.write('\n')
                    
                        single_sen=[]
                        single_tags=[]  
                        count+=1
                i+=1
            if single_sen!=[]:
                for word,tag in zip(single_sen,single_tags):
                    w_f.write(word+'\t'+tag+'\n')
                w_f.write('\n')

import json
def random_get_samples(num,train_file,out_file):
    cons=[]
    with open(train_file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=json.loads(line)
            line['doc_key']=str(line['doc_key'])+'_zh'
            cons.append(line)
    # 在0-len(cons)-1中随机产生num个数字
    lst=list(range(len(cons)))
    choice_idxs=random.sample(lst,num)
    wf=open(out_file,'w',encoding='utf-8')
    for idx in choice_idxs:
        json.dump(cons[idx],wf,ensure_ascii=False)
        wf.write('\n')


if __name__=="__main__":
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/100-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/100-shot/txt/train.txt'

    # convert_json_to_txt(in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/50-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/50-shot/txt/train.txt'

    # convert_json_to_txt(in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/10-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/10-shot/txt/train.txt'

    # convert_json_to_txt(in_file,out_file)

    split_file_en('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/10-shot/txt/train.txt','/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/10-shot/txt/split/train.txt')
    split_file_en('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/50-shot/txt/train.txt','/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/50-shot/txt/split/train.txt')
    split_file_en('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/100-shot/txt/train.txt','/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/random-shot/100-shot/txt/split/train.txt')


    in_file='/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/train.json'
    out_file='/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/train.txt'

    # convert_json_to_txt(in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/dev.json'
    out_file='/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/dev.txt'

    # convert_json_to_txt(in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/dev.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/txt/dev.txt'

    # convert_json_to_txt(in_file,out_file)

    # split_file_en('/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/train.txt','/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/split/train.txt')
    # split_file_en('/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/dev.txt','/data1/gl/project/ner-relation/ner/datasets/ADE/translate/eng_to_zh/txt/split/dev.txt')
    # split_file_zh('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/txt/train.txt','/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/txt/split/train.txt')
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/zero-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/100-shot/train.json'
    # random_get_samples(100,in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/zero-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/50-shot/train.json'
    # random_get_samples(50,in_file,out_file)
    in_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/zh_to_eng/zero-shot/train.json'
    out_file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/ADE/translate/eng_to_zh/random-shot/10-shot/train.json'
    # random_get_samples(10,in_file,out_file)
    # convert_json_to_txt(in_file,out_file)