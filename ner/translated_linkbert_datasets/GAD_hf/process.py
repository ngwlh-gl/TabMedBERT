import json

def convert(file,out_file):
    w_f=open(out_file,'w',encoding='utf-8')
    wrong=0
    with open(file,'r',encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            con=json.loads(line)
            id=con['id']
            sentence=con['sentence']
            label=con['label']
            tokens=sentence.split()
            ner=[]
            relations=[]
            new_tokens=[]
            count=0
            for i,token in enumerate(tokens):
                if '@GENE$' in token and '@DISEASE$' not in token:
                    pos=token.find('@GENE$')
                    if token[:pos]:
                        new_tokens.append(token[:pos])
                        count+=1
                    new_tokens.append(token[pos:pos+6])
                    ner.append([count,count,'GENE'])
                    count+=1
                    if token[pos+6:]:
                        new_tokens.append(token[pos+6:])
                        count+=1
                elif '@DISEASE$' in token and '@GENE$' not in token:
                    pos=token.find('@DISEASE$')
                    if token[:pos]:
                        new_tokens.append(token[:pos])
                        count+=1
                    new_tokens.append(token[pos:pos+9])
                    ner.append([count,count,'DISEASE'])
                    count+=1
                    if token[pos+9:]:
                        new_tokens.append(token[pos+9:])
                        count+=1  
                elif  '@DISEASE$' in token and '@GENE$' in token:
                    pos=token.find('@GENE$')
                    if token[:pos]:
                        new_tokens.append(token[:pos])
                        count+=1
                    new_tokens.append(token[pos:pos+6])
                    ner.append([count,count,'GENE'])
                    count+=1
                    token=token[pos+6:]
                    pos=token.find('@DISEASE$')
                    if token[:pos]:
                        new_tokens.append(token[:pos])
                        count+=1
                    new_tokens.append(token[pos:pos+9])
                    ner.append([count,count,'DISEASE'])
                    count+=1
                    if token[pos+9:]:
                        new_tokens.append(token[pos+9:])
                        count+=1
                else:
                    new_tokens.append(token)
                    count+=1
            if len(ner)!=2:
                print('wrong')
                wrong+=1
            else:
                if label!='0':
                    if ner[0][-1]=='GENE':
                        relations.append([ner[0][0],ner[0][1],ner[1][0],ner[1][1],'relation'])
                    else:
                        relations.append([ner[1][0],ner[1][1],ner[0][0],ner[0][1],'relation'])
            json.dump({'doc_key':id,'sentences':[new_tokens],'ner':[ner],'relations':[relations]},w_f,ensure_ascii=False)
            w_f.write('\n')
    print(wrong)

file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/train.json'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/json/train.json'
convert(file,out_file)

file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/dev.json'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/json/dev.json'
convert(file,out_file)

file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/test.json'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf/json/test.json'
convert(file,out_file)