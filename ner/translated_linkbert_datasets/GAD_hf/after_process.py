import json

def filter_data(file,out_file):
    wf=open(out_file,'w',encoding='utf-8')
    with open(file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=json.loads(line)
            if not line["sentence"]:
                continue
            if '@GENE$' in line['sentence'] and '@DISEASE$' in line['sentence']:
                json.dump(line,wf,ensure_ascii=False)
                wf.write('\n')

file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/GAD_hf/train.json.translated'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf_zh/train.json'
filter_data(file,out_file)

file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/GAD_hf/dev.json.translated'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf_zh/dev.json'
filter_data(file,out_file)

file='/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/GAD_hf/test.json.translated'
out_file='/data1/gl/project/ner-relation/LinkBERT/data/seqcls/GAD_hf_zh/test_zh.json'
filter_data(file,out_file)