
import json


with open('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/eng_to_zh/zero-shot/train.json','r',encoding='utf-8') as f:
    cons1=f.readlines()

with open('/data1/gl/project/ner-relation/ner/data/official_CDR/json/after_process/train.json','r',encoding='utf-8') as f:
    cons2=f.readlines()

for con1,con2 in zip(cons1,cons2):
    con1=json.loads(con1)
    con2=json.loads(con2)
    text1=con1['sentences']
    text2=con2['sentences']
    if text1!=text2:
        print('text not match')
    ner1=con1['ner']
    ner2=con2['ner']
    if ner1!=ner2:
        print('ner not match')
    rels1=con1['relations']
    rels2=con2['relations']
    if rels1!=rels2:
        print('relations not match')