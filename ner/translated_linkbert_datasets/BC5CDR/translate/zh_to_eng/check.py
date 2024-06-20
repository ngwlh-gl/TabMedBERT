
import json


with open('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/zh_to_eng/zero-shot/dev.json','r',encoding='utf-8') as f:
    cons1=f.readlines()

# with open('/data1/gl/project/ner-relation/ner/data/official_CDR/json/after_process/train.json','r',encoding='utf-8') as f:
#     cons2=f.readlines()
wf= open('/data1/gl/project/ner-relation/ner/translated_linkbert_datasets/BC5CDR/translate/zh_to_eng/zero-shot/after/zero-shot/dev.json','w',encoding='utf-8') 
for con1 in cons1:
    con1=json.loads(con1)
    # con2=json.loads(con2)
    text1=con1['sentences']
    # text2=con2['sentences']
    # if text1!=text2:
    #     print('text not match')
    ner1=con1['ner']
    # ner2=con2['ner']
    # if ner1!=ner2:
    #     print('ner not match')
    rels1=con1['relations'][0]
    # rels2=con2['relations']
    # if rels1!=rels2:
    #     print('relations not match')
    new_rel=[]
    for rel in rels1:
        if rel not in new_rel:
            new_rel.append(rel)
        else:
            print('debug')
    json.dump({'sentences':text1,'ner':ner1,'relations':[new_rel],'doc_key':con1['doc_key'],'predicted_ner':ner1},wf,ensure_ascii=False)
    wf.write('\n')
