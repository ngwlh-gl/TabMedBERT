import json
same_num=0
total_num=0
with open('/data1/gl/project/ner-relation/ner/revise_data/biored/train.json','r',encoding='utf-8') as f:
    while True:
        line=f.readline()
        if not line:
            break
        con=json.loads(line)
        ner=con['ner']
        for tags in ner:
            num=len(tags)
            exist_labels=[]
            for tag in tags:
                if tag[-1] not in exist_labels:
                    exist_labels.append(tag[-1])
            if len(exist_labels)==num:
                same_num+=1
            total_num+=1
print(same_num,total_num)