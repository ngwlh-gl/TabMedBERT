
import json
import time
import requests
import numpy as np
from tqdm import tqdm
from openpyxl import Workbook,load_workbook
from sklearn.metrics import classification_report
import time
from multiprocessing.dummy import Pool
key = 'VpOcOHT6SpVUZW96M6ciRtXEXPWyjbB3'

prompt_type_dict={
    0:"zero-shot",
    1:"zero-shot-with-explain",
    2:"few-shot"
}

def openai_completion(query,prompt_type):
    url="https%3A%2F%2Fapi.openai.com%2Fv1%2Fchat%2Fcompletions"
    url = "http://mime-test.baidu-int.com/sapi/v1/aichat/cg?url=%s" % url
    zero_shot_payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "你是一个很有用的助手."
            }, 
            {
                "role": "user",
                "content": "请你告诉我以下以[]分隔的文本：[%s] 属于\{情绪感情；广告；生活分享；祝福；乱码\}中的哪一类，以{\"category\": \"类别\"}这种json格式输出结果。" % query
            }
        ],
        "temperature":0.001,
        "model":"gpt-3.5-turbo"
    })

    zero_shot_payload_with_explain = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "你是一个很有用的助手."
            }, 
            {
                "role": "user",
                "content": "请你告诉我以下以[]分隔的文本：[%s] 属于\{情绪感情；广告；生活分享；祝福；乱码\}中的哪一类，以{\"category\": \"类别\"}这种json格式输出结果。并解释你这样分类的原因。" % query
            }
        ],
        "temperature":0.001,
        "model":"gpt-3.5-turbo"
    })

    few_shot_payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "你是一个很有用的助手."
            }, 
            {
                "role": "user",
                "content": "想象你是一个文本分类小助手，你需要根据我给你的要求返回输入文本的类别。"
            },
            {   
                "role": "assistant",
                "content": "我明白了。作为一个文本分类小助手，我将会对您提供的输入文本进行分类。那么，请问您需要将文本分成哪几个类别？"
            },
            {   
                "role": "user",
                "content": "有四个类别：\{情绪感情、广告、生活分享、祝福、乱码\}。下面是我给你的一些例子，请你根据例子对以[]为分隔符的输入的文本进行回答，将你给出的回答以json格式回答。\n \
                输入文本1：[时光漫漫 岁月浅浅] 结果：{\"category\": \"情绪感情\"} \n \
                输入文本3：[出售25布袋大红花] 结果：{\"category\": \"广告\"} \n \
                输入文本4：[快点保存，再给他剪一波] 结果：{\"category\": \"生活分享\"} \n \
                输入文本5：[各位小姐各位爷，祝您小中考大捷！] 结果：{\"category\": \"祝福\"} \n \
                输入文本6：[народ] 结果：{\"category\": \"乱码\"} "
            },
            {   
                "role": "assistant",
                "content": "好的，我明白了。那么您的输入文本是？"
            },
            {   
                "role": "user",
                "content": "输入文本：[%s] 结果：" % query
            }
        ],
        "temperature":0.001,
        "model":"gpt-3.5-turbo"
    })
    if prompt_type=='few-shot':
        prompt=few_shot_payload
    elif prompt_type=='zero-shot':
        prompt=zero_shot_payload
    elif prompt_type=='zero-shot-with-explain':
        prompt=zero_shot_payload_with_explain
    count=0
    while True:
        # print(payload)
        headers = {
            'Content-Type': 'application/json',
            'api-key' : key,
        }
        # time.sleep(1)
        response = requests.request("POST", url, headers=headers, data=prompt).json()
        # time.sleep(1)
        count+=1
        try:
            result = response['choices'][0]['message']['content']
            break
        except Exception as e:
            time.sleep(1)
            # Handle any exceptions that occur during the network request
            # if count == 19:
            #     result=None
            #     print(f'Error processing {query}: {e}')
            #     break
            continue
    # results.append(result)
    
    return result

def evaluate(out_file):
    golds=[]
    preds=[]
    with open(out_file,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=json.loads(line)
            golds.append(line['label'])
            preds.append(line['pred_label'])
    golds=np.array(golds)
    preds=np.array(preds)
    TP = ((preds == golds) & (preds != 0)).astype(int).sum().item()
    P_total = (preds != 0).astype(int).sum().item()
    L_total = (golds != 0).astype(int).sum().item()
    P = TP / P_total if P_total else 0
    R = TP / L_total if L_total else 0
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    result= {"precision": P, "recall": R, "F1": F1}
    target_names = ['情绪感情', '广告', '生活分享','祝福','乱码']
    # target_names = list(label_to_id.keys())
    res=classification_report(golds, preds, target_names=target_names)
    print(res)
    print(result)

def process_file(in_file,out_file):
    # 对测试集中的文本进行分类
    dic={"情绪感情":0,"广告":1,"生活分享":2,"祝福":3,"乱码":4}
    wf=open(out_file,'w',encoding='utf-8')
    golds=[]
    preds=[]
    # evaluate(golds,preds)
    with open(in_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    lines=tqdm(lines)
    flag=False
    for line in lines:
    
        line=json.loads(line)
        if line['id']=="2762":
            flag=True
        if not flag:
            continue
        query=line['sentence']
        result=openai_completion(query,prompt_type_dict[0])
        try:
            
            start=result.find('{')
            end=result.find('}')
            result=json.loads(result[start:end+1])
            # pred_label=dic[json.loads(result)['category']]
            pred_label=dic[result['category']]
            line['pred_label']=pred_label
            golds.append(line['label'])
            preds.append(pred_label)
            json.dump(line,wf,ensure_ascii=False)
            wf.write('\n')
            # print(line)
            # print()
        except:
            print(line['sentence'])
    # metrics=evaluate(golds,preds)
    # print(metrics)

def process_text(line):
    # 对测试集中的文本进行分类
    dic={"情绪感情":0,"广告":1,"生活分享":2,"祝福":3,"乱码":4}
    reverse_dic={v:k for k,v in dic.items()}
#     queries=[
#     # {"id": "3826", "sentence": "发个[号码]圈", "label": 2}
#     {"id": "2762", "sentence": "晒晒我的好东西，眼福，口福，", "label": 2}

# ]
    # for query in queries:
    # line=json.loads(line)
    q=line['sentence']
    result=openai_completion(q,prompt_type_dict[0])
    start=result.find('{')
    end=result.find('}')
    try:   
        start=result.find('{')
        end=result.find('}')
        result=json.loads(result[start:end+1])
        # pred_label=dic[json.loads(result)['category']]
        pred_label=dic[result['category']]
        line['pred_label']=pred_label
        return line
    except:
        print(line['sentence'])

def init():
    ...


def multi_chatgpt(in_file,out_file):
    w_f=open(out_file,'w',encoding='utf-8')
    results=[]
    with open(in_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    # pool=Pool(50)
    # pool.map(translate_list,results)
    with Pool(5, initializer=init, initargs=()) as pool:
        with tqdm(pool.imap(process_text, lines), total=len(results),desc = "building datasets...") as pbar:
            for res in pbar:
                if res!=None:
                    json.dump(res, w_f, ensure_ascii=False)
                    w_f.write('\n')


if __name__=="__main__":
    query={"id": "2762", "sentence": "晒晒我的好东西，眼福，口福，", "label": 2}
    process_text(query)
    # in_file="/Users/genglei06/program/data/pyq/origin/train.json"
    # out_file="/Users/genglei06/program/data/pyq/chatgpt_train.json"
    # multi_chatgpt(in_file,out_file)
    # evaluate(out_file)
    # 721078@Nam