import json
import multiprocessing as mp
from tqdm import tqdm

import os
from tqdm import tqdm
from glob import glob
import requests
import re,json
from urllib import parse
import html
import time


os.environ["http_proxy"]='127.0.0.1:7890'
os.environ["https_proxy"]='127.0.0.1:7890'
proxies = {
	"http":"http://127.0.0.1:7890",
	"https":"http://127.0.0.1:7890",
}
GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'


def translate(wait_trans, to_language="zh-CN", text_language="en"):
    result = []
    while True:
        time.sleep(1)
        try:
            text = parse.quote(wait_trans)
            url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
            response = requests.get(url,proxies=proxies)
            data = response.text
            expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
            result = re.findall(expr, data)
            break
        except TimeoutError:
            print("time out")
            continue  # 如果请求超时，继续循环发送请求
        except Exception as e:
            # 其他异常处理
            print(e, wait_trans)
            break
    if len(result):
        return html.unescape(result[0])

if __name__ =="__main__":
    print("start of process...")

    args_list = []
    # 遍历当前目录下的所有文件
    for filename in glob('./*'):
        # 判断是否为文件，并且文件名以'.json'结尾
        if filename.endswith('.json'):
            # 执行待定的文件操作
            # 这里只是简单地打印文件名
            print(f"Processing file {filename}")
            content = []
            with open(filename, 'r') as file:
                for line in tqdm(file, desc=filename):
                    line = line.strip()
                    dct = json.loads(line)
                    dct['sentence1'] = translate(dct['sentence1'])
                    dct['sentence2'] = translate(dct['sentence2'])
                    content.append(dct)

            with open(filename+".translated", 'w', encoding="utf8") as file:
                for translated_dct in content:
                    file.write(json.dumps(translated_dct, ensure_ascii=False)+'\n')

    print("end of process!")