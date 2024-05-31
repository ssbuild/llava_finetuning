# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50


import json



x =[
    {
        "id": 0,
        "p" : '',
        "text": "图中是一只拉布拉多犬",
        "img": "../assets/demo.jpeg"
    },
    {
        "id": 1,
        "p": '',
        "text": "图中是重庆的城市天际线",
        "img": "../assets/Chongqing.jpeg"
    },
    {
        "id": 2,
        "p": '',
        "text": "图中是北京的天际线",
        "img": "../assets/Beijing.jpeg"
    },
]


with open('./finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(100):
        for j in range(len(x)):
            index += 1
            x[j]['id'] = index
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )
