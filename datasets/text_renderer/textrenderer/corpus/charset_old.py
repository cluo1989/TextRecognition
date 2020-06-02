#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Created on 

@author: HuZhicheng
"""

charset = '0123456789.￥零壹贰叁肆伍陆柒捌玖拾佰仟万分角元圆整一二三四五六七八九十百千年月日abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

with open('set.txt','w') as f:
    for c in charset:
        f.write(c + '\n')

