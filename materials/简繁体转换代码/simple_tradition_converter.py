from langconv import *
'''
chinese simple and tradition converter
'''
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line

# ref: https://www.cnblogs.com/tangxin-blog/p/5616415.html