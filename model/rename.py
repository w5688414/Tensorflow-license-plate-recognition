# -*- coding:utf-8 -*-
import os
# 原路径
path="E:/acer/Documents/code/data/crop/"
# 现路径
outputpath="E:/acer/Documents/code/data/change_name/"
for i, filename in enumerate(os.listdir(path)):
    try:
        if (filename.endswith('.jpg')):
            f = filename.split("_")[0]+".jpg"
            print(f)
            # path = os.path.join(outputpath,f)
            ori_name = path + filename
            os.rename(ori_name,outputpath+f)
    except:
        pass
