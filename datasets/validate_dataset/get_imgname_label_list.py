#coding: utf-8
import glob


imglist = glob.glob(r'./*.jpg')
for i in range(len(imglist)):
    imgname = imglist[i].strip().split('./')[-1]
    label = imgname.split('.jpg')[0]
    idx1 = label.find('_')          # index of 1st '_'
    idx2 = label.find('_', idx1+1) # start=idx1+1, to find the 2nd '_'
    label = label[idx2+1:]

    #if len(list(label)) != 10:
    #    print(20*"-", label, imgname)

    print(imgname, label)
