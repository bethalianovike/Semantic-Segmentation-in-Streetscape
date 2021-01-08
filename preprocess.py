import json
import cv2
import os
import glob
import numpy as np
import time

## read image
root = os.path.join(os.getcwd(),"gtFine","val")
img_list = glob.glob(os.path.join(root,"*_color.png"))
img_list.sort()
print("Total train data : ",len(img_list))
# h x w
label_num = 0
label_dic = {}
num = 1
start_time = time.time()
for p in img_list:
    print(num)
    img = cv2.imread(p)
    name = p.split("/")[-1].replace("gtFine_color.png","leftImg8bit")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if list(img[i,j,:]) not in list(label_dic.values()):
                label_dic.update({str(label_num):list(img[i,j,:])})
                path = os.path.join(os.getcwd(),"gtFine","val_mask",str(label_num))
                if not os.path.isdir(path):
                  os.makedirs(path)
                label_num += 1
    picture = np.zeros((len(label_dic),img.shape[0],img.shape[1]))
    key = list(label_dic.keys())
    value = list(label_dic.values())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if list(img[i,j,:]) in value:
                picture[int(key[value.index(list(img[i,j,:]))]),i,j] = 255
    for k in range(picture.shape[0]):
    	if not np.all(picture[k]==0):
        	cv2.imwrite(os.path.join(os.getcwd(),"gtFine","val_mask",str(k),name+'.png'), picture[k])

    num += 1
print(time.time()-start_time,"s")
print(label_dic)


''' read json 
root = os.path.join(os.getcwd(),"gtFine","train")
json_list = glob.glob(os.path.join(root,"*.json"))
json_list.sort()
#print(json_list)
print("Total train data : ",len(json_list))

num = 1
label_num = 0
label_dic = {}
for j in json_list:
  print(num)
  with open(j,'r') as f : 
    name = j.split("/")[-1].replace(".json","")
    data = json.load(f)
    img_h = data['imgHeight']
    img_w = data['imgWidth']
    obj = data['objects']
    label = {}
    for o in obj:
      if o['label'] not in list(label.keys()):
        path = os.path.join(os.getcwd(),"gtFine","train_mask",str(label_num))
        if not os.path.isdir(path):
          os.makedirs(path)
        label.update({o['label']:[o['polygon']]})
        label_dic.update({o['label']:str(label_num)})
        label_num += 1
      else:
        label[o['label']].append([o['polygon']])

    print(label_dic)
    for l in list(label.keys()):
      img = np.zeros((img_h,img_w))
      for p in label[l]:
        cv2.fillConvexPoly(img,np.array(p), (255,255,255))
      cv2.imwrite(os.path.join(os.getcwd(),"gtFine","train_mask",label_dic[l],name+'.png'), img)
    num += 1
    break

'''