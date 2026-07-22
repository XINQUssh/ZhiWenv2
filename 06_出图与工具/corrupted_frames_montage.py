# -*- coding: utf-8 -*-
"""导出2根问题手指的 模板帧(好) vs test帧(坏) 对比图, 供客户核查采集退化。"""
import os, glob
import numpy as np, cv2
def load_img(p):
    with open(p,'rb') as f: return cv2.imdecode(np.frombuffer(f.read(),np.uint8), cv2.IMREAD_GRAYSCALE)
def enh(img):  # CLAHE增强(和匹配同预处理), 便于看脊线
    return cv2.createCLAHE(4.0,(8,8)).apply(img)
specs=[('btp_fys_R0','f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏/fys_R0/Rgd1237'),
       ('wtp_xyz_R1','f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏/xyz_R1/Rgd1245'),
       ('btp_xzc_R2','f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏/xzc_R2/Rgd1237')]
def tile(img,sz=130):
    im=cv2.normalize(enh(img),None,0,255,cv2.NORM_MINMAX); return cv2.resize(im,(sz,sz),interpolation=cv2.INTER_NEAREST)
rows=[]
labels_top=['template f0','template f35','template f69','TEST f75','TEST f85','TEST f95']
for name,d in specs:
    ps=sorted(glob.glob(os.path.join(d,'*.bmp')))
    sel=[0,35,69,75,85,min(95,len(ps)-1)]
    imgs=[tile(load_img(ps[i])) for i in sel]
    # 在模板与test之间加红色分隔
    sep=np.zeros((130,6,3),np.uint8); sep[:]= (0,0,200)
    def c3(g): return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    row=np.hstack([c3(imgs[0]),c3(imgs[1]),c3(imgs[2]),sep,c3(imgs[3]),c3(imgs[4]),c3(imgs[5])])
    # 左侧标签条
    lab=np.full((130,90,3),255,np.uint8); cv2.putText(lab,name,(4,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    rows.append(np.hstack([lab,row]))
    rows.append(np.full((8,rows[-1].shape[1],3),255,np.uint8))
canvas=np.vstack(rows[:-1])
# 顶部列标题
hdr=np.full((26,canvas.shape[1],3),255,np.uint8)
xs=[90, 90+130, 90+260, 90+396, 90+526, 90+656]
for x,t in zip(xs,labels_top):
    col=(0,0,200) if 'TEST' in t else (0,0,0)
    cv2.putText(hdr,t,(x+4,18),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
out=np.vstack([hdr,canvas])
ok,buf=cv2.imencode('.png',out)
with open('f:/1111/指纹/损坏帧核查_2根手指.png','wb') as f: f.write(buf.tobytes())
print('saved 损坏帧核查_2根手指.png  ok=%s  (左3列=模板帧好, 右3列=test帧退化)'%ok)
