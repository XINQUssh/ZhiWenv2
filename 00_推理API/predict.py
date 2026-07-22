# -*- coding: utf-8 -*-
"""
指纹匹配 可调用接口（自包含, 无需其他脚本）。
方法: 自训纹理描述子(真实跨帧对应训练) + 密集采样 + 变换约束几何验证 + 重叠面积归一化。
固定测试集: FAR=0 时 FRR=2.34%(全27组) / 0.64%(清洁手指)。

用法:
    from predict import FingerprintMatcher
    m = FingerprintMatcher()                       # 默认加载 best.pth, 阈值见 DEFAULT_THRESHOLD
    g = m.enroll([img1, img2, ...])                # 注册: 多帧模板 -> 特征(预计算一次)
    score, is_same = m.verify(probe_img, g)        # 验证: 探针 vs 模板, 返回(分数, 是否同指)
    s = m.match_score(imgA, imgB)                   # 或: 单对匹配分数
img 可为灰度 numpy 数组(HxW, uint8)或 .bmp 路径。

环境: python(torch+cv2), 见 README.md。需设 KMP_DUPLICATE_LIB_OK=TRUE。
"""
import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

# ============== 配置 ==============
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_DIR, 'best.pth')      # 最优描述子(=texdesc_v3)
DEFAULT_THRESHOLD = 20.0   # FAR=0 工作点(测试集冒充内点max=19.76); 调高更严, 调低更松
                            # 参考工作点(40模板, 重叠归一化分数): FAR=0->thr19.8/FRR2.34%; FAR<0.002%->thr22.5/FRR2.34%
PATCH = 32; HALF = 16; STRIDE = 6

# ============== 预处理(与训练/评估一致) ==============
def _clahe(img): return cv2.createCLAHE(4.0,(8,8)).apply(img)
def _up2(img): return cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
def _keep_cc(m):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8,cv2.CV_32S)
    return m if n<=1 else ((l==1+np.argmax(s[1:,cv2.CC_STAT_AREA]))*255).astype(np.uint8)
def _fillh(m):
    bg=cv2.bitwise_not(m); n,l,s,_=cv2.connectedComponentsWithStats(bg,8,cv2.CV_32S); h,w=m.shape
    for i in range(1,n):
        L,T,W,H=s[i,cv2.CC_STAT_LEFT],s[i,cv2.CC_STAT_TOP],s[i,cv2.CC_STAT_WIDTH],s[i,cv2.CC_STAT_HEIGHT]
        if L>0 and L+W<w-1 and T>0 and T+H<h-1: m[l==i]=255
    return m
def _gmask(img,sig=13/3.,pct=95,r=.2):
    dx=cv2.Sobel(img,cv2.CV_32F,1,0,3); dy=cv2.Sobel(img,cv2.CV_32F,0,1,3); mg=cv2.magnitude(dx,dy)
    gs=int(np.ceil(3*sig))*2+1; ma=cv2.GaussianBlur(mg,(gs,gs),sig)
    th=np.percentile(mg.flatten(),pct)*r; _,m=cv2.threshold(ma,th,255,0); m=m.astype(np.uint8)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,se,iterations=6); m=_keep_cc(m); m=_fillh(m)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,se,iterations=2); return _keep_cc(m)
def _prep(img):
    e=_clahe(img); u=_up2(e); m=_gmask(u); mk=u.copy(); mk[m==0]=0
    v=mk[m>0].astype(np.float32); mu,sd=(v.mean(),v.std()+1e-6) if len(v) else (0,1)
    norm=(mk.astype(np.float32)-mu)/sd; norm[m==0]=0
    return norm,m

# ============== 描述子网络 ==============
class _Desc(nn.Module):
    def __init__(s,d=128):
        super().__init__()
        def cbr(i,o,st=1): return [nn.Conv2d(i,o,3,st,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.net=nn.Sequential(*cbr(1,32),*cbr(32,32),*cbr(32,64,2),*cbr(64,64),*cbr(64,128,2),*cbr(128,128),nn.AdaptiveAvgPool2d(1))
        s.fc=nn.Linear(128,d)
    def forward(s,x): z=s.net(x).flatten(1); return F.normalize(s.fc(z),dim=1)

# ============== 匹配器 ==============
HO=20000.0  # 重叠面积归一化常数
class FingerprintMatcher:
    def __init__(self, model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD, device=None):
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = _Desc().to(self.dev); self.net.load_state_dict(torch.load(model_path,map_location=self.dev)); self.net.eval()
        self.threshold = threshold; self.bf = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _load(img):
        if isinstance(img,str):
            with open(img,'rb') as f: img=cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_GRAYSCALE)
        return img

    @torch.no_grad()
    def feat(self, img):
        """单帧 -> (关键点坐标(N,2), 描述子(N,128), 前景mask)。可用于预计算模板。"""
        img=self._load(img); norm,m=_prep(img); H,W=norm.shape
        pts=[]; pat=[]
        for y in range(HALF,H-HALF,STRIDE):
            for x in range(HALF,W-HALF,STRIDE):
                if m[y,x]>0: pts.append((x,y)); pat.append(norm[y-HALF:y+HALF,x-HALF:x+HALF])
        if len(pat)<4: return (np.zeros((0,2),np.float32),None,m)
        X=torch.tensor(np.stack(pat)[:,None],dtype=torch.float32).to(self.dev); d=[]
        for i in range(0,len(X),1024): d.append(self.net(X[i:i+1024]).cpu().numpy())
        return (np.float32(pts),np.concatenate(d).astype(np.float32),m)

    def _score(self,f1,f2,ratio=0.92,reproj=6,slo=0.85,shi=1.18,rotmax=30):
        p1,d1,m1=f1; p2,d2,m2=f2
        if d1 is None or d2 is None or len(p1)<4 or len(p2)<4: return 0.0
        knn=self.bf.knnMatch(d1,d2,k=2); g=[a for pr in knn if len(pr)==2 for a,b in [pr] if a.distance<ratio*b.distance]
        if len(g)<4: return 0.0
        src=p1[[x.queryIdx for x in g]].reshape(-1,1,2); dst=p2[[x.trainIdx for x in g]].reshape(-1,1,2)
        M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=reproj,maxIters=3000,confidence=0.99)
        if M is None or mask is None: return 0.0
        a,b=M[0,0],M[1,0]; sc=np.sqrt(a*a+b*b); rot=abs(np.degrees(np.arctan2(b,a))); rot=min(rot,360-rot)
        if sc<slo or sc>shi or rot>rotmax: return 0.0
        inl=int(mask.sum())
        if inl==0: return 0.0
        H,W=m2.shape; w1=cv2.warpAffine(m1,M,(W,H),flags=cv2.INTER_NEAREST)
        ov=int(np.count_nonzero((w1>0)&(m2>0)))
        return inl*float(np.sqrt(HO/ov)) if ov>=500 else float(inl)

    def match_score(self, imgA, imgB):
        """两帧 -> 重叠归一化内点匹配分(越高越像同指)。"""
        return self._score(self.feat(imgA), self.feat(imgB))

    def enroll(self, template_imgs):
        """注册: 多帧模板 -> 预计算特征列表(供verify反复使用)。"""
        return [self.feat(t) for t in template_imgs]

    def verify(self, probe_img, enrolled, threshold=None):
        """验证: 探针 vs 已注册模板特征(或模板图列表) -> (匹配分数, 是否同指)。分数=对各模板取max。"""
        thr=self.threshold if threshold is None else threshold
        pf=self.feat(probe_img)
        feats=enrolled if (enrolled and isinstance(enrolled[0],tuple)) else [self.feat(t) for t in enrolled]
        score=max((self._score(pf,t) for t in feats), default=0.0)
        return score, bool(score>=thr)

if __name__=='__main__':
    import sys
    if len(sys.argv)>=3:
        m=FingerprintMatcher()
        s=m.match_score(sys.argv[1],sys.argv[2])
        print(f"match_score={s:.2f}  decision={'同指' if s>=m.threshold else '非同指'} (threshold={m.threshold})")
    else:
        print("用法: python predict.py <fp1.bmp> <fp2.bmp>")
