# -*- coding: utf-8 -*-
import os, sys, io, csv, json, math, base64, queue, hashlib, sqlite3, shutil, threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np, requests
from PIL import Image, ImageOps, ImageTk
import pillow_heif
pillow_heif.register_heif_opener()

APP_TITLE='峻爸製作 AI智慧照片分類器 2.0 R1'
ENGINE_VERSION='2.0-R1-20260925'
YUNET='face_detection_yunet_2023mar.onnx'; ARCFACE='w600k_r50.onnx'; YOLO='yolov8n.onnx'
EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.heic','.heif','.tif','.tiff'}
COCO=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
ZH={'person':'人物','bicycle':'自行車','car':'汽車','motorcycle':'機車','airplane':'飛機','bus':'公車','train':'火車','truck':'卡車','boat':'船','traffic light':'紅綠燈','bird':'鳥','cat':'貓','dog':'狗','horse':'馬','backpack':'背包','umbrella':'雨傘','handbag':'手提包','tie':'領帶','suitcase':'行李箱','sports ball':'球類','baseball bat':'棒球棒','baseball glove':'棒球手套','bottle':'瓶子','wine glass':'酒杯','cup':'杯子','fork':'叉子','knife':'刀具','spoon':'湯匙','bowl':'碗','banana':'香蕉','apple':'蘋果','sandwich':'三明治','orange':'橘子','pizza':'披薩','donut':'甜甜圈','cake':'蛋糕','chair':'椅子','couch':'沙發','potted plant':'盆栽','bed':'床','dining table':'餐桌','toilet':'馬桶','tv':'螢幕／電視','laptop':'筆電','mouse':'滑鼠','remote':'遙控器','keyboard':'鍵盤','cell phone':'手機','book':'書籍','clock':'時鐘'}
INDOOR={'chair','couch','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','sink','refrigerator','book','clock'}
FOOD={'dining table','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','pizza','donut','cake'}
SPORT={'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket'}
TRAFFIC={'bicycle','car','motorcycle','bus','train','truck','traffic light','stop sign','parking meter'}
NATURE={'bird','horse','sheep','cow','elephant','bear','zebra','giraffe'}


def rp(name): return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))/name

def appdir():
    p=Path(os.getenv('LOCALAPPDATA',str(Path.home())))/'chunba_photo_ai'; p.mkdir(parents=True,exist_ok=True); return p

def exif_date(exif):
    for k in (36867,36868,306):
        v=exif.get(k)
        if v:
            for fmt in ('%Y:%m:%d %H:%M:%S','%Y-%m-%d %H:%M:%S'):
                try:return datetime.strptime(str(v).strip(),fmt).strftime('%Y-%m-%d %H:%M:%S')
                except:pass
    return ''

def load_img(path,max_dim=1100):
    with Image.open(path) as im:
        d=exif_date(im.getexif()); im=ImageOps.exif_transpose(im).convert('RGB')
        if max(im.size)>max_dim:
            s=max_dim/max(im.size); im=im.resize((int(im.width*s),int(im.height*s)),Image.Resampling.LANCZOS)
        arr=np.asarray(im)
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR),d

def fallback_date(p):
    try:return datetime.fromtimestamp(Path(p).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except:return ''

def next_run(base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True);day=datetime.now().strftime('%Y%m%d')
    for i in range(1,10000):
        rid=f'{day}_{i:03d}'; folder=base/f'chunba_{rid}'
        if not folder.exists(): folder.mkdir(); return rid,folder
    raise RuntimeError('run id exhausted')

def signature(cfg,refs):
    d={k:v for k,v in cfg.items() if k not in ('api_key','source','output_root')}; h=hashlib.sha1(json.dumps(d,ensure_ascii=False,sort_keys=True).encode())
    for p in refs:
        try:
            st=Path(p).stat(); h.update(f'{Path(p).resolve()}|{st.st_size}|{st.st_mtime_ns}'.encode())
        except:pass
    return h.hexdigest()

class Cache:
    def __init__(self):
        self.c=sqlite3.connect(appdir()/'cache.sqlite3',check_same_thread=False); self.lock=threading.Lock()
        self.c.execute('CREATE TABLE IF NOT EXISTS c(path TEXT,size INTEGER,mtime INTEGER,sig TEXT,engine TEXT,j TEXT,PRIMARY KEY(path,size,mtime,sig,engine))');self.c.commit()
    def get(self,p,sig):
        try:
            st=Path(p).stat(); key=(str(Path(p).resolve()),st.st_size,st.st_mtime_ns,sig,ENGINE_VERSION)
            with self.lock:r=self.c.execute('SELECT j FROM c WHERE path=? AND size=? AND mtime=? AND sig=? AND engine=?',key).fetchone()
            return json.loads(r[0]) if r else None
        except:return None
    def put(self,p,sig,r):
        try:
            st=Path(p).stat(); key=(str(Path(p).resolve()),st.st_size,st.st_mtime_ns,sig,ENGINE_VERSION,json.dumps(r,ensure_ascii=False))
            with self.lock:self.c.execute('INSERT OR REPLACE INTO c VALUES(?,?,?,?,?,?)',key);self.c.commit()
        except:pass

class Yolo:
    def __init__(self):self.net=cv2.dnn.readNetFromONNX(str(rp(YOLO)))
    def run(self,img,conf=.29,size=640):
        self.net.setInput(cv2.dnn.blobFromImage(img,1/255.0,(size,size),swapRB=True,crop=False)); a=np.squeeze(self.net.forward())
        if a.ndim!=2:return {'objects':[],'person_count':0,'person_max':0}
        if a.shape[0]<=100 and a.shape[0]<a.shape[1]:a=a.T
        boxes=[];scores=[];classes=[]
        for row in a:
            sc=row[4:]; cid=int(np.argmax(sc)); s=float(sc[cid])
            if s<conf or cid>=len(COCO):continue
            cx,cy,w,h=map(float,row[:4]); boxes.append([int(cx-w/2),int(cy-h/2),int(w),int(h)]);scores.append(s);classes.append(cid)
        ids=cv2.dnn.NMSBoxes(boxes,scores,conf,.45) if boxes else []
        keep=[] if ids is None else [int(i) for i in np.array(ids).reshape(-1)]
        objs=[(COCO[classes[i]],scores[i]) for i in keep]; persons=[x for x in objs if x[0]=='person']
        return {'objects':objs,'person_count':len(persons),'person_max':max([x[1] for x in persons],default=0)}

class Faces:
    def __init__(self,score=.54):self.det=cv2.FaceDetectorYN.create(str(rp(YUNET)),'',(320,320),score,.3,5000)
    def detect(self,img):
        h,w=img.shape[:2];self.det.setInputSize((w,h));_,f=self.det.detect(img);return [] if f is None else [x.copy() for x in f]
    @staticmethod
    def iou(a,b):
        ax,ay,aw,ah=a[:4];bx,by,bw,bh=b[:4];x1,y1=max(ax,bx),max(ay,by);x2,y2=min(ax+aw,bx+bw),min(ay+ah,by+bh);inter=max(0,x2-x1)*max(0,y2-y1);u=aw*ah+bw*bh-inter;return inter/u if u else 0
    def precise(self,img):
        out=self.detect(img);h,w=img.shape[:2]
        if min(h,w)<900:return out
        tw,th=int(w*.58),int(h*.58)
        for ox,oy in [(0,0),(w-tw,0),(0,h-th),(w-tw,h-th)]:
            tile=img[oy:oy+th,ox:ox+tw]
            for f in self.detect(tile):
                f[0]+=ox;f[1]+=oy
                for j in range(4,14,2):f[j]+=ox;f[j+1]+=oy
                if not any(self.iou(f,e)>.45 for e in out):out.append(f)
        return out


def norm(v):
    v=np.asarray(v,dtype=np.float32).reshape(-1);n=float(np.linalg.norm(v));return v/n if n>1e-8 else v

class Arc:
    T=np.array([[73.5318,51.5014],[38.2946,51.6963],[56.0252,71.7366],[70.7299,92.2041],[41.5493,92.3655]],dtype=np.float32)
    def __init__(self):self.net=cv2.dnn.readNetFromONNX(str(rp(ARCFACE)));self.refs=[];self.center=None
    def emb(self,img,f):
        pts=np.asarray(f[4:14],dtype=np.float32).reshape(5,2);M,_=cv2.estimateAffinePartial2D(pts,self.T,method=cv2.LMEDS)
        if M is None:return None
        crop=cv2.warpAffine(img,M,(112,112));self.net.setInput(cv2.dnn.blobFromImage(crop,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True));return norm(self.net.forward())
    def build(self,paths,det):
        a=[]
        for p in paths:
            try:
                im,_=load_img(p,2200);fs=det.precise(im)
                if fs:
                    e=self.emb(im,max(fs,key=lambda x:float(x[2]*x[3])));a.append(e) if e is not None else None
            except:pass
        self.refs=a;self.center=norm(np.mean(np.stack(a),axis=0)) if a else None;return len(a)
    def score(self,img,faces):
        if self.center is None:return None,0
        best=None;bq=0
        for f in faces:
            x,y,w,h=map(float,f[:4])
            if min(w,h)<22:continue
            x1,y1=max(0,int(x)),max(0,int(y));x2,y2=min(img.shape[1],int(x+w)),min(img.shape[0],int(y+h));c=img[y1:y2,x1:x2]
            if c.size==0:continue
            blur=cv2.Laplacian(cv2.cvtColor(c,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var();q=min(1,min(w,h)/100)*.65+min(1,blur/120)*.35
            if q<.16:continue
            e=self.emb(img,f)
            if e is None:continue
            s=.72*float(np.dot(e,self.center))+.28*max(float(np.dot(e,r)) for r in self.refs)
            if best is None or s>best:best=s;bq=q
        return best,bq

def scene(names):
    s=set(names)
    if s&SPORT:return '運動場景'
    if s&FOOD:return '餐飲場景'
    if s&TRAFFIC:return '街景／交通'
    if s&NATURE:return '戶外／自然'
    if s&INDOOR:return '室內'
    if 'person' in s:return '人物活動'
    return '其他／待確認'

def bucket(n):
    if n is None:return 'people_unknown'
    if n<=0:return 'people_00'
    if n==1:return 'people_01'
    if n==2:return 'people_02'
    if n<=5:return 'people_03_05'
    if n<=10:return 'people_06_10'
    return 'people_11_plus'

def gemini(img,key):
    im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((768,768));b=io.BytesIO();im.save(b,'JPEG',quality=82)
    payload={'contents':[{'parts':[{'text':'只分析場景與一般物件，不做人臉身分判定。只回JSON：{"scene":"室內|人物活動|戶外／自然|街景／交通|餐飲場景|運動場景|其他／待確認","tags":["繁中標籤"],"reason":"20字內繁中理由"}'},{'inline_data':{'mime_type':'image/jpeg','data':base64.b64encode(b.getvalue()).decode()}}]}],'generationConfig':{'temperature':0.1,'responseMimeType':'application/json'}}
    r=requests.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent',headers={'x-goog-api-key':key,'Content-Type':'application/json'},json=payload,timeout=40);r.raise_for_status();t=r.json()['candidates'][0]['content']['parts'][0]['text'];return json.loads(t)

def save_csv(path,rs):
    cols=['selected','original_path','original_name','date','person_present','person_confidence','people_count','scene','local_tags','online_tags','match_score','match_level','analysis_level','ai_reason','error']
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for r in rs:
            z=r.copy();z['selected']='Y' if r.get('selected') else '';w.writerow(z)

class Preview(tk.Toplevel):
    def __init__(self,app,idx):
        super().__init__(app);self.app=app;self.idx=idx;self.title('照片大圖預覽｜峻爸製作');self.geometry('1000x740')
        b=ttk.Frame(self,padding=8);b.pack(fill='x');ttk.Button(b,text='← 上一張',command=lambda:self.move(-1)).pack(side='left');ttk.Button(b,text='下一張 →',command=lambda:self.move(1)).pack(side='left',padx=5);ttk.Button(b,text='切換選取',command=self.toggle).pack(side='left',padx=10);ttk.Button(b,text='關閉',command=self.destroy).pack(side='right')
        self.pic=ttk.Label(self,anchor='center');self.pic.pack(fill='both',expand=True);self.info=tk.Text(self,height=8,wrap='word');self.info.pack(fill='x',padx=10,pady=8);self.render()
    def move(self,d):
        v=self.app.filtered
        if self.idx in v and v:self.idx=v[(v.index(self.idx)+d)%len(v)];self.render()
    def toggle(self):self.app.results[self.idx]['selected']=not self.app.results[self.idx].get('selected');self.app.render();self.render()
    def render(self):
        r=self.app.results[self.idx]
        try:
            with Image.open(r['original_path']) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((950,500));self.ph=ImageTk.PhotoImage(im)
            self.pic.config(image=self.ph,text='')
        except:self.pic.config(image='',text='無法預覽')
        s=r.get('match_score');st='' if s in (None,'') else f'{float(s):.3f}'
        t=f"{'☑ 已選取' if r.get('selected') else '☐ 未選取'}\n檔案：{r.get('original_name')}\n日期：{r.get('date') or '未分析'}｜人數：{r.get('people_count') if r.get('people_count') is not None else '未分析'}｜場景：{r.get('scene') or '未分析'}\n指定人物：{r.get('match_level') or '未分析'} {st}\nAI內容標籤：{r.get('local_tags') or '—'}\n網路AI：{r.get('online_tags') or '—'}\nAI判斷依據：{r.get('ai_reason') or '—'}"
        self.info.delete('1.0','end');self.info.insert('1.0',t)

class App(tk.Tk):
    PAGE=24
    def __init__(self):
        super().__init__();self.title(APP_TITLE);self.geometry('1260x820');self.minsize(1040,700);self.q=queue.Queue();self.cache=Cache();self.results=[];self.filtered=[];self.refs=[];self.page=0;self.stop=False;self.run_id='';self.run_folder=None;self.th=[]
        self.src=tk.StringVar();self.dst=tk.StringVar();self.person=tk.BooleanVar(value=True);self.count=tk.BooleanVar(value=True);self.date=tk.BooleanVar(value=True);self.scn=tk.BooleanVar(value=True);self.match=tk.BooleanVar(value=False);self.prec=tk.StringVar(value='標準');self.strict=tk.StringVar(value='標準');self.cache_on=tk.BooleanVar(value=True);self.online=tk.BooleanVar(value=False);self.online_quick=tk.BooleanVar(value=False);self.key=tk.StringVar();self.fp=tk.StringVar(value='全部');self.fs=tk.StringVar(value='全部');self.fm=tk.StringVar(value='全部');self.fd=tk.StringVar();self.only=tk.BooleanVar(value=False);self.prog=tk.DoubleVar();self.ui();self.after(100,self.poll)
    def ui(self):
        m=ttk.Frame(self);m.pack(fill='both',expand=True);ttk.Label(m,text=APP_TITLE,font=('Microsoft JhengHei UI',19,'bold')).pack(pady=(8,0));ttk.Label(m,text='快速掃描預覽 → 點圖大圖確認 → 精確進階 → 分類輸出').pack();ttk.Label(m,text='製作：峻爸｜程式與輸出檔名統一使用 chunba',font=('Microsoft JhengHei UI',9,'bold')).pack(pady=(0,5))
        p=ttk.Frame(m);p.pack(fill='x',padx=12)
        for r,(t,v,c) in enumerate([('來源照片',self.src,self.pick_src),('輸出根目錄',self.dst,self.pick_dst)]):ttk.Label(p,text=t+'：',width=10).grid(row=r,column=0);ttk.Entry(p,textvariable=v).grid(row=r,column=1,sticky='ew',padx=5);ttk.Button(p,text='選擇',command=c).grid(row=r,column=2)
        p.columnconfigure(1,weight=1)
        o=ttk.LabelFrame(m,text='每次快速掃描可自由單選／複選');o.pack(fill='x',padx=12,pady=4)
        ttk.Checkbutton(o,text='純人物辨識',variable=self.person).grid(row=0,column=0,padx=5,pady=4);ttk.Label(o,text='精準度').grid(row=0,column=1);ttk.Combobox(o,textvariable=self.prec,state='readonly',width=8,values=['快速','標準','高精準']).grid(row=0,column=2);ttk.Checkbutton(o,text='人數分類',variable=self.count).grid(row=0,column=3,padx=5);ttk.Checkbutton(o,text='日期分類',variable=self.date).grid(row=0,column=4,padx=5);ttk.Checkbutton(o,text='場景分類',variable=self.scn).grid(row=0,column=5,padx=5);ttk.Checkbutton(o,text='指定人物搜尋',variable=self.match).grid(row=0,column=6,padx=5);ttk.Button(o,text='加入參考照',command=self.add_refs).grid(row=0,column=7,padx=5);ttk.Label(o,text='比對').grid(row=0,column=8);ttk.Combobox(o,textvariable=self.strict,state='readonly',width=8,values=['嚴格','標準','寬鬆']).grid(row=0,column=9);self.rlabel=ttk.Label(o,text='參考照：0 張');self.rlabel.grid(row=1,column=0,columnspan=4,sticky='w',padx=5);ttk.Checkbutton(o,text='使用快取',variable=self.cache_on).grid(row=1,column=5);ttk.Label(o,text='AI內容標籤自動保留，預覽顯示 AI 判斷依據。').grid(row=1,column=6,columnspan=4,sticky='w')
        n=ttk.LabelFrame(m,text='網路 AI 輔助（選用｜照片會傳送至服務商）');n.pack(fill='x',padx=12,pady=3);ttk.Checkbutton(n,text='Gemini 2.5 Flash-Lite 輔助場景／標籤',variable=self.online).grid(row=0,column=0,padx=5,pady=4);ttk.Label(n,text='API Key').grid(row=0,column=1);ttk.Entry(n,textvariable=self.key,show='*',width=30).grid(row=0,column=2,padx=4);ttk.Checkbutton(n,text='快速掃描也使用（較慢）',variable=self.online_quick).grid(row=0,column=3,padx=5);ttk.Label(n,text='預設僅精確進階使用；網路AI不做身分比對。').grid(row=0,column=4,padx=6)
        a=ttk.Frame(m,padding=(12,5));a.pack(fill='x');self.bscan=ttk.Button(a,text='① 快速掃描預覽',command=self.start_scan);self.bscan.pack(side='left');self.badv=ttk.Button(a,text='② 精確進階分析所選',command=self.start_adv,state='disabled');self.badv.pack(side='left',padx=5);self.bexp=ttk.Button(a,text='③ 輸出所選分類檔案',command=self.start_export,state='disabled');self.bexp.pack(side='left',padx=5);ttk.Button(a,text='停止',command=self.stop_now).pack(side='left',padx=10);ttk.Button(a,text='開啟本次資料夾',command=self.open_run).pack(side='left');self.runlabel=ttk.Label(a,text='本次編號：尚未開始');self.runlabel.pack(side='right')
        f=ttk.LabelFrame(m,text='預覽篩選／整頁選取');f.pack(fill='x',padx=12,pady=3);ttk.Label(f,text='人數').grid(row=0,column=0);ttk.Combobox(f,textvariable=self.fp,state='readonly',width=9,values=['全部','0','1','2','3-5','6-10','11+']).grid(row=0,column=1);ttk.Label(f,text='場景').grid(row=0,column=2,padx=(7,0));ttk.Combobox(f,textvariable=self.fs,state='readonly',width=13,values=['全部','室內','人物活動','戶外／自然','街景／交通','餐飲場景','運動場景','其他／待確認']).grid(row=0,column=3);ttk.Label(f,text='人物比對').grid(row=0,column=4,padx=(7,0));ttk.Combobox(f,textvariable=self.fm,state='readonly',width=13,values=['全部','高度符合','疑似符合','低品質待確認','不符合','未偵測人臉']).grid(row=0,column=5);ttk.Label(f,text='日期含').grid(row=0,column=6,padx=(7,0));ttk.Entry(f,textvariable=self.fd,width=10).grid(row=0,column=7);ttk.Checkbutton(f,text='只看已選',variable=self.only,command=self.apply).grid(row=0,column=8,padx=5);ttk.Button(f,text='套用',command=self.apply).grid(row=0,column=9);ttk.Button(f,text='全選此頁',command=self.select_page).grid(row=0,column=10,padx=3);ttk.Button(f,text='全選篩選結果',command=self.select_all).grid(row=0,column=11,padx=3);ttk.Button(f,text='清除選取',command=self.clear).grid(row=0,column=12,padx=3)
        s=ttk.Frame(m);s.pack(fill='x',padx=12);ttk.Progressbar(s,variable=self.prog,maximum=100).pack(fill='x');self.status=ttk.Label(s,text='請選來源照片後開始。');self.status.pack(anchor='w')
        nav=ttk.Frame(m);nav.pack(fill='x',padx=12,pady=(3,0));ttk.Button(nav,text='← 上一頁',command=lambda:self.chpage(-1)).pack(side='left');ttk.Button(nav,text='下一頁 →',command=lambda:self.chpage(1)).pack(side='left',padx=5);self.plabel=ttk.Label(nav,text='第 0 / 0 頁');self.plabel.pack(side='left',padx=8);ttk.Label(nav,text='點照片＝大圖預覽；勾選＝精確分析／輸出。').pack(side='right')
        self.cv=tk.Canvas(m,highlightthickness=0);self.cv.pack(fill='both',expand=True,padx=12,pady=(3,8));sb=ttk.Scrollbar(self.cv,orient='vertical',command=self.cv.yview);self.cv.configure(yscrollcommand=sb.set);sb.pack(side='right',fill='y');self.cards=ttk.Frame(self.cv);self.cw=self.cv.create_window((0,0),window=self.cards,anchor='nw');self.cards.bind('<Configure>',lambda e:self.cv.configure(scrollregion=self.cv.bbox('all')));self.cv.bind('<Configure>',lambda e:self.cv.itemconfigure(self.cw,width=max(100,e.width-18)))
    def pick_src(self):
        p=filedialog.askdirectory();
        if p:self.src.set(p);self.dst.set(self.dst.get() or str(Path(p).parent/'chunba_output'))
    def pick_dst(self):
        p=filedialog.askdirectory();
        if p:self.dst.set(p)
    def add_refs(self):
        for f in filedialog.askopenfilenames(filetypes=[('Images','*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp *.tif *.tiff')]):
            if f not in self.refs:self.refs.append(f)
        self.match.set(True);self.rlabel.config(text=f'參考照：{len(self.refs)} 張｜'+ '、'.join(Path(x).name for x in self.refs[:5]))
    def cfg(self):return {'source':self.src.get(),'output_root':self.dst.get(),'person':self.person.get(),'count':self.count.get(),'date':self.date.get(),'scene':self.scn.get(),'match':self.match.get(),'precision':self.prec.get(),'strict':self.strict.get(),'cache':self.cache_on.get(),'online':self.online.get(),'online_quick':self.online_quick.get(),'api_key':self.key.get().strip()}
    def valid(self,c):
        if not Path(c['source']).is_dir():messagebox.showwarning('提示','請選來源資料夾');return False
        if not c['output_root']:messagebox.showwarning('提示','請選輸出根目錄');return False
        if not any(c[k] for k in ('person','count','date','scene','match')):messagebox.showwarning('提示','至少選一項掃描功能');return False
        if c['match'] and not self.refs:messagebox.showwarning('提示','指定人物搜尋需加入參考照');return False
        if c['online'] and not c['api_key']:messagebox.showwarning('提示','啟用網路AI需輸入 Gemini API Key');return False
        return True
    def engines(self,c,precise=False):
        y=Yolo() if c['person'] or c['count'] or c['scene'] else None;d=a=None
        if c['match']:
            d=Faces(.48 if precise else {'快速':.62,'標準':.54,'高精準':.46}[c['precision']]);a=Arc();n=a.build(self.refs,d)
            if not n:raise RuntimeError('參考照片沒有可用的人臉模板')
        return y,d,a
    def analyze(self,p,c,y,d,a,precise=False,online=False):
        md=2600 if precise else {'快速':820,'標準':1100,'高精準':1500}[c['precision']];img,dt=load_img(p,md);r={'original_path':str(p),'original_name':Path(p).name,'date':'','person_present':'','person_confidence':'','people_count':None,'scene':'','local_tags':'','online_tags':'','match_score':None,'match_level':'','analysis_level':'精確進階' if precise else '快速掃描','ai_reason':'','error':'','selected':False};why=[]
        if c['date']:r['date']=dt or fallback_date(p);why.append('日期：EXIF優先，無EXIF則檔案日期')
        if y:
            z=y.run(img,.22 if precise else {'快速':.36,'標準':.29,'高精準':.24}[c['precision']],960 if precise else 640);names=[x[0] for x in z['objects']];seen=[]
            for n in names:
                q=ZH.get(n,n)
                if q not in seen:seen.append(q)
            r['local_tags']='、'.join(seen[:12])
            if c['person']:r['person_present']='是' if z['person_count'] else '否';r['person_confidence']=round(z['person_max'],4);why.append(f"純人物：person={z['person_count']}，最高信心{z['person_max']:.2f}")
            if c['count']:r['people_count']=z['person_count'];why.append(f"人數：NMS後 {z['person_count']} 人")
            if c['scene']:r['scene']=scene(names);why.append('場景：本機AI標籤→'+r['scene'])
        if c['match']:
            fs=d.precise(img) if precise else d.detect(img);sc,q=a.score(img,fs);r['match_score']=sc;hi,mi={'嚴格':(.58,.48),'標準':(.51,.41),'寬鬆':(.45,.35)}[c['strict']]
            r['match_level']='低品質待確認' if sc is None and fs else ('未偵測人臉' if sc is None else ('高度符合' if sc>=hi else ('疑似符合' if sc>=mi else '不符合')));why.append(f"指定人物：ArcFace {'' if sc is None else f'{sc:.3f}'}，品質{q:.2f}")
        if online and c['online']:
            try:g=gemini(img,c['api_key']);r['scene']=g.get('scene') or r['scene'];r['online_tags']='、'.join(g.get('tags',[])[:10]);why.append('網路AI：'+str(g.get('reason',''))[:60])
            except Exception as e:why.append('網路AI失敗：'+str(e)[:80])
        r['ai_reason']='｜'.join(why);return r
    def stop_now(self):self.stop=True;self.status.config(text='正在停止…')
    def start_scan(self):
        c=self.cfg();
        if not self.valid(c):return
        self.stop=False;self.bscan.config(state='disabled');self.badv.config(state='disabled');self.bexp.config(state='disabled');self.results=[];self.filtered=[];self.page=0;self.run_id,self.run_folder=next_run(c['output_root']);self.runlabel.config(text='本次編號：chunba_'+self.run_id);threading.Thread(target=self.scan_worker,args=(c,),daemon=True).start()
    def scan_worker(self,c):
        try:
            files=[p for p in Path(c['source']).rglob('*') if p.is_file() and p.suffix.lower() in EXTS];y,d,a=self.engines(c);sig=signature(c,self.refs);out=[];hit=0
            for i,p in enumerate(files,1):
                if self.stop:break
                z=self.cache.get(p,sig) if c['cache'] and not(c['online'] and c['online_quick']) else None
                if z:z['selected']=False;out.append(z);hit+=1
                else:
                    try:r=self.analyze(p,c,y,d,a,False,c['online'] and c['online_quick']);out.append(r);save=r.copy();save.pop('selected',None);self.cache.put(p,sig,save) if c['cache'] and not(c['online'] and c['online_quick']) else None
                    except Exception as e:out.append({'original_path':str(p),'original_name':p.name,'date':'','person_present':'','person_confidence':'','people_count':None,'scene':'','local_tags':'','online_tags':'','match_score':None,'match_level':'','analysis_level':'快速掃描','ai_reason':'','error':str(e),'selected':False})
                if i%2==0 or i==len(files):self.q.put(('progress',100*i/max(1,len(files)),f'快速掃描 {i}/{len(files)}｜快取 {hit}'))
            save_csv(self.run_folder/f'chunba_{self.run_id}_preview.csv',out);self.q.put(('scan_done',out))
        except Exception as e:self.q.put(('error',str(e)))
    def selected(self):return [i for i,r in enumerate(self.results) if r.get('selected')]
    def start_adv(self):
        ids=self.selected();c=self.cfg()
        if not ids:messagebox.showwarning('提示','請先勾選要精確分析的照片');return
        if not self.valid(c):return
        self.stop=False;self.badv.config(state='disabled');self.bexp.config(state='disabled');threading.Thread(target=self.adv_worker,args=(c,ids),daemon=True).start()
    def adv_worker(self,c,ids):
        try:
            y,d,a=self.engines(c,True)
            for n,idx in enumerate(ids,1):
                if self.stop:break
                try:r=self.analyze(self.results[idx]['original_path'],c,y,d,a,True,c['online']);r['selected']=True;self.results[idx]=r
                except Exception as e:self.results[idx]['error']=str(e)
                self.q.put(('progress',100*n/max(1,len(ids)),f'精確進階 {n}/{len(ids)}'))
            save_csv(self.run_folder/f'chunba_{self.run_id}_advanced.csv',self.results);self.q.put(('adv_done',))
        except Exception as e:self.q.put(('error',str(e)))
    def export_path(self,r,c):
        x=[]
        if c['date']:x.append('date_'+(r.get('date','')[:7] or 'unknown'))
        if c['scene']:x.append({'室內':'scene_indoor','人物活動':'scene_people_event','戶外／自然':'scene_outdoor_nature','街景／交通':'scene_street_traffic','餐飲場景':'scene_food','運動場景':'scene_sport','其他／待確認':'scene_other'}.get(r.get('scene'),'scene_unknown'))
        if c['count']:x.append(bucket(r.get('people_count')))
        if c['match']:x.append({'高度符合':'match_high','疑似符合':'match_maybe','低品質待確認':'match_low_quality','不符合':'match_no','未偵測人臉':'match_no_face'}.get(r.get('match_level'),'match_not_run'))
        if c['person']:x.append('person_yes' if r.get('person_present')=='是' else 'person_no')
        return Path(*x) if x else Path('selected')
    def start_export(self):
        ids=self.selected();c=self.cfg()
        if not ids:messagebox.showwarning('提示','請先選照片');return
        self.bexp.config(state='disabled');threading.Thread(target=self.export_worker,args=(c,ids),daemon=True).start()
    def export_worker(self,c,ids):
        try:
            base=self.run_folder/'classified';base.mkdir(exist_ok=True);rows=[]
            for n,idx in enumerate(ids,1):
                r=self.results[idx];src=Path(r['original_path']);folder=base/self.export_path(r,c);folder.mkdir(parents=True,exist_ok=True);ext=src.suffix.lower() or '.jpg';dest=folder/f'chunba_{self.run_id}_{n:05d}{ext}';k=2
                while dest.exists():dest=folder/f'chunba_{self.run_id}_{n:05d}_{k}{ext}';k+=1
                shutil.copy2(src,dest);z=r.copy();z['export_path']=str(dest);rows.append(z);self.q.put(('progress',100*n/max(1,len(ids)),f'分類輸出 {n}/{len(ids)}'))
            with open(self.run_folder/f'chunba_{self.run_id}_export_manifest.csv','w',newline='',encoding='utf-8-sig') as f:
                cols=['original_path','original_name','export_path','date','person_present','person_confidence','people_count','scene','local_tags','online_tags','match_score','match_level','analysis_level','ai_reason','error'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
            self.q.put(('export_done',len(rows)))
        except Exception as e:self.q.put(('error',str(e)))
    def apply(self):
        out=[]
        for i,r in enumerate(self.results):
            if self.only.get() and not r.get('selected'):continue
            p=self.fp.get();n=r.get('people_count');ok=p=='全部' or (p=='0' and n==0) or (p=='1' and n==1) or (p=='2' and n==2) or (p=='3-5' and n is not None and 3<=n<=5) or (p=='6-10' and n is not None and 6<=n<=10) or (p=='11+' and n is not None and n>=11)
            if not ok:continue
            if self.fs.get()!='全部' and r.get('scene')!=self.fs.get():continue
            if self.fm.get()!='全部' and r.get('match_level')!=self.fm.get():continue
            if self.fd.get().strip() and self.fd.get().strip() not in r.get('date',''):continue
            out.append(i)
        self.filtered=out;self.page=0;self.render()
    def select_page(self):
        st=self.page*self.PAGE
        for i in self.filtered[st:st+self.PAGE]:self.results[i]['selected']=True
        self.render()
    def select_all(self):
        for i in self.filtered:self.results[i]['selected']=True
        self.render()
    def clear(self):
        for r in self.results:r['selected']=False
        self.render()
    def chpage(self,d):
        pages=max(1,math.ceil(len(self.filtered)/self.PAGE)) if self.filtered else 1;self.page=max(0,min(pages-1,self.page+d));self.render()
    def thumb(self,p):
        try:
            with Image.open(p) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((180,118));bg=Image.new('RGB',(180,118),(28,28,28));bg.paste(im,((180-im.width)//2,(118-im.height)//2));return ImageTk.PhotoImage(bg)
        except:return None
    def render(self):
        for w in self.cards.winfo_children():w.destroy()
        self.th=[];total=len(self.filtered);pages=math.ceil(total/self.PAGE) if total else 0;st=self.page*self.PAGE;ids=self.filtered[st:st+self.PAGE]
        for pos,idx in enumerate(ids):
            r=self.results[idx];c=ttk.Frame(self.cards,relief='ridge',padding=4);c.grid(row=pos//6,column=pos%6,padx=4,pady=4);ph=self.thumb(r['original_path']);self.th.append(ph);tk.Button(c,image=ph if ph else '',text='' if ph else '無預覽',width=180,height=118,command=lambda i=idx:Preview(self,i)).pack();v=tk.BooleanVar(value=r.get('selected'));ttk.Checkbutton(c,text='選取',variable=v,command=lambda i=idx,x=v:self.setsel(i,x.get())).pack(anchor='w');ttk.Label(c,text=f"{'人數 '+str(r.get('people_count')) if r.get('people_count') is not None else ''} {r.get('scene','')}"[:26],width=25).pack(anchor='w');
            
        self.plabel.config(text=f'第 {self.page+1 if pages else 0} / {pages} 頁｜篩選 {total} 張｜已選 {len(self.selected())} 張');self.badv.config(state='normal' if self.results else 'disabled');self.bexp.config(state='normal' if self.results else 'disabled')
    def setsel(self,i,v):self.results[i]['selected']=bool(v);self.plabel.config(text=self.plabel.cget('text').split('｜已選')[0]+f'｜已選 {len(self.selected())} 張')
    def open_run(self):
        if self.run_folder and self.run_folder.exists():os.startfile(self.run_folder)
        else:messagebox.showinfo('提示','尚未建立本次資料夾')
    def poll(self):
        try:
            while True:
                m=self.q.get_nowait();k=m[0]
                if k=='progress':self.prog.set(m[1]);self.status.config(text=m[2])
                elif k=='scan_done':self.results=m[1];self.bscan.config(state='normal');self.apply();self.prog.set(100);self.status.config(text=f'快速掃描完成 {len(self.results)} 張｜點圖可大圖預覽')
                elif k=='adv_done':self.badv.config(state='normal');self.bexp.config(state='normal');self.apply();self.prog.set(100);self.status.config(text='精確進階完成，已寫入本次 advanced CSV')
                elif k=='export_done':self.bexp.config(state='normal');self.prog.set(100);messagebox.showinfo('完成',f'已輸出 {m[1]} 張｜chunba_{self.run_id}')
                elif k=='error':self.bscan.config(state='normal');self.badv.config(state='normal' if self.results else 'disabled');self.bexp.config(state='normal' if self.results else 'disabled');messagebox.showerror('錯誤',m[1])
        except queue.Empty:pass
        self.after(100,self.poll)

if __name__=='__main__':App().mainloop()
