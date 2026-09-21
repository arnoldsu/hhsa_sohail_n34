#!/usr/bin/env python3
"""Matched Raw vs EMD-only vs first-level Full-HHSA Sohail ResNet ablation."""
from pathlib import Path
import sys,time,json,random
import numpy as np,pandas as pd,torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,'/g/data/p66/ars599/HHSA_WK/hhsa-python/src')
from hhsa import decompose
from hhsa.instantaneous import direct_quadrature
H=60; LEADS=[1,3,6,9,12]; K=max(LEADS); RIDX=np.array(LEADS)-1; SEEDS=[42,52,62,72,82]
class SohailResNet(nn.Module):
 def __init__(self,nout):
  super().__init__(); self.input=nn.Linear(60,64); self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(64,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64)) for _ in range(3)]); self.output=nn.Linear(64,nout)
 def forward(self,x):
  x=torch.relu(self.input(x))
  for block in self.blocks: x=torch.relu(x+block(x))
  return self.output(x)
def windows(channels,n,K=K):
 origins=np.arange(H,n-K); X=np.stack([channels[o-H:o].reshape(-1) for o in origins]); return X,origins
def target(y,origins,K=K): return np.stack([y[o:o+K] for o in origins])[:,np.array([l for l in LEADS if l<=K])-1]
def split(origins,n): return origins<int(.70*n),(origins>=int(.70*n))&(origins<int(.85*n)),origins>=int(.85*n)
def reduce60(X,tr):
 sc=StandardScaler().fit(X[tr]); Z=sc.transform(X); p=PCA(n_components=60,random_state=42).fit(Z[tr]); return p.transform(Z),sc,p
def fit_predict(X,Y,tr,va,te,seed,future=None):
 torch.manual_seed(seed); np.random.seed(seed); ym=Y[tr].mean(0); ys=np.maximum(Y[tr].std(0),1e-7); yy=(Y-ym)/ys; m=SohailResNet(Y.shape[1]); opt=torch.optim.Adam(m.parameters(),1e-3); dl=DataLoader(TensorDataset(torch.tensor(X[tr],dtype=torch.float32),torch.tensor(yy[tr],dtype=torch.float32)),32,shuffle=True,generator=torch.Generator().manual_seed(seed)); xv=torch.tensor(X[va],dtype=torch.float32); yv=torch.tensor(yy[va],dtype=torch.float32); best=None; bl=np.inf; stale=0
 for _ in range(150):
  m.train()
  for bx,by in dl: opt.zero_grad(); loss=((m(bx)-by)**2).mean(); loss.backward(); opt.step()
  m.eval()
  with torch.no_grad(): vl=float(((m(xv)-yv)**2).mean())
  if vl<bl-1e-6: bl=vl; best={k:v.clone() for k,v in m.state_dict().items()}; stale=0
  else: stale+=1
  if stale>=20: break
 m.load_state_dict(best); m.eval()
 with torch.no_grad(): pt=m(torch.tensor(X[te],dtype=torch.float32)).numpy()*ys+ym; pf=None if future is None else m(torch.tensor(future,dtype=torch.float32)).numpy()*ys+ym
 return pt,pf
def features(y):
 r=decompose(y,12,max_imfs=7,max_modulation_imfs=5,mask_order=0,mask_order2=0,upsample_level=0); c=r.IMF.shape[1]-1; freq,_,phase=direct_quadrature(r.IMF[:,:c],12)
 raw=y[:,None]; emd=r.IMF; full=np.column_stack([r.IMF,np.abs(r.am[:,:c]),freq,np.cos(phase),np.sin(phase)])
 return {'Raw':raw,'EMD-only':emd,'Full HHSA':full}
def metrics(obs,fc):
 return {'correlation':float(np.corrcoef(obs,fc)[0,1]),'rmse':float(np.sqrt(np.mean((fc-obs)**2))),'mae':float(np.mean(abs(fc-obs))),'r2':float(1-np.sum((fc-obs)**2)/np.sum((obs-obs.mean())**2))}
def catalogue(y):
 out=[]; s=None
 for i,v in enumerate(np.r_[y>=.5,False]):
  if v and s is None:s=i
  if not v and s is not None:
   if i-s>=5: out.append((s,i-1,s+int(np.argmax(y[s:i]))))
   s=None
 return out
def offline(y):
 n=len(y); rows=[]; preds=[]; fs=features(y)
 for name,ch in fs.items():
  X,O=windows(ch,n); Y=target(y,O); tr,va,te=split(O,n); Z,_,_=reduce60(X,tr)
  for seed in SEEDS:
   P,_=fit_predict(Z,Y,tr,va,te,seed)
   for q,l in enumerate(LEADS): rows.append({'experiment':'OFFLINE_DIAGNOSTIC','representation':name,'seed':seed,'lead_months':l,'n':te.sum(),**metrics(Y[te,q],P[:,q])})
  # Save one reproducible seed's predictions.
  P,_=fit_predict(Z,Y,tr,va,te,SEEDS[0])
  for q,l in enumerate(LEADS):
   for k,o in enumerate(O[te]): preds.append({'representation':name,'origin_index':o,'target_index':o+l-1,'lead_months':l,'observed':Y[te,q][k],'predicted':P[k,q]})
 return rows,preds
def causal_events(y,dates):
 rows=[]; held=[x for x in catalogue(y) if x[2]>=int(.85*len(y))]
 for ei,(s,e,p) in enumerate(held,1):
  for lead in (3,6,9):
   origin=p-lead; yy=y[:origin+1]; n=len(yy); fs=features(yy)
   for name,ch in fs.items():
    X,O=windows(ch,n,K=lead); Y=np.stack([yy[o:o+lead] for o in O]); tr=O<int(.80*n); va=O>=int(.80*n); Z,sc,pca=reduce60(X,tr); latest=pca.transform(sc.transform(ch[-H:].reshape(1,-1)))
    for seed in SEEDS[:3]:
     _,PF=fit_predict(Z,Y,tr,va,np.zeros(len(O),bool),seed+ei*100+lead,future=latest); fc=float(PF[0,-1]); rows.append({'event':ei,'peak_date':str(dates[p])[:10],'origin_date':str(dates[origin])[:10],'representation':name,'seed':seed,'lead_months':lead,'observed_peak':y[p],'predicted_peak':fc,'error':fc-y[p],'absolute_error':abs(fc-y[p])})
   print('causal event',ei,'lead',lead,flush=True)
 return rows
def main():
 t=time.time(); df=pd.read_csv(ROOT/'data/nino34_monthly.csv',parse_dates=['date']); y=df.nino34_anomaly_c.to_numpy(float); dates=df.date.to_numpy(); off,predrows=offline(y); pd.DataFrame(off).to_csv(ROOT/'results/raw_emd_hhsa_offline_by_seed.csv',index=False); pd.DataFrame(predrows).to_csv(ROOT/'results/raw_emd_hhsa_offline_predictions.csv',index=False)
 cr=causal_events(y,dates); pd.DataFrame(cr).to_csv(ROOT/'results/raw_emd_hhsa_causal_events.csv',index=False)
 osum=pd.DataFrame(off).groupby(['representation','lead_months']).agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),r2_mean=('r2','mean'),r2_std=('r2','std'),correlation_mean=('correlation','mean')).reset_index(); osum.to_csv(ROOT/'results/raw_emd_hhsa_offline_summary.csv',index=False)
 cframe=pd.DataFrame(cr); cmean=cframe.groupby(['event','peak_date','representation','lead_months'],as_index=False).agg(observed_peak=('observed_peak','first'),predicted_peak=('predicted_peak','mean'),seed_std=('predicted_peak','std')); cmean['error']=cmean.predicted_peak-cmean.observed_peak; cmean['absolute_error']=cmean.error.abs(); cmean.to_csv(ROOT/'results/raw_emd_hhsa_causal_event_means.csv',index=False); csum=cmean.groupby(['representation','lead_months']).agg(n_events=('event','size'),rmse=('error',lambda x:float(np.sqrt(np.mean(x*x)))),mae=('absolute_error','mean'),bias=('error','mean'),mean_seed_std=('seed_std','mean')).reset_index(); csum.to_csv(ROOT/'results/raw_emd_hhsa_causal_summary.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(11,4));
 for name,g in osum.groupby('representation'): axs[0].errorbar(g.lead_months,g.r2_mean,yerr=g.r2_std,marker='o',label=name); axs[1].errorbar(g.lead_months,g.rmse_mean,yerr=g.rmse_std,marker='o',label=name)
 axs[0].set(title='Offline matched ResNet',ylabel='R²'); axs[1].set(title='Offline matched ResNet',ylabel='RMSE');
 for ax in axs: ax.set_xlabel('Lead months'); ax.grid(alpha=.2); ax.legend()
 fig.tight_layout(); fig.savefig(ROOT/'figures/07_raw_emd_hhsa_offline_ablation.png',dpi=220); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for name,g in csum.groupby('representation'): ax.plot(g.lead_months,g.rmse,'o-',label=name)
 ax.set(title='Causal El Niño peak ablation',xlabel='Lead months',ylabel='RMSE °C'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/08_raw_emd_hhsa_causal_ablation.png',dpi=220); plt.close(fig)
 meta={'representations':['Raw','EMD-only','Full HHSA'],'excluded':'second-level AM-IMFs','common_input_dimension_after_training_only_PCA':60,'common_network':'Sohail Dense64 + 3 residual 256-128-64 blocks','offline_seeds':SEEDS,'causal_seeds':SEEDS[:3],'elapsed_s':time.time()-t}; (ROOT/'results/raw_emd_hhsa_ablation_metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(osum.to_string(index=False)); print(csum.to_string(index=False)); print(json.dumps(meta))
if __name__=='__main__': main()
