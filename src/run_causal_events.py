#!/usr/bin/env python3
"""Causal-prefix held-out El Nino hindcasts for the dual HHSA architecture."""
from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd,torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,'/g/data/p66/ars599/HHSA_WK/hhsa-python/src')
from run_dual_hhsa import CFG,fit,pred,period
from hhsa import decompose
from hhsa.instantaneous import direct_quadrature
H=CFG['history']; LEADS=[3,6,9]
def event_catalogue(y):
 warm=y>=.5; out=[]; s=None
 for i,v in enumerate(np.r_[warm,False]):
  if v and s is None:s=i
  if not v and s is not None:
   if i-s>=5: p=s+int(np.argmax(y[s:i])); out.append((s,i-1,p))
   s=None
 return out
def samples(z,k,L=None):
 L=H if L is None else L; o=np.arange(H,len(z)-k); X=np.stack([z[i-L:i] for i in o]); Y=np.stack([z[i:i+k] for i in o]); return X,Y,o
def train_predict(X,Y,O,n,seed,residual=False):
 cut=max(H+2,int(.80*n)); tr=O<cut; va=O>=cut
 if va.sum()<3: cut=int(np.quantile(O,.8)); tr=O<cut; va=O>=cut
 b=fit(X,Y,tr,va,seed,residual=residual); return pred(b,X[-1:])[0]
def one_origin(prefix,k,seed):
 n=len(prefix); r=decompose(prefix,12,max_imfs=CFG['max_imfs'],max_modulation_imfs=CFG['max_am_imfs'],mask_order=0,mask_order2=0,upsample_level=0); c=r.IMF.shape[1]-1
 freq,_,phase=direct_quadrature(r.IMF[:,:c],12); ph=np.stack([np.cos(phase),np.sin(phase)],-1); ia=np.abs(r.am[:,:c]); parts=r.IMF2[:,:,:c]; residual=r.IMF[:,-1]
 # M1 direct.
 X,Y,O=samples(prefix,k); direct=train_predict(X,Y,O,n,seed)[-1]
 # M2 single-state residual NN.
 state=np.column_stack([prefix,r.IMF[:,:c],ia,freq,ph.reshape(n,-1),parts.reshape(n,-1),residual]); so=np.arange(H,n-k); SX=state[so-1]; SY=np.stack([prefix[i:i+k] for i in so]); single=train_predict(SX,SY,so,n,seed+1,True)[-1]
 # Event branch.
 ets=np.column_stack([r.IMF[:,:c],freq,ph.reshape(n,-1)]); EX=np.stack([ets[i-H:i].reshape(-1) for i in so]); EY=np.stack([ph[i:i+k].transpose(1,0,2).reshape(-1) for i in so]); e=train_predict(EX,EY,so,n,seed+2).reshape(c,k,2); e/=np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-8)
 # Strength branch, separate network per AM component.
 ahat=np.zeros((c,k))
 for carrier in range(c):
  for j in range(parts.shape[1]):
   z=parts[:,j,carrier]
   if np.std(z[:int(.8*n)])<1e-9: continue
   L=int(np.clip(2*period(z[:int(.8*n)]),12,H)); X,Y,O=samples(z,k,L); ahat[carrier]+=train_predict(X,Y,O,n,seed+10+carrier*20+j)
 RX,RY,RO=samples(residual,k); rr=train_predict(RX,RY,RO,n,seed+500)
 physical=(ahat*e[:,:,0]).sum(0)+rr
 return float(prefix[-1]),float(direct),float(single),float(physical[-1]),c
def main():
 t0=time.time(); df=pd.read_csv(ROOT/'data/nino34_monthly.csv',parse_dates=['date']); y=df.nino34_anomaly_c.to_numpy(float); dates=df.date.to_numpy(); events=event_catalogue(y); held=[q for q in events if q[2]>=int(.85*len(y))]; rows=[]
 for eidx,(s,e,p) in enumerate(held,1):
  for lead in LEADS:
   origin=p-lead; vals=one_origin(y[:origin+1],lead,9000+eidx*100+lead)
   for method,fc in zip(('M0 Persistence','M1 Direct NN','M2 HHSA single ResNet','M4 Dual physical reconstruction'),vals[:4]):
    rows.append({'event':eidx,'event_start':str(dates[s])[:10],'peak_date':str(dates[p])[:10],'origin_date':str(dates[origin])[:10],'lead_months':lead,'method':method,'observed_peak':y[p],'predicted_peak':fc,'error':fc-y[p],'absolute_error':abs(fc-y[p]),'carriers_at_origin':vals[4]})
   print('event',eidx,'lead',lead,'M4',vals[3],flush=True)
 out=pd.DataFrame(rows); out.to_csv(ROOT/'results/causal_event_predictions.csv',index=False); mets=[]
 for (method,lead),g in out.groupby(['method','lead_months']):
  mets.append({'method':method,'lead_months':lead,'n':len(g),'correlation':g.observed_peak.corr(g.predicted_peak),'rmse':float(np.sqrt(np.mean(g.error**2))),'mae':float(g.absolute_error.mean()),'bias':float(g.error.mean())})
 pd.DataFrame(mets).to_csv(ROOT/'results/causal_event_metrics.csv',index=False)
 fig,axs=plt.subplots(2,2,figsize=(13,9),sharey=True)
 for ax,(event,g) in zip(axs.flat,out.groupby('event')):
  obs=g.observed_peak.iloc[0]; ax.axhline(obs,color='k',label=f'Observed {obs:.2f}')
  for method,h in g.groupby('method'): ax.plot(h.lead_months,h.predicted_peak,'o-',label=method)
  ax.invert_xaxis(); ax.set(title=f"Peak {g.peak_date.iloc[0]}",xlabel='Months before peak',ylabel='Niño3.4 °C'); ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=6); fig.suptitle('Causal-prefix El Niño peak hindcasts'); fig.tight_layout(); fig.savefig(ROOT/'figures/03_causal_el_nino_hindcasts.png',dpi=220); plt.close(fig)
 print(pd.DataFrame(mets).to_string(index=False)); print('elapsed_s',time.time()-t0)
if __name__=='__main__': main()
