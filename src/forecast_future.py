#!/usr/bin/env python3
"""Future forecast with Event NN + per-AM-IMF Strength NNs + physical synthesis."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,'/g/data/p66/ars599/HHSA_WK/hhsa-python/src')
from run_dual_hhsa import CFG,fit,pred,period
from hhsa import decompose
from hhsa.instantaneous import direct_quadrature
H=CFG['history']; K=29
def samples(z,L=H):
 o=np.arange(H,len(z)-K); X=np.stack([z[i-L:i] for i in o]); Y=np.stack([z[i:i+K] for i in o]); return X,Y,o
def train_future(X,Y,O,n,seed,residual=False):
 tr=O<int(.85*n); va=O>=int(.85*n); b=fit(X,Y,tr,va,seed,residual=residual); return b
def main():
 t0=time.time(); df=pd.read_csv(ROOT/'data/nino34_monthly.csv',parse_dates=['date']); y=df.nino34_anomaly_c.to_numpy(float); dates=df.date.to_numpy(); n=len(y)
 r=decompose(y,12,max_imfs=CFG['max_imfs'],max_modulation_imfs=CFG['max_am_imfs'],mask_order=0,mask_order2=0,upsample_level=0); c=r.IMF.shape[1]-1; freq,_,phase=direct_quadrature(r.IMF[:,:c],12); ph=np.stack([np.cos(phase),np.sin(phase)],-1); ia=np.abs(r.am[:,:c]); parts=r.IMF2[:,:,:c]; residual=r.IMF[:,-1]
 origins=np.arange(H,n-K)
 # Event branch.
 ets=np.column_stack([r.IMF[:,:c],freq,ph.reshape(n,-1)]); EX=np.stack([ets[o-H:o].reshape(-1) for o in origins]); EY=np.stack([ph[o:o+K].transpose(1,0,2).reshape(-1) for o in origins]); eb=train_future(EX,EY,origins,n,12000); latestE=ets[-H:].reshape(1,-1); ep=pred(eb,latestE).reshape(c,K,2); ep/=np.maximum(np.linalg.norm(ep,axis=-1,keepdims=True),1e-8); torch.save(eb[0].state_dict(),ROOT/'models/future_event_nn.pt')
 # Strength components.
 ahat=np.zeros((c,K))
 for i in range(c):
  for j in range(parts.shape[1]):
   z=parts[:,j,i]
   if np.std(z[:int(.85*n)])<1e-9: continue
   L=int(np.clip(2*period(z[:int(.85*n)]),12,H)); X,Y,O=samples(z,L); b=train_future(X,Y,O,n,12100+i*20+j); ahat[i]+=pred(b,z[-L:][None])[0]; torch.save(b[0].state_dict(),ROOT/f'models/future_strength_IMF{i+1}_AMIMF{j+1}.pt')
 RX,RY,RO=samples(residual); rb=train_future(RX,RY,RO,n,12900); rr=pred(rb,residual[-H:][None])[0]; torch.save(rb[0].state_dict(),ROOT/'models/future_residual_nn.pt')
 carrier=ep[:,:,0]; imfhat=ahat*carrier; signal=imfhat.sum(0)+rr; future=pd.date_range(pd.Timestamp(dates[-1])+pd.offsets.MonthBegin(1),'2028-12-01',freq='MS')
 out=pd.DataFrame({'date':future,'lead_months':np.arange(1,K+1),'forecast_nino34':signal,'forecast_residual':rr})
 for i in range(c): out[f'forecast_IA_IMF{i+1}']=ahat[i]; out[f'forecast_carrier_IMF{i+1}']=carrier[i]; out[f'forecast_IMF{i+1}']=imfhat[i]
 out.to_csv(ROOT/'results/dual_hhsa_forecast_to_2028_12.csv',index=False)
 fig,ax=plt.subplots(figsize=(13,5)); ax.plot(pd.to_datetime(df.date)[-120:],y[-120:],'k',label='Observed'); ax.plot(future,signal,'r-o',ms=3,label='M4 dual physical forecast'); ax.axvline(pd.Timestamp(dates[-1]),color='.4',ls='--'); ax.axhline(.5,color='orange',ls=':'); ax.axhline(-.5,color='royalblue',ls=':'); ax.set(title='Dual-NN HHSA experimental forecast to 2028-12',ylabel='Niño3.4 °C'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/04_dual_hhsa_forecast_to_2028_12.png',dpi=220); plt.close(fig)
 meta={'last_observation':str(dates[-1])[:10],'forecast_start':str(future[0].date()),'forecast_end':str(future[-1].date()),'horizon_months':K,'method':'M4 Event NN + individual AM-IMF Strength NNs + A*cos(phase) physical reconstruction','warning':'Experimental; retrospective full-record decomposition for training; no calibrated uncertainty; causal event hindcasts did not beat persistence.','elapsed_s':time.time()-t0}; (ROOT/'results/future_forecast_metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(out.to_string(index=False)); print(json.dumps(meta))
if __name__=='__main__': main()
