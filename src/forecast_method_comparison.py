#!/usr/bin/env python3
"""Matched future forecasts for Raw, EMD-only, Full HHSA, persistence and M4."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from run_raw_emd_hhsa_ablation import features,reduce60,fit_predict,H,SEEDS
K=29
def main():
 df=pd.read_csv(ROOT/'data/nino34_monthly.csv',parse_dates=['date']); y=df.nino34_anomaly_c.to_numpy(float); n=len(y); dates=df.date.to_numpy(); future=pd.date_range(pd.Timestamp(dates[-1])+pd.offsets.MonthBegin(1),'2028-12-01',freq='MS')
 allrows=[]; curves={'Persistence':np.repeat(y[-1],K)}
 for name,ch in features(y).items():
  origins=np.arange(H,n-K); X=np.stack([ch[o-H:o].reshape(-1) for o in origins]); Y=np.stack([y[o:o+K] for o in origins]); tr=origins<int(.85*n); va=origins>=int(.85*n); Z,sc,pca=reduce60(X,tr); latest=pca.transform(sc.transform(ch[-H:].reshape(1,-1))); members=[]
  for seed in SEEDS:
   _,pf=fit_predict(Z,Y,tr,va,np.zeros(len(origins),bool),seed+20000,future=latest); members.append(pf[0])
  a=np.stack(members); curves[name]=a.mean(0)
  for k,date in enumerate(future): allrows.append({'date':date,'lead_months':k+1,'method':name,'forecast_mean':a[:,k].mean(),'forecast_min_seed':a[:,k].min(),'forecast_max_seed':a[:,k].max(),'forecast_std_seed':a[:,k].std()})
 for k,date in enumerate(future): allrows.append({'date':date,'lead_months':k+1,'method':'Persistence','forecast_mean':y[-1],'forecast_min_seed':y[-1],'forecast_max_seed':y[-1],'forecast_std_seed':0.})
 m4=pd.read_csv(ROOT/'results/dual_hhsa_forecast_to_2028_12.csv'); curves['M4 Dual physical']=m4.forecast_nino34.to_numpy()
 for k,row in m4.iterrows(): allrows.append({'date':row.date,'lead_months':row.lead_months,'method':'M4 Dual physical','forecast_mean':row.forecast_nino34,'forecast_min_seed':np.nan,'forecast_max_seed':np.nan,'forecast_std_seed':np.nan})
 pd.DataFrame(allrows).to_csv(ROOT/'results/future_method_comparison_to_2028_12.csv',index=False)
 skill=pd.read_csv(ROOT/'results/raw_emd_hhsa_offline_summary.csv'); old=pd.read_csv(ROOT/'results/metrics.csv'); m4skill=old[(old.level=='signal')&(old.method=='M4 Dual physical reconstruction')][['lead_months','r2']]
 fig,axs=plt.subplots(1,2,figsize=(16,5.5))
 for name,g in skill.groupby('representation'): axs[0].errorbar(g.lead_months,g.r2_mean,yerr=g.r2_std,marker='o',capsize=2,label=name)
 axs[0].plot(m4skill.lead_months,m4skill.r2,'ko-',label='M4 Dual physical'); axs[0].axhline(0,color='.5',lw=.7); axs[0].set(title='Held-out offline skill (diagnostic)',xlabel='Lead months',ylabel='R²'); axs[0].grid(alpha=.2); axs[0].legend(fontsize=8)
 axs[1].plot(pd.to_datetime(df.date)[-120:],y[-120:],'k',lw=1.2,label='Observed')
 colors={'Persistence':'.5','Raw':'tab:blue','EMD-only':'tab:green','Full HHSA':'tab:orange','M4 Dual physical':'tab:red'}
 f=pd.DataFrame(allrows)
 for name in ('Persistence','Raw','EMD-only','Full HHSA','M4 Dual physical'):
  g=f[f.method==name]; axs[1].plot(pd.to_datetime(g.date),g.forecast_mean,color=colors[name],label=name)
  if name in ('Raw','EMD-only','Full HHSA'): axs[1].fill_between(pd.to_datetime(g.date),g.forecast_min_seed,g.forecast_max_seed,color=colors[name],alpha=.10)
 axs[1].axvline(pd.Timestamp(dates[-1]),color='.4',ls='--'); axs[1].axhline(.5,color='orange',ls=':',lw=.8); axs[1].axhline(-.5,color='royalblue',ls=':',lw=.8); axs[1].set(title='Experimental forecasts to 2028-12',ylabel='Niño3.4 °C'); axs[1].grid(alpha=.2); axs[1].legend(fontsize=7,ncol=2)
 fig.suptitle('Matched method capability and future forecasts'); fig.tight_layout(); fig.savefig(ROOT/'figures/04_dual_hhsa_forecast_to_2028_12.png',dpi=220); plt.close(fig)
 print(pd.DataFrame({k:v for k,v in curves.items()}).tail().to_string())
if __name__=='__main__': main()
