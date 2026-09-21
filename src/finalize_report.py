#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
m=pd.read_csv(ROOT/'results/metrics.csv'); b=pd.read_csv(ROOT/'results/branch_metrics.csv'); cm=pd.read_csv(ROOT/'results/causal_event_metrics.csv'); fc=pd.read_csv(ROOT/'results/dual_hhsa_forecast_to_2028_12.csv')
# Event branch phase skill.
e=b[b.branch=='event']; fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].plot(e.lead_months,e.circular_mae_rad,'o-'); axs[0].set(title='Event NN circular phase MAE',xlabel='Lead months',ylabel='Radians'); axs[1].plot(e.lead_months,e.phasor_cosine,'o-'); axs[1].set(title='Event NN mean phasor cosine',xlabel='Lead months');
for ax in axs: ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT/'figures/05_event_branch_phase_skill.png',dpi=220); plt.close(fig)
# Strength IA R2 heatmap.
s=b[b.branch=='strength'].pivot(index='series',columns='lead_months',values='r2'); fig,ax=plt.subplots(figsize=(8,5)); im=ax.imshow(s,aspect='auto',cmap='RdYlBu',vmin=-1,vmax=1); ax.set_xticks(range(len(s.columns)),s.columns); ax.set_yticks(range(len(s.index)),s.index); ax.set(xlabel='Lead months',title='Strength NN: reconstructed IA R²'); fig.colorbar(im,ax=ax,label='R²'); fig.tight_layout(); fig.savefig(ROOT/'figures/06_strength_branch_ia_skill.png',dpi=220); plt.close(fig)
off=m[m.level=='signal']; p=off.pivot(index='lead_months',columns='method',values='r2'); cp=cm.pivot(index='lead_months',columns='method',values='rmse'); resource=json.load(open(ROOT/'results/resource_summary.json')); future_meta=json.load(open(ROOT/'results/future_forecast_metadata.json'))
def md(df): return df.round(3).to_markdown()
text=f'''# HHSA–Sohail Dual-Network Niño3.4 Forecast Experiment

## Objective

This independent project tests whether ENSO event timing and event strength are more predictable when modeled separately. It uses the existing local Niño3.4 record and the existing project masking-EMD/direct-quadrature implementation. It does not use the unrelated third-party `emd` package.

## Data and decomposition

- Record: 1948-01 to 2026-07, 943 monthly values, no missing values.
- First level: six oscillatory IMFs plus one residual.
- Second level: each first-level instantaneous amplitude is decomposed into AM-IMFs plus its amplitude residual.
- `upsample_level=0` is used so that the decomposition reconstructs exactly.

$$x(t)=\sum_{{i=1}}^6 IMF_i(t)+r(t)$$

$$IMF_i(t)=A_i(t)\cos\phi_i(t)$$

$$A_i(t)=\sum_j AMIMF_{{ij}}(t)+r_i^{{AM}}(t)$$

## Models

- **M0 Persistence:** $\hat x(t+k)=x(t)$.
- **M1 Direct NN:** 60 raw Niño3.4 lags directly predict future Niño3.4.
- **M2 HHSA single-state ResNet:** one HHSA state vector predicts future Niño3.4.
- **M3 Event/Strength branches:** branch skills are evaluated separately.
- **M4 Physical reconstruction:** predicted amplitude and phase are recombined analytically.
- **M5 Fusion ResNet:** a small residual network combines the branch predictions.

### Event NN — When?

The event branch receives 60 months of

$$[IMF_i,IF_i,\cos\phi_i,\sin\phi_i]$$

and predicts future phase phasors $(\widehat{{\cos\phi_i}},\widehat{{\sin\phi_i}})$. Every predicted phasor is normalized to unit length before reconstruction. This avoids the $2\pi$ phase discontinuity.

### Strength NNs — How strong?

Every real AM-IMF has its own small neural network:

$$AMIMF_{{ij}}(t-L_{{ij}}+1:t)\rightarrow\widehat{{AMIMF}}_{{ij}}(t+k).$$

The history length is selected from the training-period component timescale:

$$L_{{ij}}=\operatorname{{clip}}(2T_{{ij}},12,60).$$

The amplitude is reconstructed as

$$\widehat A_i=\sum_j\widehat{{AMIMF}}_{{ij}}+\widehat r_i^{{AM}}.$$

### M4 physical synthesis

No third neural network is required:

$$\widehat{{IMF}}_i(t+k)=\widehat A_i(t+k)\cos\widehat\phi_i(t+k)$$

$$\widehat x(t+k)=\sum_{{i=1}}^6\widehat{{IMF}}_i(t+k)+\widehat r(t+k).$$

## Offline/diagnostic results

Chronological 70/15/15 splitting is used; all scaling is fit on training data only. The decomposition itself uses the full record, so these results are explicitly diagnostic rather than real-time.

R²:

{md(p)}

M4 is the strongest HHSA method. At 3/6/9/12 months its R² is 0.874/0.792/0.692/0.619. M5 performs poorly: the validation-only fusion sample is small and the fusion network overfits. M5 therefore provides no evidence of improvement.

## Causal-prefix El Niño peak hindcasts

For the held-out 2015, 2018, 2020, and 2023 peaks, decomposition and training are repeated using only data available at each 3/6/9-month forecast origin.

RMSE (°C):

{md(cp)}

M4 does not beat persistence in this causal event experiment. Its 3/6/9-month RMSE is 1.191/1.303/1.794 °C, compared with persistence 0.632/1.011/1.617 °C. Strong events are systematically underestimated. With only four events per lead, correlations are unstable.

## Experimental forecast to 2028-12

The future file contains predicted IA, carrier/phase cosine, reconstructed IMF, residual, and final Niño3.4 for every month from 2026-08 through 2028-12. The final value for 2028-12 is {fc.forecast_nino34.iloc[-1]:.3f} °C.

This is not an operational forecast: training uses retrospective full-record HHSA features, there is no calibrated uncertainty interval, and causal peak hindcasts did not beat persistence.

## Leakage audit

- Offline M0–M5: full-record decomposition; explicitly labeled diagnostic.
- Chronological split; no random time split.
- Scaling fit on training periods only.
- M5 fusion is trained only on validation-period base predictions, not base-training predictions.
- Causal event predictors use only the prefix ending at the forecast origin.
- Rolling EMD mode identity and endpoint instability remain scientific limitations.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/run_dual_hhsa.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/run_causal_events.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/forecast_future.py
python src/finalize_report.py
```

## Outputs

- `results/metrics.csv`, `results/predictions.csv`: offline M0–M5 results.
- `results/branch_metrics.csv`: Event and Strength branch skill.
- `results/causal_event_predictions.csv`, `causal_event_metrics.csv`: held-out event hindcasts.
- `results/dual_hhsa_forecast_to_2028_12.csv`: future component and signal forecasts.
- `models/`: trained PyTorch weights.
- `figures/01_M0_M5_skill.png` through `figures/06_strength_branch_ia_skill.png`.

## Resources

Offline run: {resource['wall_s']:.1f} s internal wall time, peak Python RSS {resource['peak_rss_mib']:.1f} MiB, mean CPU {resource['mean_cpu_percent']:.1f}%. Causal event run took about 6m55s and peaked near 436 MiB RSS. Future run took about 53s wall time and peaked near 443 MiB RSS.
'''
(ROOT/'README.md').write_text(text)
print('wrote',ROOT/'README.md')
