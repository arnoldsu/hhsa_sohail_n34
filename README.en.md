# HHSA–Sohail Dual-Network Niño3.4 Forecast Experiment

## Conceptual framework

The experiment applies Sohail's neural-network approach to HHSA-derived
Niño3.4 event and strength components.

![Application of Sohail's NN with HHSA](figures/sohail_nn_hhsa_framework.png)

**Figure 1. Application of Sohail's NN with HHSA.**  
Sohail's neural-network framework is adapted to HHSA by separating
event/timing information (IMF, IF and IP) from strength/amplitude
information (IA and AM-IMFs). Separate neural networks are then used
to predict the future event phase and amplitude before reconstructing
the future Niño3.4 state.

## Objective

This independent project tests whether ENSO event timing and event strength are more predictable when modeled separately. It uses the existing local Niño3.4 record and the existing project masking-EMD/direct-quadrature implementation. It does not use the unrelated third-party `emd` package.

## Data and decomposition

- Record: 1948-01 to 2026-07, 943 monthly values, no missing values.
- First level: six oscillatory IMFs plus one residual.
- Second level: each first-level instantaneous amplitude is decomposed into AM-IMFs plus its amplitude residual.
- `upsample_level=0` is used so that the decomposition reconstructs exactly.

$$x(t)=\sum_{i=1}^6 IMF_i(t)+r(t)$$

$$IMF_i(t)=A_i(t)\cos\phi_i(t)$$

$$A_i(t)=\sum_j AMIMF_{ij}(t)+r_i^{AM}(t)$$

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

and predicts future phase phasors $(\widehat{\cos\phi_i},\widehat{\sin\phi_i})$. Every predicted phasor is normalized to unit length before reconstruction. This avoids the $2\pi$ phase discontinuity.

### Strength NNs — How strong?

Every real AM-IMF has its own small neural network:

$$AMIMF_{ij}(t-L_{ij}+1:t)ightarrow\widehat{AMIMF}_{ij}(t+k).$$

The history length is selected from the training-period component timescale:

$$L_{ij}=\operatorname{clip}(2T_{ij},12,60).$$

The amplitude is reconstructed as

$$\widehat A_i=\sum_j\widehat{AMIMF}_{ij}+\widehat r_i^{AM}.$$

### M4 physical synthesis

No third neural network is required:

$$\widehat{IMF}_i(t+k)=\widehat A_i(t+k)\cos\widehat\phi_i(t+k)$$

$$\widehat x(t+k)=\sum_{i=1}^6\widehat{IMF}_i(t+k)+\widehat r(t+k).$$

## Offline/diagnostic results

Chronological 70/15/15 splitting is used; all scaling is fit on training data only. The decomposition itself uses the full record, so these results are explicitly diagnostic rather than real-time.

R²:

|   lead_months |   M0 Persistence |   M1 Direct NN |   M2 HHSA single ResNet |   M4 Dual physical reconstruction |   M5 Dual fusion ResNet |
|--------------:|-----------------:|---------------:|------------------------:|----------------------------------:|------------------------:|
|             1 |            0.92  |          0.727 |                   0.736 |                             0.893 |                  -0.013 |
|             3 |            0.603 |          0.356 |                   0.719 |                             0.874 |                  -1.389 |
|             6 |           -0.077 |         -0.09  |                   0.713 |                             0.792 |                  -0.171 |
|             9 |           -0.677 |         -0.272 |                   0.628 |                             0.692 |                  -0.22  |
|            12 |           -1.038 |         -0.327 |                   0.509 |                             0.619 |                  -1.155 |

M4 is the strongest HHSA method. At 3/6/9/12 months its R² is 0.874/0.792/0.692/0.619. M5 performs poorly: the validation-only fusion sample is small and the fusion network overfits. M5 therefore provides no evidence of improvement.

## Causal-prefix El Niño peak hindcasts

For the held-out 2015, 2018, 2020, and 2023 peaks, decomposition and training are repeated using only data available at each 3/6/9-month forecast origin.

RMSE (°C):

|   lead_months |   M0 Persistence |   M1 Direct NN |   M2 HHSA single ResNet |   M4 Dual physical reconstruction |
|--------------:|-----------------:|---------------:|------------------------:|----------------------------------:|
|             3 |            0.632 |          1.413 |                   0.87  |                             1.191 |
|             6 |            1.011 |          1.531 |                   1.43  |                             1.303 |
|             9 |            1.617 |          1.477 |                   1.901 |                             1.794 |

M4 does not beat persistence in this causal event experiment. Its 3/6/9-month RMSE is 1.191/1.303/1.794 °C, compared with persistence 0.632/1.011/1.617 °C. Strong events are systematically underestimated. With only four events per lead, correlations are unstable.

## Experimental forecast to 2028-12

The future file contains predicted IA, carrier/phase cosine, reconstructed IMF, residual, and final Niño3.4 for every month from 2026-08 through 2028-12. The final value for 2028-12 is 1.739 °C.

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

Offline run: 31.0 s internal wall time, peak Python RSS 469.5 MiB, mean CPU 95.3%. Causal event run took about 6m55s and peaked near 436 MiB RSS. Future run took about 53s wall time and peaked near 443 MiB RSS.
