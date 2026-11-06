# Model card

**Task:** twelve independent study-level knee MRI findings.  
**Intended use:** retrospective competition research and software inspection.  
**Population:** the competition's multi-series MRI cohort; external populations have not been independently validated.

## Inputs and outputs

Inputs are physically ordered MRI slice windows, series metadata, validity masks and optional anatomical ROI bags. Radiology report text supplies training labels only. Outputs are finding scores; AUC evaluates ranking and does not establish probability calibration or diagnostic thresholds.

## Training and evaluation

Historical models use report-derived labels for 4,349 studies, with 58 structured-label studies repeatedly used for development. Available checkpoint families include ResNet, ConvNeXt, CoAtNet and custom SSL experiments. See the [report](REPORT.md) for exact distinctions among validation populations and team ensembles.

The team best observed scores are public 0.937 and private 0.920. Those are ensemble submission outcomes and must not be attached to every downloadable checkpoint. Arm C's official public/private pair is 0.920/0.898. The final ensemble is not fully recoverable from the available release.

## Limitations

- Gold58 was repeatedly reused for selection and is not independent final validation.
- Weak labels can reflect report omissions and extraction errors; OOF agreement is not clinical accuracy.
- Scanner/protocol distribution shift, subgroup performance, calibration and clinical utility are not established.
- Attention weights and anatomical crops do not constitute validated explanations or lesion localization.
- No reader study, prospective deployment or regulatory clearance is claimed.
- The checkpoint catalog deliberately includes failed/collapsed SSL controls, clearly marked as such.

## Availability

The [manifest](../models/manifest.json) links 26 upstream artifacts and their digests. The selected Arm C checkpoint and several final ensemble weights are unavailable in that release. Obtain competition data through Kaggle under its terms; no patient images or reports are redistributed here.
