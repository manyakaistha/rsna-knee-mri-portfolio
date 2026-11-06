# Publication validation

Validated on **6 November 2026** in a fresh Python 3.11 CPU environment. The exact installed package versions are recorded in [validation-environment.txt](validation-environment.txt). The workflow is also defined in [GitHub Actions](../.github/workflows/ci.yml).

## Executed checks

| Check | Result |
|---|---|
| Model downloader integrity | 3 tests passed: valid artifact/reuse, corrupt download rejection, preservation of an existing mismatch |
| ROI geometry and annotation suite | 84 tests discovered: 61 passed, 23 skipped because private DICOM/cohort fixtures are absent |
| Arm C model and distributed trainer suite | 10 tests passed |
| Experimental analysis utilities | 5 tests passed |
| Synthetic DICOM window headers | Distinct SOPs, physical order, same-series isolation and oversampling checks passed |
| Six portfolio recipe contracts | Forward/backward, masked perturbation and checkpoint round trips passed with tiny CPU encoders |
| Two-rank CPU/Gloo loss normalization | Passed; maximum gradient difference **7.450580596923828e−9** against the equivalent global-batch reference |
| Annotation launcher | `--help` succeeds without data or authentication |
| Publication integrity | Python syntax, cleared notebook outputs, local Markdown links, submission invariants and source/model manifests passed |

**Total unittest cases: 102 discovered, 79 passed, 23 data-dependent skips.** The standalone header, six-recipe and distributed-gradient checks are additional to that total. No pretrained downloads are required by the check runner.

## Publication adjustments

- Resolved the annotation launcher's local package explicitly and changed the laterality test to a relative import. This avoids a namespace collision with the separately installed Kaggle API package.
- Made the historical history-refresh helper tolerate the absence of an older private ledger when it is used for a new retrieval.
- Cleared notebook outputs, execution counts and widget metadata. Consistently renamed one legacy pooling class/option to neutral `GatedResidualPool` / `xgated` terminology without changing the numerical implementation.
- Removed unused starter scaffolding, duplicate generated entrypoints, local environments, private data, operational transcripts and transient artifacts from the publication copy. The original working directory was preserved.
- Added a verified, atomic model downloader, figure generator, repository-integrity check and a CPU test runner. The distributed smoke uses an explicit localhost rendezvous to avoid machine-hostname resolution problems on macOS.
- Freshly retrieved all accessible competition submissions, including the final private scores, instead of repeating the older incomplete ledger.

The source manifest stores both original and published hashes where a preserved source file was edited. Aggregate historical results remain unchanged.

## Content and artifact checks

Text sources, documentation, metadata and cleared notebook cells were scanned for common credential formats, private keys, machine-specific home/volume paths and development-assistant references. The publication includes no raw DICOM files, radiology reports, annotation databases, patient-level arrays or credentials. Repository ignore rules exclude those artifact classes. No automated scan can replace the explicit publication allowlist used to assemble this tree.

Model assets are verified against the original release byte counts and SHA-256 digests before mirroring. The new release preserves original filenames and provenance. Models are stored as release attachments, keeping Git history small.

## Limits

These checks establish software behavior on synthetic/CPU inputs, not medical accuracy. CUDA/NCCL, full-size training, real-DICOM inference, exact final-ensemble reproduction, browser interaction with the private annotation cohort, repeated seeds and independent clinical evaluation were not performed for this publication. Data-dependent skips are reported rather than counted as passing clinical tests.
