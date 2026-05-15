# External Code

This directory contains vendored evaluation code that is not part of the
`prompt_attack` package.

## `pytorch_fid`

FID implementation copied from the existing unified attack evaluation project:
`D:\code\natural attack\unified\eval\fid`.

Local changes:
- the Inception weight file is stored under `pytorch_fid/weights/`
- `src/prompt_attack/metrics/fid.py` is the only project-facing adapter

