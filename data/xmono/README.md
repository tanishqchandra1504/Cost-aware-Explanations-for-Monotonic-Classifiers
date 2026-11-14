# Data from XMono

Files in this folder were downloaded from the authors' repository
https://github.com/jpmarquessilva/xmono (GPL-2.0, see LICENSE_XMONO.txt):

- `spec/`  feature names and domains of the StudentGrades and BankruptcyRisk benchmarks
- `inst/`  the instances XMono was run on
- `logs/xmono/` XMono's output (all AXps and CXps per instance), used in E0 to check our implementation
- `exps_*.txt` instance lists used by XMono's `run-xmono` script

We did not copy any of XMono's Perl code. The two classifiers (`grd_classifier`,
`rdt_classifier` in `scripts/HardCoded.pm`) were rewritten in Python in
`models/xmono_models.py`.
