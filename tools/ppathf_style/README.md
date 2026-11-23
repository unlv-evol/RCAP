# PPatHF-style treatment (roadmap item 8)

Runs PPatHF's method (Pan et al., ISSTA'24) on RCAP's function units, swaps those answers into RCAP's
PR builds, and compares the two treatments unit by unit.

- `generate.py`: asks the model with PPatHF's prompt format (same model settings as RCAP).
- `compare.py`: scores both treatments' answers before build.
- `compare_build.py`: compares verdicts after build and test (`order_pr.py --override-candidates`).
- `build_inputs.py`, `check_c_equivalence.py`: build the reduced inputs; check the Java port against
  PPatHF's C results.

`reducer.py`, `reducerConfig.py` and `fcu.py` (the port of PPatHF's reduction code) are not in this
repository yet: PPatHF's replication package has no licence file. They will be added once that is cleared.
