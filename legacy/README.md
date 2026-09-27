# Legacy experiments

This directory contains approaches tested during development that are retained
for reference but are not part of the active HEARSAY pipeline.

## AASIST

`aasist/TestAASIST.py` contains an early experiment using pretrained AASIST
weights. The final system does not currently use AASIST because the pretrained
model targets older ASVspoof attacks and the team moved toward WavLM, residual
fingerprints, CPPS, and other challenge-specific features.
