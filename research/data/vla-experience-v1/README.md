# VLA experience decision data v1

This directory contains source-grounded research workflow decisions extracted from public VLA repositories. It is intended for the Kev-style typed decision model: each record has a project state and a variable set of choice, score and noul questions.

Files:

- `train.jsonl`: 1,600 records / 12,375 questions
- `dev.jsonl`: 240 records / 1,856 questions
- `test.jsonl`: 480 records / 3,712 questions
- `manifest.json`: generation version, question coverage and source repositories

The records cover data contracts, RLDS/LeRobot formats, split leakage, action normalization, resource-aware fine-tuning, benchmark episode protocols, failure attribution, evidence collection and deployment safety. They are not image-action trajectories and cannot replace policy training.
