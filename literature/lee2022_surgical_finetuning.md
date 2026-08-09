# Lee et al. (2022): Surgical Fine-Tuning

- Title: Surgical Fine-Tuning Improves Adaptation to Distribution Shifts
- Authors: Yoonho Lee et al.
- arXiv: 2210.11466
- OpenAlex work: W4307079438
- Year: 2022
- Accessed: 2026-08-09

The paper finds that fine-tuning selected network layers can match or exceed full fine-tuning under
several distribution shifts, especially when target data are limited. It motivates treating layer
selection as an adaptation mechanism rather than only shrinking the global learning rate. H52 fixes
one selection before results: the final velocity-output layer, the smallest native actor subspace
that directly controls the complete Square action chunk.

Provenance: metadata and abstract were checked through OpenAlex. Semantic Scholar returned HTTP 429
during the same lookup, so it was not used as an independent metadata source.
