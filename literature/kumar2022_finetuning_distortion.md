# Kumar et al. (2022): Fine-Tuning Can Distort Pretrained Features

- Title: Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
- Authors: Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang
- arXiv: 2202.10054
- OpenAlex work: W4221149036
- Year: 2022
- Accessed: 2026-08-09

The paper shows that end-to-end fine-tuning can distort useful pretrained features and underperform
linear probing under distribution shift. Its LP-FT procedure motivates separating adaptation of a
small prediction subspace from changes to the full feature extractor. H52 does not claim that a flow
policy is a classifier; it tests the narrower transferable prediction that scarce downstream data
may be better used in the output layer than across all pretrained layers.

Provenance: metadata and abstract were checked through OpenAlex. Semantic Scholar returned HTTP 429
during the same lookup, so it was not used as an independent metadata source.
