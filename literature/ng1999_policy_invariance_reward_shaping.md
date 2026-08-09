# Policy Invariance Under Reward Transformations

- Authors: Andrew Y. Ng, Daishi Harada, Stuart Russell
- Year: 1999
- Venue: International Conference on Machine Learning (ICML)
- OpenAlex: `W1777239053`
- DOI: none indexed
- Metadata provenance: OpenAlex exact-title search on 2026-08-09; Crossref bibliographic search did
  not return the original ICML paper.

## Result Used Here

Adding a transition reward of the form
`F(s, s') = gamma * Phi(s') - Phi(s)` preserves optimal policies under the standard discounted-MDP
conditions. The shaping terms telescope in the discounted trajectory return, leaving only a
state-dependent offset.

## Relevance to FPO++

RoboSuite Square exposes bounded reach, grasp, lift, and hover stage rewards even though the released
FPO++ wrapper uses only sparse binary success. Using the maximum stage reward as `Phi` can expose
local progress to GAE without adopting RoboSuite's raw per-step dense reward, which could reward
indefinite hovering. H51 must verify the telescoping identity and nontrivial signed signal before any
policy training.

## Retrieval Provenance

- Endpoint: `GET https://api.openalex.org/works`
- Parameters: exact full-text/title phrase, `per_page=5`, selected metadata fields
- Matched record: `https://openalex.org/W1777239053`
- Crossref fallback query: bibliographic exact title, first three results; original paper absent

