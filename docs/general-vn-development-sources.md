# Additional real development sources

The [source manifest](../data/manifests/general_vn_development_sources_20261011.json) pins 60 native VNDB screenshots across six games, assigned to development before download or pixel inspection. Japanese releases and native 480×272 PSP / 960×544 Vita dimensions were verified. Known training-label VNs, current training-background identities and protected benchmark identities were excluded before access. No protected images, Google outputs or model predictions were used.

Two separate assistant visual readings accepted 25 ordinary horizontal-dialogue frames from five games. The other 35 frames contain menus, minigames, CG-only screens or vertical dialogue unsupported by the current horizontal recognizer. Names are tagged separately. Four initial transcription errors were corrected through independent review and native pixel inspection, including 姉/姐 and 容/客. These are assistant annotations, not independently human-annotated tests. Exact font-family identities remain unknown.

| Platform / VN | Accepted frames | Body characters | Body hiragana |
|---|---:|---:|---:|
| PSP v10091, Elkrone no Atelier | 3 | 63 | 43 |
| PSP v11454, Koibana Days | 5 | 154 | 94 |
| Vita v15773, Hyakka Yakou | 8 | 311 | 193 |
| Vita v16128, KLAP!! | 5 | 206 | 118 |
| Vita v18163, Nil Admirari no Tenbin | 4 | 126 | 64 |

There are 78 labelled lines, including 53 body lines with 860 normalized characters and 512 hiragana. Every source byte hash, decoded RGB pixel hash, native box bound and dialogue-region bound was checked; all line boxes were also visually inspected. Screenshots, transcripts, intermediate review files and crops stay private and ignored. Only URLs, hashes, classifications and small derived counts are committed.

The [derived review record](../eval/general_vn_development_review_20261011.json) binds private inputs and measurement commands. At pads 0/2/4, both polarities and raw/24-pixel heights, the 906 distinct development crop hashes have no overlap with 42,515 hashes from the exact current bitmap recipe's 21,522 training samples in 44 shards. This checks available current inputs; the original historical training shards are unavailable. It does not establish that the released baseline never encountered these games. Protected decoded pixels were not reread and are not covered by this pixel-equality statement.

This cohort is **not ready for model selection** under the unchanged [prospective protocol](general-vn-development.md). It needs a third PSP game and more body dialogue per game. These sources supplement the earlier cohort; they do not lower quotas, replace frozen old references, reopen old checkpoint decisions or supply training data. No CER or accuracy improvement was measured here.
