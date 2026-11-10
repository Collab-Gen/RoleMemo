# RoleMemo: Curated Public Data

This curated evaluation snapshot contains **1,746 reasoning queries**, **151 NPC personas with 151 paired user personas**, and **151 dialogue histories** across 23 domains.

## Download and extract

1. Download [RoleMemo-curated-data-2026-11-10.zip](RoleMemo-curated-data-2026-11-10.zip).
2. Extract the archive. Its top-level directory is `RoleMemo/`.
3. From that extracted directory, run:

```bash
python data/load_example.py
```

The loader, JSON files, full data README, per-file manifest, and checksums are inside the archive. In particular, `RoleMemo/data/SCHEMA.md` describes the exact fields and `RoleMemo/data/load_example.py` provides the dependency-free loader. These paths refer to files after extraction. A copy of the schema is also available directly as [SCHEMA.md](SCHEMA.md).

| Query category | Stored label | Queries |
|---|---|---:|
| Decision Guidance | 指导决策类 | 1,399 |
| Value Judgment | 价值判断类 | 136 |
| Interpretive Attribution | 解释归因类 | 110 |
| Contradiction Revelation | 揭示矛盾类 | 101 |
| **Total** | | **1,746** |

The archive contains 198 original JSON files totaling 12,224,253 bytes. QA, NPC profiles, paired user profiles, and dialogue filenames share the same 151 identifiers. Source sample identifiers and all dialogue strings are preserved.

## License and citation

The dataset files stored under `RoleMemo/data/benchmark/` after extraction are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See [DATA_LICENSE](../DATA_LICENSE) for the exact scope and full legal text. This does not change the licenses of repository code, model checkpoints, or other materials.

Please cite the original RoleMemo paper using [CITATION.cff](../CITATION.cff).

## Verify the download

The ZIP checksum is recorded in the repository-level [SHA256SUMS](../SHA256SUMS). After extraction, the archive's `RoleMemo/SHA256SUMS` and `RoleMemo/data/manifest.json` verify its individual files.
