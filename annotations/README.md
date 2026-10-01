# WildlifeVid annotation licensing

The tables in this directory combine WildIcon annotation contributions with
identifiers, labels, and other information referring to the original datasets.

## CC BY 4.0 scope

Original annotations, prompts, and metadata created by the WildIcon authors are
released under [CC BY 4.0](LICENSE), to the extent that the authors hold the
copyright or similar rights needed to license them. This includes our original
labeling, prompt-writing, selection, arrangement, and compilation contributions.
When sharing those contributions, retain the WildIcon attribution and license
reference and indicate your changes, as required by CC BY 4.0. The paper citation
is available in the [main README](../README.md#citation).

The CC BY 4.0 grant applies to our original contributions. Source-derived
metadata, source videos, reference images, and other third-party materials remain
subject to the licenses and terms of their respective sources. Licensing follows
the authorship and applicable rights of each contribution, as described below.

## Files and field provenance

[`WildlifeVid.csv`](WildlifeVid.csv) is the merged table. The files in `source/`
are per-source WildlifeVid tables used as inputs to the
[merge script](../dataset/scripts/merge_wildlifevid_metadata.py). They combine
WildIcon annotation contributions with source-derived information; the table
below describes the provenance and licensing scope of each field.

| Fields | Provenance and licensing scope |
| --- | --- |
| `video`, `reference_image`, `track_id`, `species` | Paths, clip identifiers, and species labels refer to source material. WildIcon adds local path prefixes and curation; the original source terms continue to apply to the underlying material. |
| `dataset`, `species_id`, `identity`, `identity_str`, `global_identity` | The merge script adds dataset labels and assigns identifiers in the WildlifeVid namespace using the per-source tables. CC BY 4.0 covers our original mappings and labeling contributions. Underlying source labels and grouping information retain their respective source terms. |
| `prompt` | Prompt text is retained from the per-source WildlifeVid table. Prompts authored by the WildIcon authors are covered by CC BY 4.0; any text copied from an original source retains that source's terms. |
| `source_species_id`, `source_identity`, `source_identity_str`, `source_global_identity` | These preserve values from the per-source WildlifeVid tables before merging. The `source_` prefix identifies the merge input. Our earlier annotation contributions and inherited source content retain their respective licensing scopes. |
| `segment_prompt`, `segment_status`, `segment_score`, `segmented_image` | Records of the local foreground-preparation workflow. Our original record contributions are covered by CC BY 4.0. Populate foreground image paths when preparing local training metadata; see [Training](../README.md#training). |

Each CSV can contain both WildIcon contributions and source-derived information.
The CC BY 4.0 text in `LICENSE` must be read together with this scope statement.

## Original sources

Obtain source media from the original providers and follow their published
licenses and terms:

| WildlifeVid source table | Original source |
| --- | --- |
| [`source/AiM_metadata_identity_merged.csv`](source/AiM_metadata_identity_merged.csv) | [Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion) |
| [`source/AnimalKingdom_metadata.csv`](source/AnimalKingdom_metadata.csv) | [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom) |
| [`source/LoTE-Animal_metadata.csv`](source/LoTE-Animal_metadata.csv) | [LoTE-Animal](https://lote-animal.github.io/) |
| [`source/MammalNet_metadata.csv`](source/MammalNet_metadata.csv) | [MammalNet](https://mammal-net.github.io/) |

Code used to prepare, merge, or evaluate the annotations is covered by the code
licenses described in the [main README](../README.md#license) and
[`NOTICE`](../NOTICE).
