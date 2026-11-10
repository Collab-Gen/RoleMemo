# Curated RoleMemo Data Schema

All data files are UTF-8 JSON. Field names and identifiers are preserved without renaming.

## `benchmark/qa.json`

A JSON array with 1,746 objects:

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | integer | Unique source sample identifier; values are not consecutive. |
| `npc_id` | string | NPC identifier used to join profiles and dialogue filenames. |
| `question` | string | Evaluation query. |
| `super_gt_answer` | string | Reference answer. |
| `evidence` | object | Supporting facts and persona-conditioned interpretation. |
| `evidence.core_fragments` | array of two strings | The two supporting factual fragments. |
| `evidence.core_insight` | string | Persona-conditioned insight associated with the query. |
| `final_question_type` | string | One of the four Chinese labels in the data README. |

Questions, reference answers, and core insights are nonempty in this snapshot. All sample IDs are unique; their observed range is 0–2311.

## NPC profiles

Each file in `benchmark/personas/NPC完整人设/` is an array of NPC profile objects, grouped by domain. There are 151 profiles in 23 files. `id` is an integer; the remaining fields contain character descriptions.

The exact keys are:

```text
id, role_name, title, category, background, current_status,
core_values, goals, dilemma, personality, hobbies, content,
喜爱的小动物, 喜爱的植物, 故乡, 旅游城市, 日常爱好, 爱吃的食物, 表达风格
```

The full character prompt is stored in `content`.

## User profiles

Each file in `benchmark/personas/User人设/` is an array of objects with integer `id` and string `content`. There are 151 user profiles in 23 files. The `id` links each user profile to the corresponding NPC.

## Dialogue histories

Each file in `benchmark/dialogues/` is named `{npc_id}.json` and contains an array of original utterance strings. There are 151 files, with 140–428 utterances per file and 38,341 utterances in total.

The common role prefixes are `User:` and `Assistant:`. A small number of entries have nonstandard prefixes or empty strings; the loading example preserves such entries as raw text instead of assigning an unsupported role or dropping data. Utterance counts do not define token lengths.

## Joining data

Normalize profile identifiers with `str(profile["id"])`, join to QA `npc_id`, and load `benchmark/dialogues/{npc_id}.json`. All four identifier sets match exactly in this curated snapshot.
