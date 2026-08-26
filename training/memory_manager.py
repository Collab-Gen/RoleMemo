"""
MemoryManager — dual-stream memory bank for f_theta (DualMem).

Maintains a persistent memory bank M = ∪ M_i composed of:
  - Fact entries    F_i : short factual statements extracted from dialogue
  - Insight entries I_i : persona-conditioned interpretations that reference
                          one or more grounding Fact IDs

Used during RL training (and inference) to accumulate memory across chunks
of the long dialogue history (Section 3.2 of the paper).

Key operations:
  parse_and_merge   — parse model output, dedup facts, fix insight references
  get_prompt_memory_str   — return full memory bank as a string
  get_windowed_memory_str — return a recency-windowed view for the next prompt
"""

import re
from difflib import SequenceMatcher


class MemoryManager:
    def __init__(self):
        self.facts = []     # [{'id': int, 'text': str, 'raw': str}, ...]
        self.insights = []  # [{'id': int, 'ref': str, 'text': str, 'raw': str}, ...]

    # ── Internal helpers ───────────────────────────────────────────

    def _clean(self, text: str) -> str:
        # Strip leading attribution phrases and trailing parenthetical tags
        text = re.sub(r'^(the user mentioned|the user said|i)[^:：,，]*[：:,，]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\(.*?\)\s*$', '', text)
        text = re.sub(r'[,\.!?]', '', text)
        return text

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, self._clean(a), self._clean(b)).ratio()

    # ── Core method ────────────────────────────────────────────────

    def parse_and_merge(self, model_output: str):
        """
        Parse one chunk's model output and merge into the global memory bank.

        Steps:
          1. Extract Facts and Insights via regex.
          2. Dedup Facts against existing bank; build an ID redirect map.
          3. Fix Insight reference IDs using the redirect map.
          4. Resolve ID conflicts before storing.
        """
        fact_matches    = re.findall(r'(Fact\s+(\d+)\s*:\s+(.*))',              model_output)
        insight_matches = re.findall(r'(Insight\s+(\d+)\s*(?:\((.*)\))?\s*:\s+(.*))', model_output)

        # ── Facts ──────────────────────────────────────────────────
        id_redirect_map = {}
        max_fact_id = max((f['id'] for f in self.facts), default=0)
        temp_new_facts = []

        for _, fid_str, content in fact_matches:
            raw_fid = int(fid_str)
            text = content.strip()

            # Check for duplicates in existing bank and current batch
            dup_id = None
            for src in (self.facts, temp_new_facts):
                for entry in src:
                    if self._similarity(text, entry['text']) > 0.85:
                        dup_id = entry['id']
                        break
                if dup_id is not None:
                    break

            if dup_id is not None:
                id_redirect_map[raw_fid] = dup_id
            else:
                # Assign a non-conflicting ID
                final_id = raw_fid
                all_used = {f['id'] for f in self.facts} | {f['id'] for f in temp_new_facts}
                if final_id in all_used:
                    final_id = max_fact_id + 1
                max_fact_id = max(max_fact_id, final_id)
                id_redirect_map[raw_fid] = final_id
                temp_new_facts.append({
                    'id':  final_id,
                    'text': text,
                    'raw': f"Fact {final_id}: {text}",
                })

        self.facts.extend(temp_new_facts)

        # ── Insights ───────────────────────────────────────────────
        max_ins_id = max((i['id'] for i in self.insights), default=0)

        for _, iid_str, ref_content, content in insight_matches:
            raw_iid = int(iid_str)
            text = content.strip()

            # Remap reference IDs through the fact redirect map
            raw_ref_ids = [int(r) for r in re.findall(r'\d+', ref_content)] if ref_content else []
            fixed_refs  = sorted(set(id_redirect_map.get(r, r) for r in raw_ref_ids))
            ref_str     = " + ".join(f"Fact {i}" for i in fixed_refs)
            ref_display = f"({ref_str})" if ref_str else ""

            # Resolve Insight ID conflicts
            final_iid = raw_iid
            if any(i['id'] == final_iid for i in self.insights):
                final_iid = max_ins_id + 1
            max_ins_id = max(max_ins_id, final_iid)

            self.insights.append({
                'id':  final_iid,
                'ref': ref_display,
                'text': text,
                'raw': f"Insight {final_iid} {ref_display}: {text}",
            })

    # ── Memory serialization ───────────────────────────────────────

    def get_prompt_memory_str(self) -> str:
        """Return the full memory bank as a formatted string."""
        if not self.facts and not self.insights:
            return "No previous memory"
        lines = [f['raw'] for f in sorted(self.facts,   key=lambda x: x['id'])]
        lines += [i['raw'] for i in sorted(self.insights, key=lambda x: x['id'])]
        return "\n".join(lines)

    def get_windowed_memory_str(self, window_size: int = 4, include_insights: bool = True) -> str:
        """
        Return a recency-windowed view: the most recent `window_size` Facts,
        plus any Insights whose referenced Facts all fall within that window.
        """
        sorted_facts = sorted(self.facts, key=lambda x: x['id'])
        visible_facts = sorted_facts[-window_size:] if len(sorted_facts) > window_size else sorted_facts
        visible_ids   = {f['id'] for f in visible_facts}

        visible_insights = []
        if include_insights:
            for ins in self.insights:
                ref_ids = [int(x) for x in re.findall(r'\d+', ins['ref'])]
                if ref_ids and all(r in visible_ids for r in ref_ids):
                    visible_insights.append(ins)

        if not visible_facts and not visible_insights:
            return "No previous memory"

        lines  = [f['raw'] for f in visible_facts]
        lines += [i['raw'] for i in visible_insights]
        return "\n".join(lines)
