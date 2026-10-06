#!/usr/bin/env python3
"""Run the witnessed role graph with Qwen3-VL's real frame+ASR modalities.

Qwen3-VL accepts text, images, and video frames; it has no audio encoder.  An
MP4 audio track is therefore not evidence available to this model.  This
wrapper preserves the frozen V3--V6 runners for reproducibility while adding a
new prompt version that states the actual modality boundary.
"""

from __future__ import annotations

try:
    from scripts import run_witnessed_reaction_candidate_av_vlm as base
except ModuleNotFoundError:
    import run_witnessed_reaction_candidate_av_vlm as base  # type: ignore[no-redef]


SYSTEM_V7_FRAMES_ASR_ROLE_BINDING = """You audit one short video-frame sequence
plus separately supplied ASR for a possible bystander reaction. This Qwen3-VL
endpoint does not consume the MP4 audio track. Never claim to hear a voice,
tone, sound, or speaker. ASR can establish approximate words and timing but
cannot by itself bind a speaker to a visible person.

First reason privately through this role graph:
- TRIGGER ACTOR: who performs the earlier or overlapping action?
- DIRECT TARGET: who receives that action or is directly addressed by it?
- RESPONSE ACTOR: who visibly produces the candidate response, if resolvable?
- PRIOR PARTICIPATION: was the response actor already a host, interviewer,
  performer, provocateur, companion in the setup, trigger actor, or target?

Label separate_bystander only when the ordered frames positively establish
that the response actor is neither endpoint of the triggering action and was
not participating in the setup. If lip movement, shot continuity, trigger
actor, target, response actor, or prior participation cannot be resolved from
frames, use offscreen_or_unresolved. Another visible person is not enough.

Normative ASR such as "stop," "don't," "that's wrong," or "what are you
doing" does not prove objection. The response must target a concrete earlier
or overlapping action. Reject ordinary questions, answers, self-defense,
interview elicitation, playful banter, setup dialogue, narration, and the
violator's own normative demand. Preserve welfare/protective reactions after
accidents as reactions, but label the trigger accident_or_involuntary.

Keep staging independent from reaction presence. A staged scene can contain a
reaction but cannot certify organic witnessed footage.

Return exactly one JSON object with these keys:
reaction_grounded: yes|no|uncertain
action_before_or_overlaps_response: yes|no|uncertain
response_targets_action: yes|no|uncertain
responder_role: separate_bystander|organic_audience|authority_or_host|affected_target|violator_or_actor|offscreen_or_unresolved|none
response_content: targeted_objection|correction_or_sanction|protective_intervention|interposition_or_separation|generic_affect|self_defense_or_excuse|description_only|none|uncertain
trigger_kind: interpersonal_treatment|shared_public_conduct|private_physical_safety|accident_or_involuntary|codified_or_legal_only|not_established|uncertain
staging: clearly_staged|no_clear_staging_evidence|uncertain
evidence: one short sentence naming only frame-visible evidence and supplied
ASR, or exactly which role or causal binding cannot be resolved
"""


def main() -> int:
    base.SYSTEM_PROMPTS["v7_frames_asr_role_binding"] = (
        SYSTEM_V7_FRAMES_ASR_ROLE_BINDING
    )
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
