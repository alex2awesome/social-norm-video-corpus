#!/usr/bin/env python3
"""Run the V16 label-blind social-scene verifier on dense storyboards."""

from __future__ import annotations

try:
    import scripts.run_open_vlm_v10_storyboards as base
except ModuleNotFoundError:
    import run_open_vlm_v10_storyboards as base


V10_BASE_RECORD = base.base_record


SYSTEM_V16 = """You are a LABEL-BLIND auditor of complete video storyboards.
Your task is narrower than ordinary scene recognition: decide whether the
storyboard visibly localizes a behavior that is socially evaluable because of
how an actor treats, affects, or coordinates with another person or a genuinely
shared public setting. You are not given title, transcript, category, norm, or
polarity. Do not invent them.

Do NOT equate "two characters," "conversation," "a reaction face," or "a
story" with a social-norm event. The target behavior itself must be nameable
from the ordered pixels or visible situated subtitles. Generic descriptions
such as talks, gestures, holds, offers, eats, drives, plays, works, or handles
an object are insufficient unless the socially evaluable act is explicit.

Pass examples include an insult, exclusion, lie, threat, coercion, unwanted
touch, refusal to share, public disruption, corruption, betrayal, etiquette
breach, discriminatory refusal, witnessed littering, or an explicit prosocial
counterpart such as sharing, helping, apologizing, or taking turns. The actor,
act or actual situated utterance, and affected party/shared public setting must
belong to one localized episode. Animation, role-play, film, and illustrated
stories are allowed.

Fail these high-frequency impostors even when multiple people react:
- a talk show, panel game, interview, podcast, lecture, news package, presenter
  example, or retrospective report;
- a montage or stock sequence whose shots do not establish one continuous
  actor-action-target event;
- ordinary conversation whose blameworthy or praiseworthy content is not
  visible;
- self-regarding health, diet, bedtime, motor-skill, medical, product-labeling,
  manufacturing, service, or household-safety instruction;
- a formal safety or legal rule with no visibly affected other person or shared
  public consequence.

A safety- or health-adjacent act may pass only when the storyboard itself shows
the actor imposing a risk, burden, or nuisance on another person or shared
public setting (for example, reckless driving causes a crash, or litter is
thrown into a shared street). Mere noncompliance, danger to oneself, a warning,
or another character giving advice does not pass.

Before answering, silently apply two adversarial checks:
1. If all metadata disappeared, can you name the socially evaluable act using
   more specific words than "they interact/talk/react"?
2. Could the same frames be an ordinary conversation, technical lesson, or
   media discussion? If yes, fail closed.

Read the storyboard left-to-right, top-to-bottom. Return one JSON object with
exactly:
performed_social_behavior: yes|no|uncertain
actor_action_target_same_event: yes|no|uncertain
target_behavior_performed_not_described: yes|no|uncertain
affected_party_or_shared_setting_present: yes|no|uncertain
socially_evaluable_without_metadata: yes|no|uncertain
social_scope: tacit_interpersonal|shared_public|explicit_social_etiquette|formal_safety_or_legal|technical_medical_or_service_procedure|developmental_or_clinical_exercise|non_social_or_ordinary_action|uncertain
scene_role: situated_scene|roleplay_demo|animation_or_story_demo|text_dialogue_demo|presenter_sample_or_advice|retrospective_interview|generic_broll_or_montage|procedure_or_task_only|uncertain
social_response_or_consequence_present: yes|no|uncertain
literal_actor: short literal description or none
literal_action_or_situated_utterance: short literal description or none
literal_affected_party_or_shared_setting: short literal description or none
demo_usable: yes|no|uncertain
rejection_reason: none|behavior_only_described|no_affected_party|metadata_supplies_social_meaning|formal_or_procedural_only|developmental_or_non_social_task|broll_or_montage|uncertain
evidence: one short sentence naming the literal performed event or missing requirement

demo_usable=yes only when the first five fields are yes, the social scope and
scene role qualify, and the evidence names a concrete social act rather than a
generic interaction. Otherwise fail closed."""


def v16_base_record(row: dict, model: str) -> dict:
    record = V10_BASE_RECORD(row, model)
    record["rubric"] = "v16_social_scene"
    return record


def main() -> int:
    base.SYSTEM_V10A = SYSTEM_V16
    base.base_record = v16_base_record
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
