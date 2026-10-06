#!/usr/bin/env python3
"""Run a local OpenAI-compatible video VLM over a frozen scene benchmark.

Results are append-only JSONL.  Existing records are skipped, so interrupted
runs resume safely.  The script never edits the source manifest or corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SYSTEM_V1 = """You are auditing whether a video directly SHOWS a usable social event.
Be conservative and use only pixels across time. A transcript, title, narration,
caption, or supplied label is not proof that the action is visible.

Return one JSON object with exactly these keys:
physical_social_interaction_visible: yes|no|uncertain
actor_visible: yes|no|uncertain
target_or_shared_context_visible: yes|no|uncertain
action_directly_observable: yes|no|uncertain
described_or_implied_only: yes|no|uncertain
depiction_type: enacted_demo|organic_recording|screen_demo|talking_head|broll|graphics|mixed|uncertain
labeled_event_directly_depicted: yes|no|uncertain
label_fit_confidence: number from 0 to 1
evidence: one short sentence describing only visible evidence

Talking heads recounting an event, news B-roll, unrelated footage, and text cards
are not direct depictions. A staged role-play is a direct depiction. A screen
demo is a direct depiction only when the labeled social action itself occurs in
the interface. Do not infer off-screen behavior."""

SYSTEM_V2 = """You are auditing whether a video directly SHOWS a concrete,
norm-relevant behavior event. A valid event may involve:
- one visible actor in a public/shared context (littering, theft, unsafe driving);
- two or more people interacting;
- an observable speech act;
- an enacted, animated, or screen-mediated demonstration.

Do not require multiple people. Do not treat motion, people, a setting, narration,
captions, or the proposed label as proof. Talking heads recounting an event, news
B-roll, context-only footage, and an aftermath without the act are not direct
depictions. Treat the supplied label/explanation as a fallible hypothesis. A
brief visible act counts even if it occupies only part of the clip.

Return one JSON object with exactly:
concrete_behavior_event_visible: yes|no|uncertain
proposed_label_event_visible: yes|no|uncertain
event_completeness: complete|partial|context_only|described_only|uncertain
actor_visible: yes|no|uncertain
target_or_shared_context_visible: yes|no|uncertain
depiction_type: enacted_demo|organic_recording|animation|screen_demo|talking_head|broll|graphics|mixed|uncertain
brief_action_may_be_missed: yes|no
label_fit_confidence: number from 0 to 1
evidence: one short sentence describing the visible action, or saying none is visible

Judge what happens in pixels across time. Do not infer off-screen behavior."""

SYSTEM_V3 = """You are auditing whether a video contains a situated social
scenario that can serve as weak supervision. Do not confuse a dialogue close-up
with a talking head: a talking head addresses the audience/camera outside a
scene; characters speaking to each other inside a shared situation are an
enacted or recorded social scene. Situated speech acts (lying, insulting,
refusing, confronting, apologizing) count as behavior. Brief physical acts count.

For instructional videos, any clear demonstration format is allowed: staged
role-play, film/TV excerpt, hidden-camera example, animation, or screen-mediated
scenario. It need not be organic. Judge whether the supplied label/explanation
is plausibly being demonstrated by the scene, while treating that text as a
fallible hypothesis rather than visual proof.

For witnessed videos, separately judge whether the violation action itself is
visible before/independently of the reaction and whether it appears organic.
A staged scene can still be a useful instructional demo but is not organic
witnessed evidence. A reaction, accusation, aftermath, or setting alone is not
the violation action.

For commentary videos, a commentator alone is not an event scene. A cutaway,
embedded source clip, reenactment, animation, or screen capture showing the
discussed event can be.

Return one JSON object with exactly:
situated_social_scenario_visible: yes|no|uncertain
characters_interact_or_act_in_shared_context: yes|no|uncertain
concrete_action_or_situated_speech_visible: yes|no|uncertain
proposed_norm_plausibly_demonstrated: yes|no|uncertain
witnessed_violation_action_visible: yes|no|uncertain|na
reaction_only_or_aftermath: yes|no|uncertain
visual_content_class: enacted_scene|organic_scene|animation|screen_scenario|talking_head|broll|graphics|mixed|uncertain
organic_vs_staged: organic|staged|mixed|uncertain|na
event_completeness: complete|partial|context_only|described_only|uncertain
brief_action_may_be_missed: yes|no
label_fit_confidence: number from 0 to 1
evidence: one short sentence describing the visible situated act or exchange

Use the full visual sequence, including visible subtitles. Do not infer an
off-screen event solely from narration or the proposed label."""

SYSTEM_V4 = """You audit whether a video directly shows a usable social-norm
event. Any depiction format can qualify: organic footage, role-play, film/TV,
animation, puppets, toys, or a screen-mediated scenario. Situated dialogue is
behavior even when nobody says "example" or "demonstration." Alternating
close-ups can be one shared conversation; use the full sequence and subtitles.

Operational boundary:
- A social norm concerns human/anthropomorphic interpersonal or public conduct
  that others can socially evaluate: courtesy, discrimination, honesty,
  boundaries, harassment, cooperation, public etiquette, responsibility.
- Physical self-defense technique, exercise/motor commands, sports technique,
  animal handling, medical/technical/academic procedures, and mere emotion
  expression are not social norms unless a separate socially evaluated behavior
  is directly shown.
- Talking-head advice, news context, generic B-roll, aftermath, and narration
  about an off-screen event are not a situated event.
- Direct-to-camera speech can itself be the event when the speech is the
  behavior (for example a threat or racist rant), but not when it merely
  explains another event.

Treat the proposed norm as fallible. "usable_demo_after_relabel" asks whether
there is a concrete social-norm scene worth keeping under some accurate label.
"proposed_norm_supported" separately asks whether the supplied label fits.
A long source with an event somewhere but no clean localization is broad, not a
clean clip.

Pillar rules:
- instructional: any clearly demonstrated social situation is valid; it need
  not be organic or explicitly pedagogical.
- witnessed: the violating action itself must be visible before/independently
  of a targeted normative reaction and appear organic rather than staged.
- commentary: a commentator alone is invalid; event footage, reenactment,
  animation, or screen capture of the discussed behavior can qualify.

Return one JSON object with exactly:
social_norm_domain: yes|no|uncertain
situated_social_scenario_visible: yes|no|uncertain
observable_social_behavior_or_speech: yes|no|uncertain
usable_demo_after_relabel: yes|no|uncertain
proposed_norm_supported: yes|no|uncertain
procedural_or_nonsocial_activity_only: yes|no|uncertain
presentation_or_context_only: yes|no|uncertain
instructional_demo_present: yes|no|uncertain|na
witnessed_action_then_reaction_organic: yes|no|uncertain|na
commentary_event_footage_present: yes|no|uncertain|na
localization_quality: clean|broad|event_not_found|uncertain
depiction_type: organic_scene|enacted_scene|film_tv|animation|puppet_or_toy|screen_scenario|talking_head|broll|graphics|mixed|uncertain
evidence: one short sentence naming the visible behavior, or saying why none qualifies

Judge pixels across time. Do not infer off-screen behavior from the label."""

SYSTEM_V5 = """You are a second-stage verifier for instructional social-norm
demonstrations. The first-stage models over-accept presenters, advice, formal
procedures, and visible people whose claimed interaction never occurs. Require
the behavior event itself.

A demonstration passes only when:
1. an actor/character visibly performs a concrete action or situated speech act;
2. an affected person or genuinely shared social setting is visible or is an
   explicit participant in the depicted scene; and
3. the act is informal interpersonal/shared-public conduct that observers can
   socially approve or disapprove.

Any medium may pass: live action, role-play, film, animation, puppets, toys,
screen scenarios, and static illustrated stories. A static illustration passes
only if it depicts the actor, conduct, and target/shared setting—not when it
merely shows a portrait, question, label, or explanatory text.

Do not pass:
- a presenter, interviewee, or character merely telling the audience about a
  behavior that nobody performs in the depicted situation;
- service phrases taught to camera without a customer/service exchange;
- an isolated choice when the claimed norm is another person's respect for it;
- generic conversation whose claimed conduct is supplied only by metadata;
- security/key-control, police/legal compliance, technical/medical procedures,
  motor skills, or diagnostic/personality traits;
- context, aftermath, B-roll, or an interval too broad to localize.

Direct-to-camera speech can itself be the event only when the actual utterance
is socially evaluable conduct (for example, a threat or slur), not when it is
advice or retrospective explanation. A public act such as littering does not
need a second person, but the affected shared setting and act must be visible.
Treat the proposed norm and explanation as fallible.

Return one JSON object with exactly:
social_norm_domain: yes|no|uncertain
behavior_occurs_in_scene: yes|no|uncertain
affected_party_or_shared_setting_visible: yes|no|uncertain
situated_interaction_complete: yes|no|uncertain
audience_directed_explanation_only: yes|no|uncertain
formal_procedure_trait_or_skill_only: yes|no|uncertain
illustrated_action_not_just_text: yes|no|uncertain|na
usable_demo_after_relabel: yes|no|uncertain
proposed_norm_supported: yes|no|uncertain
localization_quality: clean|broad|event_not_found|uncertain
depiction_type: organic_scene|enacted_scene|film_tv|animation|puppet_or_toy|illustrated_story|screen_scenario|talking_head|broll|graphics|mixed|uncertain
rejection_reason: none|behavior_not_shown|presentation_only|not_social_norm|unlocalized|uncertain
evidence: one short sentence naming the behavior that occurs, or the missing requirement

Judge the full visual sequence. Subtitles may clarify situated dialogue but may
not turn a presenter or missing/off-screen event into a demonstration."""

SYSTEM_V6 = """You are a BLIND observable-event recorder. Describe what the
visual sequence itself shows before anyone tells you its topic or proposed
label. Do not infer a hidden action from a setting, character identity, emotion,
title, narration, or likely story. Do not convert an object or occupation into
an abstract value (for example, sorting objects is not "acceptance").

Find the single strongest concrete event in the saved clip. An event may be a
physical act, a visible public act, or situated speech whose meaning is shown by
visible subtitles. Any medium may qualify: live action, film, animation,
puppets, toys, screen scenarios, or illustrated stories.

Evidence requirements:
- name the actor, the literal observable act, and the affected character or
  shared setting;
- give start/end positions as percentages of this saved clip, not source-video
  timestamps;
- distinguish before, during, and after evidence. For a static illustration,
  repeat the depicted state and explain that no temporal change occurs;
- if the claimed act would be unknowable without metadata, mark
  metadata_needed_to_identify_action=yes;
- a presenter, interview, generic conversation, relevant location, B-roll,
  reaction, or aftermath is not an event unless the actual conduct is visible;
- do not claim a handshake, transfer, response, insult, refusal, helping act, or
  other interaction unless it appears inside this clip.

Return one JSON object with exactly:
visually_observable_event: yes|no|uncertain
event_start_percent: integer 0-100, or -1 when no event is found
event_end_percent: integer 0-100, or -1 when no event is found
actor_visible_description: short literal description or none
action_or_situated_speech_description: short label-free description or none
affected_party_or_shared_setting_description: short literal description or none
evidence_before: short visible description or none
evidence_during: short visible description or none
evidence_after: short visible description or none
metadata_needed_to_identify_action: yes|no|uncertain
presentation_or_context_only: yes|no|uncertain
depiction_type: organic_scene|enacted_scene|film_tv|animation|puppet_or_toy|illustrated_story|screen_scenario|talking_head|broll|graphics|mixed|uncertain
confidence: number from 0 to 1
evidence: one short sentence containing only observable evidence

Never name a norm or moral interpretation. Record the action literally."""

SYSTEM_V7 = """You audit candidate WITNESSED social-norm sequences by recording
atomic observations. Do not decide "keep" or collapse the fields into a general
impression. The video is visual evidence; the supplied timestamped transcript
may clarify situated speech and reaction semantics, but may not prove that an
off-screen action was visible.

First identify the alleged violating action itself. Then separately identify a
later reaction. A strict witnessed signal requires an organic bystander,
authority/host, or organic audience to object, correct, sanction, or
protectively intervene because of the action.

Do not count as normative reactions:
- the affected target's personal complaint or fear;
- the actor's excuse, self-defense, apology, or compliance;
- generic surprise, laughter, fear, profanity, anger, or disgust;
- narration/commentary, accidents, legal/safety warnings, or personal
  preferences with no tacit interpersonal/shared-public expectation.

Scripted scenes may be useful instructional demonstrations, but they are not
organic witnessed evidence. Treat the proposed label and detector explanation
as fallible hypotheses.

Return one JSON object with exactly:
action_visible: yes|no|uncertain
action_voluntary: yes|no|uncertain
expectation_kind: interpersonal_treatment|shared_public_conduct|personal_preference|safety_or_legal_only|accident_or_involuntary|technical_or_procedural|uncertain|none
reaction_visible_or_audibly_grounded: yes|no|uncertain
reaction_source_role: bystander|authority_or_host|organic_audience|affected_target|violator_or_actor|narrator_or_commentator|uncertain|none
reaction_content: targeted_objection|correction_or_sanction|protective_intervention|generic_affect|self_defense_or_excuse|compliance_or_apology|description_only|uncertain|none
action_established_before_reaction: yes|no|uncertain
reaction_targets_action: yes|no|uncertain
authenticity: organic|hidden_camera_genuine|scripted|animation|news_or_commentary|uncertain
proposed_label_relation: exact|repairable|mismatch|uncertain
pre_reaction_demo_quality: clear_visual|clear_audiovisual|incomplete_or_ambiguous|contains_reaction_signal|none|uncertain
action_end_percent: integer 0-100, or -1 when no reliable bound
reaction_start_percent: integer 0-100, or -1 when no reliable bound
action_evidence: one short literal sentence or none
reaction_evidence: one short sentence identifying speaker/role and response or none
evidence: one short sentence explaining the atomic distinction

Fail closed on role, causality, and timing. Do not infer a bystander objection
from mere escalation or from the affected target reacting."""

SYSTEM_V7B = SYSTEM_V7 + """

Boundary clarification: action_end_percent is the end of the alleged action
before the response; reaction_start_percent is the beginning of the later
response. Therefore action_end_percent must be less than or equal to
reaction_start_percent. If the two cannot be separated reliably, return -1 for
both bounds."""

SYSTEM_V7C = SYSTEM_V7B + """

Final fail-closed reminder: overlapping action and response do not provide a
clean pre-reaction boundary. If action_end_percent would be greater than
reaction_start_percent, or either boundary is uncertain, return -1 for both."""

SYSTEM_V8 = """You are a fail-closed verifier of instructional social-norm
demonstrations. The question is NOT whether the clip contains people, motion,
conversation, or topic-relevant imagery. The question is whether the saved
interval itself is a training example of a concrete tacit social behavior.

A usable demonstration requires all of these:
1. an on-screen actor or character performs the behavior, including a situated
   speech act whose words are audible or visible in subtitles;
2. the affected person or genuinely shared public setting belongs to that same
   event, rather than appearing elsewhere in a montage;
3. an observer can socially evaluate the depicted conduct from this clip; and
4. the event is localized, informal interpersonal/shared-public conduct rather
   than a formal legal, safety, medical, cleaning, finance, traffic, security,
   employment-dress, or technical procedure.

Any deliberate depiction medium may pass: role-play, film/TV, animation,
puppets/toys, screen scenarios, or static illustrated stories. Static imagery
passes only when it depicts actor + conduct + affected person/shared setting.

Fail these recurrent look-alikes:
- an interview or presenter discusses bullying, empathy, relationships, or
  another norm but nobody enacts it;
- generic couples, classrooms, crowds, workplaces, distressed faces, hugs,
  handshakes, or people reviewing papers are used as B-roll under narration;
- a montage contains related settings but no complete actor-action-target event;
- a police stop/search, traffic rule, escalator warning, cleaning routine,
  navigation lesson, or money-allocation procedure;
- the consequence, reaction, or police response is visible but the claimed
  social act happened off-screen;
- generic dialogue is called honesty, respect, responsibility, or inclusion
  only because metadata supplies that interpretation;
- a title, narration, explanation, or proposed label asserts an event that the
  audiovisual scene does not perform.

Positive boundary examples: a customer asks and a worker responds in the same
store scene; an animation shows one child litter and peers confront the child;
an illustrated dialogue shows a manager overload a worker and the worker
negotiates priorities. Negative boundary examples: a teacher interview over
classroom footage; relationship advice over unrelated couples; body-camera
traffic enforcement.

Treat the supplied norm and explanation as fallible hypotheses. Relabeling is
allowed only when another concrete socially evaluable event is unambiguously
depicted. Do not reward topical relevance.

Return one JSON object with exactly:
on_screen_social_event: yes|no|uncertain
actor_performs_target_behavior: yes|no|uncertain
affected_party_or_shared_context_same_event: yes|no|uncertain
behavior_socially_evaluable_from_clip: yes|no|uncertain
event_temporally_localized: yes|no|uncertain
informal_social_conduct_not_formal_procedure: yes|no|uncertain
evidence_source: physical_action|situated_dialogue_or_subtitles|static_depicted_action|narration_only|metadata_only|generic_motion|uncertain
visual_role: demonstrated_event|generic_context_broll|presenter_or_interview|montage_or_compilation|aftermath_or_reaction_only|formal_or_procedural|ambiguous_story_excerpt|uncertain
usable_demo_after_relabel: yes|no|uncertain
proposed_norm_supported: yes|no|uncertain
literal_actor: short literal description or none
literal_action: short literal description or none
literal_affected_party_or_shared_context: short literal description or none
rejection_reason: none|behavior_not_shown|context_only|presentation_only|montage_only|aftermath_only|formal_or_procedural|ambiguous_or_metadata_dependent|uncertain
evidence: one short sentence naming the literal actor-action-target event or the missing requirement

The fields must agree. usable_demo_after_relabel=yes is permitted only when the
first six fields are all yes, evidence_source is physical_action,
situated_dialogue_or_subtitles, or static_depicted_action, and visual_role is
demonstrated_event. Otherwise fail closed."""

SYSTEM_V9A = """You are the first, LABEL-BLIND pass of a two-pass audit. Record
only the strongest literal event that the saved clip itself shows. You are not
given a title, transcript, category, proposed norm, polarity, or explanation.
Never guess any of them.

An event needs a visible actor/character, a literal action or situated utterance,
and an affected person or shared setting in the same localized episode. Use
audible or visibly subtitled dialogue only when it occurs inside the depicted
situation. Any deliberate depiction medium can qualify, including role-play,
film/TV, animation, puppets, screen scenarios, and static illustrated stories.
An explicit etiquette demonstration is an event when the behavior is actually
performed; do not reject it merely because it is formal or pedagogical.

Do not turn these into events:
- presenters or interviewees explaining conduct to an audience;
- generic people, objects, settings, reaction faces, aftermath, or B-roll;
- a montage whose separate shots do not contain one actor-action-target event;
- text or narration asserting an action that the imagery does not depict;
- a still portrait or generic group image with no identifiable conduct;
- an object-handling, exercise, game, safety, cleaning, driving, or technical
  routine unless the literal visible event also contains interpersonal conduct.

Use clip-relative percentages. If no localized event exists, use -1 for both
bounds. Do not assign a norm and do not decide whether a proposed label fits.

Return one JSON object with exactly:
observable_event: yes|no|uncertain
event_start_percent: integer 0-100, or -1 when no localized event is found
event_end_percent: integer 0-100, or -1 when no localized event is found
literal_actor: short visible description or none
literal_action_or_situated_utterance: short literal description or none
literal_affected_party_or_shared_setting: short visible description or none
same_event_actor_action_target: yes|no|uncertain
event_temporally_localized: yes|no|uncertain
evidence_source: physical_action|situated_dialogue_or_subtitles|static_depicted_action|generic_motion|narration_or_text_only|none|uncertain
scene_role: demonstrated_event|presenter_or_interview|generic_context_broll|montage_or_compilation|aftermath_or_reaction_only|formal_or_skill_procedure|ambiguous_story_excerpt|none|uncertain
demonstration_kind: interpersonal_conduct|shared_public_conduct|explicit_social_etiquette|formal_or_safety_rule|motor_game_or_technical_skill|generic_or_ambiguous|none|uncertain
presentation_only: yes|no|uncertain
montage_only: yes|no|uncertain
aftermath_only: yes|no|uncertain
metadata_needed_to_name_action: yes|no|uncertain
evidence_before: short literal visible description or none
evidence_during: short literal visible description or none
evidence_after: short literal visible description or none
evidence: one short sentence containing only literal audiovisual evidence

The fields must agree. observable_event=yes requires ordered temporal bounds,
same_event_actor_action_target=yes, event_temporally_localized=yes, an event
evidence_source, scene_role=demonstrated_event, and all three literal roles.
Otherwise fail closed."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def successful_prediction_keys(records: list[dict]) -> set[tuple[str, str, str, str]]:
    """Return only completed predictions, never failed append-only attempts."""
    return {
        (
            record["item_id"],
            record["mode"],
            record["model"],
            record.get("rubric", "v1"),
        )
        for record in records
        if record.get("result") is not None and record.get("error") is None
    }


def parse_json(text: str, rubric: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    truncated_evidence_repaired = False
    if start >= 0 and end < start and rubric == "v7c":
        # Qwen occasionally completes every atomic decision field, then loops
        # inside the final free-text evidence string until generation truncates.
        # Recover only that non-gating tail; the untouched raw response remains
        # in the append-only record for audit.
        marker = re.search(r',\s*"evidence"\s*:\s*"', text)
        if marker:
            text = (
                text[: marker.start()]
                + ', "evidence": "truncated after complete atomic fields"}'
            )
            end = text.rfind("}")
            truncated_evidence_repaired = True
    if start < 0 or end < start:
        raise ValueError("response contained no JSON object")
    result = json.loads(text[start : end + 1])
    if (
        rubric == "v8"
        and "informal_social_conduct_not_formal_procedure" not in result
        and "informal_social_conduct_not_formal_procedural" in result
    ):
        result["informal_social_conduct_not_formal_procedure"] = result.pop(
            "informal_social_conduct_not_formal_procedural"
        )
        result["informal_social_conduct_key_validation_repair"] = (
            "procedural_to_procedure_exact_key_alias"
        )
    if rubric == "v9a" and "evidence" not in result and result.get(
        "evidence_during"
    ):
        # Some GLM responses complete every gating field but omit only the
        # redundant one-line evidence tail. Preserve the literal during-event
        # text and the untouched raw response; this repair does not change a
        # gating decision.
        result["evidence"] = result["evidence_during"]
        result["evidence_validation_repair"] = (
            "missing_tail_copied_from_evidence_during"
        )
    if truncated_evidence_repaired:
        result["evidence_validation_repair"] = (
            "truncated_free_text_tail_replaced"
        )
    if rubric == "v1":
        required = {
            "physical_social_interaction_visible",
            "actor_visible",
            "target_or_shared_context_visible",
            "action_directly_observable",
            "described_or_implied_only",
            "depiction_type",
            "labeled_event_directly_depicted",
            "label_fit_confidence",
            "evidence",
        }
    elif rubric == "v2":
        required = {
            "concrete_behavior_event_visible",
            "proposed_label_event_visible",
            "event_completeness",
            "actor_visible",
            "target_or_shared_context_visible",
            "depiction_type",
            "brief_action_may_be_missed",
            "label_fit_confidence",
            "evidence",
        }
    elif rubric == "v3":
        required = {
            "situated_social_scenario_visible",
            "characters_interact_or_act_in_shared_context",
            "concrete_action_or_situated_speech_visible",
            "proposed_norm_plausibly_demonstrated",
            "witnessed_violation_action_visible",
            "reaction_only_or_aftermath",
            "visual_content_class",
            "organic_vs_staged",
            "event_completeness",
            "brief_action_may_be_missed",
            "label_fit_confidence",
            "evidence",
        }
    elif rubric == "v4":
        required = {
            "social_norm_domain",
            "situated_social_scenario_visible",
            "observable_social_behavior_or_speech",
            "usable_demo_after_relabel",
            "proposed_norm_supported",
            "procedural_or_nonsocial_activity_only",
            "presentation_or_context_only",
            "instructional_demo_present",
            "witnessed_action_then_reaction_organic",
            "commentary_event_footage_present",
            "localization_quality",
            "depiction_type",
            "evidence",
        }
    elif rubric == "v5":
        required = {
            "social_norm_domain",
            "behavior_occurs_in_scene",
            "affected_party_or_shared_setting_visible",
            "situated_interaction_complete",
            "audience_directed_explanation_only",
            "formal_procedure_trait_or_skill_only",
            "illustrated_action_not_just_text",
            "usable_demo_after_relabel",
            "proposed_norm_supported",
            "localization_quality",
            "depiction_type",
            "rejection_reason",
            "evidence",
        }
    elif rubric in {"v7", "v7b", "v7c"}:
        required = {
            "action_visible",
            "action_voluntary",
            "expectation_kind",
            "reaction_visible_or_audibly_grounded",
            "reaction_source_role",
            "reaction_content",
            "action_established_before_reaction",
            "reaction_targets_action",
            "authenticity",
            "proposed_label_relation",
            "pre_reaction_demo_quality",
            "action_end_percent",
            "reaction_start_percent",
            "action_evidence",
            "reaction_evidence",
            "evidence",
        }
    elif rubric == "v6":
        required = {
            "visually_observable_event",
            "event_start_percent",
            "event_end_percent",
            "actor_visible_description",
            "action_or_situated_speech_description",
            "affected_party_or_shared_setting_description",
            "evidence_before",
            "evidence_during",
            "evidence_after",
            "metadata_needed_to_identify_action",
            "presentation_or_context_only",
            "depiction_type",
            "confidence",
            "evidence",
        }
    elif rubric == "v9a":
        required = {
            "observable_event",
            "event_start_percent",
            "event_end_percent",
            "literal_actor",
            "literal_action_or_situated_utterance",
            "literal_affected_party_or_shared_setting",
            "same_event_actor_action_target",
            "event_temporally_localized",
            "evidence_source",
            "scene_role",
            "demonstration_kind",
            "presentation_only",
            "montage_only",
            "aftermath_only",
            "metadata_needed_to_name_action",
            "evidence_before",
            "evidence_during",
            "evidence_after",
            "evidence",
        }
    else:
        required = {
            "on_screen_social_event",
            "actor_performs_target_behavior",
            "affected_party_or_shared_context_same_event",
            "behavior_socially_evaluable_from_clip",
            "event_temporally_localized",
            "informal_social_conduct_not_formal_procedure",
            "evidence_source",
            "visual_role",
            "usable_demo_after_relabel",
            "proposed_norm_supported",
            "literal_actor",
            "literal_action",
            "literal_affected_party_or_shared_context",
            "rejection_reason",
            "evidence",
        }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if rubric in {"v7", "v7b", "v7c"}:
        action_end = result["action_end_percent"]
        reaction_start = result["reaction_start_percent"]
        if not isinstance(action_end, (int, float)) or not isinstance(
            reaction_start, (int, float)
        ):
            raise ValueError("v7 event percentages must be numeric")
        if not (-1 <= action_end <= 100 and -1 <= reaction_start <= 100):
            if rubric == "v7c":
                result["action_end_percent"] = -1
                result["reaction_start_percent"] = -1
                result["temporal_bounds_validation_repair"] = (
                    "out_of_range_fail_closed"
                )
                action_end = -1
                reaction_start = -1
            else:
                raise ValueError(
                    "v7 event percentages must be between -1 and 100"
                )
        if (action_end == -1) != (reaction_start == -1):
            if rubric == "v7c":
                result["action_end_percent"] = -1
                result["reaction_start_percent"] = -1
                result["temporal_bounds_validation_repair"] = (
                    "unpaired_bound_fail_closed"
                )
                action_end = -1
                reaction_start = -1
            else:
                raise ValueError("v7 event percentages must both be -1 or both be set")
        if action_end != -1 and reaction_start != -1 and action_end > reaction_start:
            if rubric == "v7c":
                # Preserve the atomic model judgments but fail closed on an
                # incompatible proposed cut. This repair cannot create a
                # positive selection because downstream witnessed rules require
                # reliable, ordered bounds. The raw response remains alongside
                # this parsed result for manual audit.
                result["action_end_percent"] = -1
                result["reaction_start_percent"] = -1
                result["temporal_bounds_validation_repair"] = (
                    "reverse_or_overlap_fail_closed"
                )
            else:
                raise ValueError("v7 action end must not follow reaction start")
        allowed = {
            "action_visible": {"yes", "no", "uncertain"},
            "action_voluntary": {"yes", "no", "uncertain"},
            "expectation_kind": {
                "interpersonal_treatment",
                "shared_public_conduct",
                "personal_preference",
                "safety_or_legal_only",
                "accident_or_involuntary",
                "technical_or_procedural",
                "uncertain",
                "none",
            },
            "reaction_visible_or_audibly_grounded": {"yes", "no", "uncertain"},
            "reaction_source_role": {
                "bystander",
                "authority_or_host",
                "organic_audience",
                "affected_target",
                "violator_or_actor",
                "narrator_or_commentator",
                "uncertain",
                "none",
            },
            "reaction_content": {
                "targeted_objection",
                "correction_or_sanction",
                "protective_intervention",
                "generic_affect",
                "self_defense_or_excuse",
                "compliance_or_apology",
                "description_only",
                "uncertain",
                "none",
            },
            "action_established_before_reaction": {"yes", "no", "uncertain"},
            "reaction_targets_action": {"yes", "no", "uncertain"},
            "authenticity": {
                "organic",
                "hidden_camera_genuine",
                "scripted",
                "animation",
                "news_or_commentary",
                "uncertain",
            },
            "proposed_label_relation": {
                "exact",
                "repairable",
                "mismatch",
                "uncertain",
            },
            "pre_reaction_demo_quality": {
                "clear_visual",
                "clear_audiovisual",
                "incomplete_or_ambiguous",
                "contains_reaction_signal",
                "none",
                "uncertain",
            },
        }
        if rubric == "v7c" and result["authenticity"] == "none":
            result["authenticity"] = "uncertain"
            result["authenticity_validation_repair"] = (
                "none_to_uncertain_fail_closed"
            )
        for key, values in allowed.items():
            if result[key] not in values:
                if rubric == "v7c" and "uncertain" in values:
                    raw_value = result[key]
                    result[key] = "uncertain"
                    result[f"{key}_validation_repair"] = (
                        f"out_of_schema_{raw_value}_to_uncertain_fail_closed"
                    )
                else:
                    raise ValueError(f"invalid v7 {key}: {result[key]!r}")
    if rubric == "v6":
        start_percent = result["event_start_percent"]
        end_percent = result["event_end_percent"]
        if not isinstance(start_percent, (int, float)) or not isinstance(
            end_percent, (int, float)
        ):
            raise ValueError("v6 event percentages must be numeric")
        if not (-1 <= start_percent <= 100 and -1 <= end_percent <= 100):
            # Some otherwise schema-valid responses use source seconds despite
            # the percent instruction. Preserve the literal event judgment and
            # raw response, but fail closed on unusable localization bounds.
            result["event_start_percent"] = -1
            result["event_end_percent"] = -1
            result["temporal_bounds_validation_repair"] = (
                "out_of_range_to_unlocalized"
            )
            start_percent = -1
            end_percent = -1
        if (start_percent == -1) != (end_percent == -1):
            raise ValueError("v6 event percentages must both be -1 or both be set")
        if start_percent != -1 and start_percent > end_percent:
            raise ValueError("v6 event start must not follow event end")
        if (
            result["visually_observable_event"] == "yes"
            and start_percent == -1
            and "temporal_bounds_validation_repair" not in result
        ):
            raise ValueError("v6 visible events require a temporal span")
        confidence = result["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("v6 confidence must be between 0 and 1")
    if rubric == "v8":
        yes_no_uncertain = {"yes", "no", "uncertain"}
        visual_role_aliases = {
            "presentation_only": "presenter_or_interview",
            "static_depicted_action": "ambiguous_story_excerpt",
            "title_card": "ambiguous_story_excerpt",
            "ambiguous_or_metadata_dependent": "ambiguous_story_excerpt",
        }
        if result["visual_role"] in visual_role_aliases:
            raw_role = result["visual_role"]
            result["visual_role"] = visual_role_aliases[raw_role]
            result["visual_role_validation_repair"] = (
                f"{raw_role}_to_{result['visual_role']}_fail_closed"
            )
        rejection_reason_aliases = {
            "metadata_only": "ambiguous_or_metadata_dependent",
            "narration_only": "presentation_only",
            "generic_motion": "context_only",
            "generic_context_broll": "context_only",
            "presenter_or_interview": "presentation_only",
            "aftermath_or_reaction_only": "aftermath_only",
            "proposed_norm_supported": "uncertain",
            "static_depicted_action": "ambiguous_or_metadata_dependent",
            "affected_party_or_shared_context_same_event": "context_only",
        }
        if result["rejection_reason"] in rejection_reason_aliases:
            raw_reason = result["rejection_reason"]
            result["rejection_reason"] = rejection_reason_aliases[raw_reason]
            result["rejection_reason_validation_repair"] = (
                f"{raw_reason}_to_{result['rejection_reason']}_fail_closed"
            )
        if result["evidence_source"] == "visual_role":
            result["evidence_source"] = "generic_motion"
            result["evidence_source_validation_repair"] = (
                "visual_role_to_generic_motion_fail_closed"
            )
        for key in (
            "on_screen_social_event",
            "actor_performs_target_behavior",
            "affected_party_or_shared_context_same_event",
            "behavior_socially_evaluable_from_clip",
            "event_temporally_localized",
            "informal_social_conduct_not_formal_procedure",
            "usable_demo_after_relabel",
            "proposed_norm_supported",
        ):
            if result[key] not in yes_no_uncertain:
                raise ValueError(f"invalid v8 {key}: {result[key]!r}")
        if result["evidence_source"] not in {
            "physical_action",
            "situated_dialogue_or_subtitles",
            "static_depicted_action",
            "narration_only",
            "metadata_only",
            "generic_motion",
            "uncertain",
        }:
            raise ValueError("invalid v8 evidence_source")
        if result["visual_role"] not in {
            "demonstrated_event",
            "generic_context_broll",
            "presenter_or_interview",
            "montage_or_compilation",
            "aftermath_or_reaction_only",
            "formal_or_procedural",
            "ambiguous_story_excerpt",
            "uncertain",
        }:
            raise ValueError("invalid v8 visual_role")
        if result["rejection_reason"] not in {
            "none",
            "behavior_not_shown",
            "context_only",
            "presentation_only",
            "montage_only",
            "aftermath_only",
            "formal_or_procedural",
            "ambiguous_or_metadata_dependent",
            "uncertain",
        }:
            raise ValueError("invalid v8 rejection_reason")
        strict_fields = (
            "on_screen_social_event",
            "actor_performs_target_behavior",
            "affected_party_or_shared_context_same_event",
            "behavior_socially_evaluable_from_clip",
            "event_temporally_localized",
            "informal_social_conduct_not_formal_procedure",
        )
        internally_strict = (
            all(result[key] == "yes" for key in strict_fields)
            and result["evidence_source"]
            in {
                "physical_action",
                "situated_dialogue_or_subtitles",
                "static_depicted_action",
            }
            and result["visual_role"] == "demonstrated_event"
        )
        if result["usable_demo_after_relabel"] == "yes" and not internally_strict:
            result["usable_demo_after_relabel"] = "uncertain"
            result["usable_demo_validation_repair"] = (
                "inconsistent_yes_to_uncertain_fail_closed"
            )
    if rubric == "v9a":
        yes_no_uncertain = {"yes", "no", "uncertain"}
        if result["evidence_source"] == "generic_context_broll":
            result["evidence_source"] = "generic_motion"
            result["evidence_source_validation_repair"] = (
                "generic_context_broll_to_generic_motion_fail_closed"
            )
        for key in (
            "observable_event",
            "same_event_actor_action_target",
            "event_temporally_localized",
            "presentation_only",
            "montage_only",
            "aftermath_only",
            "metadata_needed_to_name_action",
        ):
            if result[key] not in yes_no_uncertain:
                raise ValueError(f"invalid v9a {key}: {result[key]!r}")
        if result["evidence_source"] not in {
            "physical_action",
            "situated_dialogue_or_subtitles",
            "static_depicted_action",
            "generic_motion",
            "narration_or_text_only",
            "none",
            "uncertain",
        }:
            raise ValueError("invalid v9a evidence_source")
        if result["scene_role"] not in {
            "demonstrated_event",
            "presenter_or_interview",
            "generic_context_broll",
            "montage_or_compilation",
            "aftermath_or_reaction_only",
            "formal_or_skill_procedure",
            "ambiguous_story_excerpt",
            "none",
            "uncertain",
        }:
            raise ValueError("invalid v9a scene_role")
        if result["demonstration_kind"] not in {
            "interpersonal_conduct",
            "shared_public_conduct",
            "explicit_social_etiquette",
            "formal_or_safety_rule",
            "motor_game_or_technical_skill",
            "generic_or_ambiguous",
            "none",
            "uncertain",
        }:
            raise ValueError("invalid v9a demonstration_kind")
        start_percent = result["event_start_percent"]
        end_percent = result["event_end_percent"]
        if not isinstance(start_percent, (int, float)) or not isinstance(
            end_percent, (int, float)
        ):
            raise ValueError("v9a event percentages must be numeric")
        if not (-1 <= start_percent <= 100 and -1 <= end_percent <= 100):
            result["event_start_percent"] = -1
            result["event_end_percent"] = -1
            result["temporal_bounds_validation_repair"] = (
                "out_of_range_to_unlocalized_fail_closed"
            )
            start_percent = -1
            end_percent = -1
        if (start_percent == -1) != (end_percent == -1):
            raise ValueError("v9a event percentages must both be -1 or both be set")
        if start_percent != -1 and start_percent > end_percent:
            raise ValueError("v9a event start must not follow event end")
        literal_fields_present = all(
            str(result[key]).strip().lower() not in {"", "none"}
            for key in (
                "literal_actor",
                "literal_action_or_situated_utterance",
                "literal_affected_party_or_shared_setting",
            )
        )
        internally_strict = (
            start_percent != -1
            and result["same_event_actor_action_target"] == "yes"
            and result["event_temporally_localized"] == "yes"
            and result["evidence_source"]
            in {
                "physical_action",
                "situated_dialogue_or_subtitles",
                "static_depicted_action",
            }
            and result["scene_role"] == "demonstrated_event"
            and literal_fields_present
        )
        if result["observable_event"] == "yes" and not internally_strict:
            result["observable_event"] = "uncertain"
            result["observable_event_validation_repair"] = (
                "inconsistent_yes_to_uncertain_fail_closed"
            )
    return result


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    mode: str,
    timeout: int,
    retries: int,
    fps: float,
    rubric: str,
    processor_sampling: bool,
) -> dict:
    clip = row.get("proxy_clip") or row["source_clip"]
    # Manifests are transferable; use the clip beside the transferred manifest
    # when the original absolute proxy path belongs to another host.
    local_candidate = Path(row["_manifest_dir"]) / "clips" / Path(clip).name
    if local_candidate.exists():
        clip = str(local_candidate)
    label = row.get("norm") or row.get("normalized_behavior") or "unspecified"
    explanation = row.get("explanation") or ""
    if mode == "blind":
        if rubric == "v1":
            user_text = (
                "Audit this video without guessing its topic. Does it directly show "
                "a physical or screen-mediated social interaction/action, rather "
                "than merely describing one? For labeled_event_directly_depicted, "
                "answer uncertain because no label is supplied."
            )
        elif rubric == "v2":
            user_text = (
                "Audit this video without a proposed label. Does it directly show "
                "any concrete, socially meaningful behavior event rather than only "
                "describing one? For proposed_label_event_visible answer uncertain."
            )
        elif rubric == "v3":
            user_text = (
                f"This is a {row['pillar']} candidate. Without a proposed label, "
                "decide whether it contains a situated social scenario rather than "
                "only audience-directed presentation. For proposed norm fit answer "
                "uncertain."
            )
        elif rubric in {"v4", "v5", "v8"}:
            user_text = (
                f"This is a {row['pillar']} candidate. Without a proposed label, "
                "decide whether it directly shows a concrete socially evaluated "
                "behavior event. For proposed_norm_supported answer uncertain."
            )
        elif rubric == "v9a":
            user_text = (
                "Record the strongest literal actor-action-target or public-setting "
                "event in this saved clip. You have no metadata. Do not infer a "
                "topic or norm. If imagery merely presents, illustrates, or reacts "
                "to an off-screen claim, record no event."
            )
        else:
            user_text = (
                "Without using a title, transcript, topic, moral, or proposed label, "
                "record the single strongest literal event visible in this saved "
                "clip. If no actor-action-target/shared-setting event is actually "
                "shown, say no event. Use clip-relative percentages."
            )
    else:
        if rubric == "v1":
            user_text = (
                f'The proposed behavior/norm label is: "{label}". Does the video '
                "directly depict the concrete labeled event itself? The label is a "
                "hypothesis, not evidence. Also classify the visible scene."
            )
        elif rubric == "v2":
            user_text = (
                f'Proposed norm label: "{label}".\n'
                f"Detector explanation (fallible, not visual evidence): "
                f'"{explanation[:1200]}".\n'
                "Does the video itself directly show a concrete event supporting "
                "that hypothesis? Identify the visible act, not just its setting."
            )
        elif rubric in {"v3", "v4", "v5", "v8"}:
            pillar_instruction = {
                "instructional": (
                    "A staged or scripted scene is valid. Look for characters "
                    "demonstrating the situation through actions or situated dialogue."
                ),
                "witnessed": (
                    "Separate visibility of the alleged violation action from the "
                    "reaction, and separately judge organic versus staged."
                ),
                "commentary": (
                    "Separate event footage/cutaways from a commentator merely "
                    "describing the event."
                ),
            }.get(row["pillar"], "")
            user_text = (
                f"Candidate pillar: {row['pillar']}.\n"
                f'Proposed norm/behavior: "{label}".\n'
                f"Detector transcript explanation (fallible): "
                f'"{explanation[:1200]}".\n'
                f"{pillar_instruction}"
            )
        elif rubric in {"v7", "v7b", "v7c"}:
            transcript = json.dumps(
                row.get("aligned_transcript") or [],
                ensure_ascii=False,
            )[:6000]
            user_text = (
                "Audit this witnessed candidate as atomic observations. Use video "
                "for visible action/authenticity and the timestamped transcript only "
                "to clarify grounded speech/reaction semantics.\n"
                f'Fallible proposed norm/behavior: "{label}".\n'
                f'Fallible detector explanation: "{explanation[:1200]}".\n'
                f"Timestamped clip-relative transcript: {transcript}"
            )
        else:
            user_text = (
                "First ignore the following fallible hypothesis and record only the "
                "literal actor-action-target event visible in the saved clip. Do not "
                "name or endorse the norm in your event description.\n"
                f'Fallible proposed norm: "{label}".\n'
                f'Fallible detector explanation: "{explanation[:1200]}".'
            )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": (
            800
            if rubric in {"v7", "v7b", "v7c"}
            else (700 if rubric in {"v6", "v8", "v9a"} else 500)
        ),
        "messages": [
            {
                "role": "system",
                "content": {
                    "v1": SYSTEM_V1,
                    "v2": SYSTEM_V2,
                    "v3": SYSTEM_V3,
                    "v4": SYSTEM_V4,
                    "v5": SYSTEM_V5,
                    "v6": SYSTEM_V6,
                    "v7": SYSTEM_V7,
                    "v7b": SYSTEM_V7B,
                    "v7c": SYSTEM_V7C,
                    "v8": SYSTEM_V8,
                    "v9a": SYSTEM_V9A,
                }[rubric],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": Path(clip).resolve().as_uri()}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if processor_sampling:
        # Opt in because some Qwen3-VL/vLLM combinations have an off-by-one
        # bug when the HF processor resamples an already decoded video. The
        # default path uses pre-rendered low-fps proxies and vLLM's bounded
        # sampling.
        payload["mm_processor_kwargs"] = {"fps": fps, "do_sample_frames": True}
    encoded = json.dumps(payload).encode()
    last_error = ""
    last_content = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            message = body["choices"][0]["message"]
            content = message.get("content")
            if not content:
                raise ValueError(
                    "empty response content; message="
                    f"{json.dumps(message, sort_keys=True)[:1200]}"
                )
            last_content = content
            parsed = parse_json(content, rubric)
            return {
                "item_id": row["item_id"],
                "pillar": row["pillar"],
                "uid": row["uid"],
                "gold_scene_visible": row.get("gold_scene_visible"),
                "gold_social_scene_visible": row.get("gold_social_scene_visible"),
                "gold_label_matched_visible": row.get("gold_label_matched_visible"),
                "gold_usable": row.get("gold_usable"),
                "mode": mode,
                "rubric": rubric,
                "model": model,
                "fps": fps,
                "processor_sampling": processor_sampling,
                "result": parsed,
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:2000]
            except Exception:
                detail = ""
            last_error = f"HTTPError {exc.code}: {detail}"
            if attempt < retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)
    return {
        "item_id": row["item_id"],
        "pillar": row["pillar"],
        "uid": row["uid"],
        "gold_scene_visible": row.get("gold_scene_visible"),
        "gold_social_scene_visible": row.get("gold_social_scene_visible"),
        "gold_label_matched_visible": row.get("gold_label_matched_visible"),
        "gold_usable": row.get("gold_usable"),
        "mode": mode,
        "rubric": rubric,
        "model": model,
        "fps": fps,
        "processor_sampling": processor_sampling,
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8271/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--mode", choices=("blind", "conditioned"), required=True)
    parser.add_argument(
        "--rubric",
        choices=(
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v7b",
            "v7c",
            "v8",
            "v9a",
        ),
        default="v1",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--processor-sampling",
        action="store_true",
        help=(
            "Ask the model processor to resample at --fps. Off by default; "
            "prefer pre-rendered low-fps proxies for Qwen3-VL stability."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pillar",
        choices=("instructional", "witnessed", "commentary"),
        help="Optionally evaluate only one pillar from a combined frozen manifest.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    if args.pillar:
        rows = [row for row in rows if row.get("pillar") == args.pillar]
    for row in rows:
        row["_manifest_dir"] = str(args.manifest.parent)
    completed = (
        successful_prediction_keys(load_jsonl(args.out))
        if args.out.exists()
        else set()
    )
    pending = [
        row
        for row in rows
        if (row["item_id"], args.mode, args.model, args.rubric) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.mode,
                args.timeout,
                args.retries,
                args.fps,
                args.rubric,
                args.processor_sampling,
            ): row["item_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(futures)} {result['item_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
