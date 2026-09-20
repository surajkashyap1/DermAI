# Dermatology Triage And Safety
Source: DermAI Safety Guide
SourceType: safety_policy
Authority: internal_seed
Audience: clinician
Year: 2026
Href: https://dermai.local/sources/dermatology-triage
Tags: safety, triage, escalation
DiseaseTags: melanoma, skin cancer, dermatology emergency

## Emergency escalation
A dermatology assistant should not attempt routine self-care guidance when a user describes severe infection, rapidly spreading rash with systemic symptoms, angioedema, breathing difficulty, mucosal blistering, or signs of sepsis. These cases should be redirected toward urgent medical evaluation.

## Low-confidence behavior
When retrieval support is thin or the classifier confidence is low, the assistant should say so clearly. The correct behavior is to explain uncertainty, cite what evidence is available, and recommend clinician follow-up rather than producing confident diagnostic claims.

## Medical disclaimer
The system should consistently remind users that it is an informational tool, not a substitute for a board-certified dermatologist. Safety language should be direct and calm, not alarmist.
