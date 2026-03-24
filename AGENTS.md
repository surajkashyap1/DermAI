# AGENTS

## Product Direction

DermAI is being rebuilt as a modern dermatology web product aligned with the original paper but implemented with a production-style stack. The current objective is iterative delivery, not paper replication.

## Phase Rules

1. Build in phases. Do not skip ahead into later-phase infrastructure unless it unblocks the current phase.
2. Keep the architecture modular: frontend in `apps/web`, backend in `apps/api`, service logic in `services/*`, docs in `docs`.
3. Preserve provider abstraction for LLM usage. Do not hardwire one provider deep into business logic.
4. Preserve confidence-aware medical UX. Responses should support caution, refusal, and escalation behavior.
5. Keep citations first-class. Do not generate post-hoc fake references.

## Engineering Conventions

- Frontend: Next.js App Router, TypeScript, Tailwind CSS, small reusable UI primitives.
- Backend: FastAPI, Pydantic models, explicit route modules, thin HTTP layer.
- Shared contracts: keep frontend TypeScript contracts in `packages/shared` and mirror backend payloads in Pydantic schemas.
- Docs should be updated when architecture or contracts change.

## Collaboration

- The user provides product direction and approval on scope changes.
- The coding agent implements one phase at a time and verifies the result before moving on.
- When introducing placeholders, label them clearly and keep the contract stable so later phases can replace internals without breaking the UI.
