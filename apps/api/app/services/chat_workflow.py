from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from app.schemas.contracts import Citation
from app.services.providers import GenerationRequest
from app.services.retrieval import RetrievalHit, RetrievalService


Intent = Literal["greeting", "product_help", "dermatology_qa", "off_topic", "emergency"]
EvidenceStatus = Literal["sufficient", "partial", "insufficient"]
VerificationStatus = Literal["verified", "partial", "weak"]


EMERGENCY_PATTERNS = (
    "trouble breathing",
    "cannot breathe",
    "swollen lips",
    "swollen tongue",
    "rapidly spreading rash",
    "blistering",
    "sepsis",
    "high fever",
    "anaphylaxis",
)

OFF_TOPIC_PATTERNS = (
    "weather",
    "bitcoin",
    "football",
    "movie",
    "stock market",
    "recipe",
    "capital of",
)

GREETING_PATTERNS = {"hi", "hello", "hey", "yo", "good morning", "good evening"}
PRODUCT_HELP_PATTERNS = (
    "upload an image",
    "upload image",
    "send an image",
    "send image",
    "upload a photo",
    "send a photo",
    "what can this app do",
    "what can you do",
    "how does this work",
)

QUERY_REWRITES = {
    "what is melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "define melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "what is a melanoma": "melanoma definition skin cancer warning signs dermatologist evaluation",
    "what is skin cancer": "skin cancer melanoma lesion warning signs dermatologist evaluation",
}

IMAGE_LABEL_HINTS = {
    "malignant_pattern": "malignant pattern melanoma skin cancer warning signs urgent review biopsy",
    "benign_pattern": "benign pattern nevus non malignant lesion follow-up monitoring",
}

RETRY_FACET_HINTS = {
    "melanoma": "melanoma overview diagnosis early recognition risk factors",
    "bcc": "basal cell carcinoma bcc non melanoma skin cancer features differential",
    "scc": "squamous cell carcinoma scc non melanoma skin cancer features differential",
    "warning_signs": "warning signs red flags abcde evolution change bleeding asymmetry",
    "follow_up": "surveillance follow-up dermatologist review biopsy threshold monitoring",
    "triage": "triage urgent review escalation dangerous lesion symptoms",
    "skin_cancer_types": "melanoma basal cell carcinoma squamous cell carcinoma non melanoma comparison overview",
    "melanoma_overview": "melanoma basics overview definition why it matters",
    "image_follow_up": "image follow-up lesion interpretation cautious context",
}


class WorkflowState(TypedDict, total=False):
    message: str
    mode: str
    session_history: str
    image_analysis: dict[str, Any] | None
    normalized_message: str
    intent: Intent
    rewritten_query: str
    effective_query: str
    expected_facets: list[str]
    decomposed_queries: list[str]
    covered_facets: list[str]
    missing_facets: list[str]
    image_context: str
    retrieved_hits: list[RetrievalHit]
    top_score: float
    evidence_status: EvidenceStatus
    evidence_summary: str
    verification_status: VerificationStatus
    verification_summary: str
    retry_count: int
    retry_query: str
    confidence: Literal["low", "medium", "high"]
    answer_text: str
    citations: list[Citation]
    follow_ups: list[str]
    disclaimer: str
    trace: list[str]


class ChatWorkflow:
    def __init__(self, retrieval: RetrievalService, provider) -> None:
        self.retrieval = retrieval
        self.provider = provider
        self.checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[RetrievalHit, Citation]
            )
        )
        self.graph = self._build()

    def checkpoint_config(self, session_id: str) -> dict[str, dict[str, str]]:
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }

    def _append_trace(self, state: WorkflowState, label: str) -> list[str]:
        return [*(state.get("trace") or []), label]

    def _normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", message.strip().lower())

    def _facet_label(self, facet: str) -> str:
        labels = {
            "melanoma": "melanoma",
            "bcc": "basal cell carcinoma",
            "scc": "squamous cell carcinoma",
            "warning_signs": "warning signs",
            "follow_up": "follow-up and monitoring",
            "triage": "triage and escalation",
            "skin_cancer_types": "different skin cancer types",
            "melanoma_overview": "melanoma overview",
            "image_follow_up": "image follow-up context",
        }
        return labels.get(facet, facet.replace("_", " "))

    def _extract_expected_facets(self, normalized: str, mode: str) -> list[str]:
        facets: list[str] = []
        if "melanoma" in normalized:
            facets.append("melanoma")
        if "basal cell" in normalized or "bcc" in normalized:
            facets.append("bcc")
        if "squamous cell" in normalized or "scc" in normalized:
            facets.append("scc")
        if "warning sign" in normalized or "red flag" in normalized or "abcde" in normalized:
            facets.append("warning_signs")
        if "follow-up" in normalized or "follow up" in normalized or "self-monitoring" in normalized:
            facets.append("follow_up")
        if "triage" in normalized or "urgent" in normalized:
            facets.append("triage")
        if (
            "types of skin cancer" in normalized
            or "different types of skin cancer" in normalized
            or ("skin cancer" in normalized and ("types" in normalized or "different" in normalized or "overview" in normalized))
        ):
            facets.append("skin_cancer_types")
        if ("tell me more" in normalized or "overview" in normalized or "explain" in normalized) and "melanoma" in normalized:
            facets.append("melanoma_overview")
        if mode == "image_follow_up":
            facets.append("image_follow_up")
        return list(dict.fromkeys(facets))

    def _build_decomposed_queries(self, base_query: str, facets: list[str]) -> list[str]:
        if len(facets) < 2:
            return []

        normalized_base = self._normalize(base_query)
        sub_queries: list[str] = []
        for facet in facets[:3]:
            hint = RETRY_FACET_HINTS.get(facet)
            if not hint:
                continue
            sub_queries.append(f"{normalized_base} {hint}".strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for query in sub_queries:
            normalized = query.lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(query)
        return deduped

    def _merge_hits(self, hit_groups: list[list[RetrievalHit]], limit: int) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hits in hit_groups:
            for rank, hit in enumerate(hits, start=1):
                candidate_score = round(hit.score - (rank - 1) * 0.005, 4)
                existing = merged.get(hit.id)
                if existing is None or candidate_score > existing.score:
                    merged[hit.id] = RetrievalHit(
                        id=hit.id,
                        title=hit.title,
                        source=hit.source,
                        section=hit.section,
                        snippet=hit.snippet,
                        href=hit.href,
                        score=candidate_score,
                        text=hit.text,
                        year=hit.year,
                        topic_tags=hit.topic_tags,
                        disease_tags=hit.disease_tags,
                        source_type=hit.source_type,
                        authority_level=hit.authority_level,
                        audience=hit.audience,
                        pmid=hit.pmid,
                        section_path=hit.section_path,
                    )
        merged_hits = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return merged_hits[:limit]

    def _hit_matches_facet(self, hit: RetrievalHit, facet: str) -> bool:
        normalized_title = f"{hit.title} {hit.section}".lower()
        topic_tokens = " ".join(hit.topic_tags).lower()
        disease_tokens = " ".join(hit.disease_tags).lower()
        combined = f"{normalized_title} {topic_tokens} {disease_tokens}"
        if facet in {"melanoma", "melanoma_overview"}:
            return "melanoma" in combined
        if facet == "bcc":
            return "bcc" in combined or "basal cell" in combined
        if facet == "scc":
            return "scc" in combined or "squamous cell" in combined
        if facet == "warning_signs":
            return any(token in combined for token in ("warning", "abcde", "red flag", "evolution", "triage"))
        if facet == "follow_up":
            return any(token in combined for token in ("follow-up", "follow up", "surveillance", "monitor"))
        if facet == "triage":
            return "triage" in combined or "urgent" in combined
        if facet == "skin_cancer_types":
            return any(
                token in combined
                for token in ("non-melanoma", "bcc", "basal cell", "scc", "squamous cell", "keratinocyte")
            )
        if facet == "image_follow_up":
            return True
        return False

    def _assess_evidence(
        self,
        hits: list[RetrievalHit],
        facets: list[str],
        top_score: float,
    ) -> tuple[EvidenceStatus, Literal["low", "medium", "high"], str, list[str], list[str]]:
        if not hits:
            return "insufficient", "low", "No grounded evidence was retrieved.", [], facets

        unique_titles = len({hit.title for hit in hits})
        covered_facets = [facet for facet in facets if any(self._hit_matches_facet(hit, facet) for hit in hits)]
        missing_facets = [facet for facet in facets if facet not in covered_facets]

        if not facets:
            if len(hits) >= 4 and top_score >= 0.22:
                return "sufficient", "high", "Evidence coverage is focused and supported by multiple retrieved chunks.", [], []
            if len(hits) >= 2 and top_score >= 0.16:
                return "partial", "medium", "Evidence coverage is useful but not broad.", [], []
            return "insufficient", "low", "Evidence coverage is narrow.", [], []

        coverage_ratio = len(covered_facets) / max(len(facets), 1)
        has_breadth = unique_titles >= 2
        if coverage_ratio >= 1.0 and has_breadth and top_score >= 0.2:
            status: EvidenceStatus = "sufficient"
            confidence: Literal["low", "medium", "high"] = "high"
        elif coverage_ratio >= 0.6 and top_score >= 0.16:
            status = "partial"
            confidence = "medium"
        else:
            status = "insufficient"
            confidence = "low"

        covered_text = ", ".join(self._facet_label(facet) for facet in covered_facets) if covered_facets else "none"
        if missing_facets:
            missing_text = ", ".join(self._facet_label(facet) for facet in missing_facets)
            summary = f"Evidence clearly covers: {covered_text}. Evidence is partial or weak for: {missing_text}."
        else:
            summary = f"Evidence clearly covers: {covered_text}."

        if unique_titles < 2 and confidence == "high":
            status = "partial"
            confidence = "medium"
        return status, confidence, summary, covered_facets, missing_facets

    def _build_retry_query(self, state: WorkflowState) -> str:
        base_query = state.get("rewritten_query") or state.get("effective_query") or state["message"]
        missing_facets = state.get("missing_facets") or []
        additions = [RETRY_FACET_HINTS[facet] for facet in missing_facets if facet in RETRY_FACET_HINTS]
        if not additions:
            additions = ["dermatology overview differential guideline triage follow-up"]
        return f"{base_query} {' '.join(additions)}".strip()

    def _apply_image_confidence_modifier(
        self,
        confidence: Literal["low", "medium", "high"],
        image_analysis: dict[str, Any] | None,
        mode: str,
    ) -> Literal["low", "medium", "high"]:
        if mode != "image_follow_up" or not image_analysis:
            return confidence

        image_band = image_analysis.get("confidenceBand")
        if image_band == "low":
            if confidence == "high":
                return "medium"
            return "low"
        if image_band == "medium" and confidence == "high":
            return "medium"
        return confidence

    def _content_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in {
                "and",
                "are",
                "can",
                "for",
                "from",
                "into",
                "that",
                "the",
                "this",
                "with",
            }
        }

    def _sentence_supported(self, sentence: str, hits: list[RetrievalHit]) -> bool:
        sentence_tokens = self._content_tokens(sentence)
        if len(sentence_tokens) < 3:
            return True

        for hit in hits:
            support_text = " ".join(
                [
                    hit.title,
                    hit.section,
                    hit.text,
                    " ".join(hit.topic_tags),
                    " ".join(hit.disease_tags),
                ]
            )
            hit_tokens = self._content_tokens(support_text)
            overlap = sentence_tokens.intersection(hit_tokens)
            if len(overlap) >= 3:
                return True
            if len(overlap) >= 2 and len(overlap) / max(len(sentence_tokens), 1) >= 0.25:
                return True
        return False

    def _verify_answer_support(
        self,
        answer_text: str,
        hits: list[RetrievalHit],
    ) -> tuple[VerificationStatus, str]:
        if not answer_text.strip() or not hits:
            return "weak", "No answer or evidence was available for citation verification."

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", answer_text)
            if part.strip()
        ]
        factual_sentences = [sentence for sentence in sentences if len(self._content_tokens(sentence)) >= 3]
        if not factual_sentences:
            return "verified", "Answer content is too short or generic to require detailed citation verification."

        supported = [sentence for sentence in factual_sentences if self._sentence_supported(sentence, hits)]
        support_ratio = len(supported) / len(factual_sentences)
        if support_ratio >= 0.95:
            return "verified", "Answer claims are well aligned with the retrieved evidence."
        if support_ratio >= 0.6:
            return "partial", "Some answer claims are supported, but broader parts of the response are only partially matched."
        return "weak", "The answer contains claims that are not strongly supported by the retrieved evidence."

    def classify_intent(self, state: WorkflowState) -> WorkflowState:
        normalized = self._normalize(state["message"])
        intent: Intent = "dermatology_qa"

        if normalized in GREETING_PATTERNS or (len(normalized.split()) <= 3 and normalized.startswith(tuple(GREETING_PATTERNS))):
            intent = "greeting"
        elif any(pattern in normalized for pattern in PRODUCT_HELP_PATTERNS):
            intent = "product_help"
        elif any(pattern in normalized for pattern in EMERGENCY_PATTERNS):
            intent = "emergency"
        elif any(pattern in normalized for pattern in OFF_TOPIC_PATTERNS):
            intent = "off_topic"
        elif state.get("mode") == "image_follow_up":
            intent = "dermatology_qa"

        return {
            "normalized_message": normalized,
            "intent": intent,
            "trace": self._append_trace(state, f"classify:{intent}"),
        }

    def route_intent(self, state: WorkflowState) -> str:
        return state["intent"]

    def respond_greeting(self, state: WorkflowState) -> WorkflowState:
        image_hint = ""
        if state.get("image_analysis"):
            image_hint = " You can also ask what the uploaded image result means."
        return {
            "answer_text": (
                f"Hello. Ask me about skin cancer, suspicious lesions, or upload an image and ask what it means.{image_hint}"
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:greeting"),
        }

    def respond_off_topic(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "I can help with dermatology-focused questions, lesion warning signs, and cautious triage guidance. "
                "I am not meant for general-purpose chat."
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [
                "What is melanoma?",
                "When should a lesion not be reassured as benign?",
            ],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:off_topic"),
        }

    def respond_product_help(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "Yes. You can chat here about skin cancer and suspicious lesions, and you can also upload an image on the right side of the app. "
                "Once the image is attached, you can keep chatting and ask follow-up questions about that result in the same session."
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:product_help"),
        }

    def respond_emergency(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "This sounds like a potentially urgent medical situation. DermAI should not handle possible "
                "emergencies such as breathing difficulty, severe blistering, or rapidly worsening systemic symptoms. "
                "Please seek urgent medical care immediately."
            ),
            "confidence": "high",
            "citations": [],
            "follow_ups": [
                "Do you want a summary of why this should be escalated urgently?",
            ],
            "disclaimer": "DermAI is informational only and does not replace emergency or specialist care.",
            "trace": self._append_trace(state, "respond:emergency"),
        }

    def rewrite_query(self, state: WorkflowState) -> WorkflowState:
        normalized = state["normalized_message"]
        rewritten = QUERY_REWRITES.get(normalized, normalized)
        expected_facets = self._extract_expected_facets(normalized, state.get("mode", "chat"))

        if rewritten == normalized:
            tokens = normalized.split()
            if "melanoma" in tokens and "definition" not in tokens and "what" in tokens:
                rewritten = f"{normalized} definition skin cancer warning signs"
            elif len(tokens) <= 4:
                rewritten = f"{normalized} dermatology lesion triage"
            elif "different types of skin cancer" in normalized or "types of skin cancer" in normalized:
                rewritten = (
                    f"{normalized} melanoma basal cell carcinoma squamous cell carcinoma "
                    "non melanoma skin cancer overview differences"
                )
            elif "tell me more" in normalized and "melanoma" in normalized and "skin cancer" in normalized:
                rewritten = (
                    f"{normalized} melanoma overview basal cell carcinoma squamous cell carcinoma "
                    "skin cancer types comparison"
                )

        image_context = ""
        effective_query = rewritten
        image_analysis = state.get("image_analysis")
        if state.get("mode") == "image_follow_up" and image_analysis:
            predicted_class = image_analysis.get("predictedClass", "")
            summary = image_analysis.get("summary", "")
            quality = image_analysis.get("quality", {}) or {}
            issues = quality.get("issues", [])
            quality_note = f" Quality issues: {'; '.join(issues)}." if issues else ""
            image_context = (
                f"Uploaded image result: {predicted_class}. "
                f"Summary: {summary}. "
                f"Confidence band: {image_analysis.get('confidenceBand', 'unknown')}.{quality_note}"
            ).strip()
            hint = IMAGE_LABEL_HINTS.get(predicted_class, "lesion follow-up dermatology evaluation")
            effective_query = f"{rewritten} {hint}"

        decomposed_queries = self._build_decomposed_queries(effective_query, expected_facets)

        return {
            "rewritten_query": rewritten,
            "effective_query": effective_query,
            "expected_facets": expected_facets,
            "decomposed_queries": decomposed_queries,
            "image_context": image_context,
            "trace": self._append_trace(state, "rewrite"),
        }

    def retrieve(self, state: WorkflowState) -> WorkflowState:
        requested_facets = state.get("expected_facets") or []
        top_k = 6 if len(requested_facets) >= 2 else 4
        if state.get("retry_count", 0) > 0:
            top_k += 2
        base_query = state.get("effective_query") or state["rewritten_query"]
        hit_groups = [self.retrieval.search(base_query, top_k=top_k)]

        if state.get("retry_count", 0) == 0:
            for sub_query in state.get("decomposed_queries") or []:
                hit_groups.append(self.retrieval.search(sub_query, top_k=max(3, top_k - 2)))

        hits = self._merge_hits(hit_groups, top_k)
        top_score = hits[0].score if hits else 0.0
        return {
            "retrieved_hits": hits,
            "top_score": top_score,
            "trace": self._append_trace(
                state,
                f"retrieve:{len(hits)}:queries={1 + len(state.get('decomposed_queries') or []) if state.get('retry_count', 0) == 0 else 1}",
            ),
        }

    def assess_evidence(self, state: WorkflowState) -> WorkflowState:
        hits = state.get("retrieved_hits") or []
        requested_facets = state.get("expected_facets") or []
        evidence_status, confidence, evidence_summary, covered_facets, missing_facets = self._assess_evidence(
            hits,
            requested_facets,
            state.get("top_score", 0.0),
        )
        confidence = self._apply_image_confidence_modifier(
            confidence,
            state.get("image_analysis"),
            state.get("mode", "chat"),
        )

        return {
            "evidence_status": evidence_status,
            "evidence_summary": evidence_summary,
            "covered_facets": covered_facets,
            "missing_facets": missing_facets,
            "confidence": confidence,
            "trace": self._append_trace(state, f"assess_evidence:{evidence_status}:{confidence}"),
        }

    def route_evidence(self, state: WorkflowState) -> str:
        retry_count = state.get("retry_count", 0)
        if retry_count < 1 and state.get("intent") == "dermatology_qa":
            if not state.get("retrieved_hits"):
                return "retry"
            if state.get("evidence_status") == "insufficient":
                return "retry"
            if state.get("evidence_status") == "partial" and state.get("missing_facets"):
                return "retry"
        return "no_hits" if not state.get("retrieved_hits") else "generate"

    def prepare_retry(self, state: WorkflowState) -> WorkflowState:
        retry_query = self._build_retry_query(state)
        return {
            "effective_query": retry_query,
            "retry_query": retry_query,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": self._append_trace(state, "retry:prepare"),
        }

    def respond_no_hits(self, state: WorkflowState) -> WorkflowState:
        return {
            "answer_text": (
                "I do not have a clear answer for that yet. Try asking in a more specific way, for example about a lesion type, warning sign, or skin cancer category."
            ),
            "confidence": "low",
            "citations": [],
            "follow_ups": [],
            "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
            "trace": self._append_trace(state, "respond:no_hits"),
        }

    async def generate_answer(self, state: WorkflowState) -> WorkflowState:
        hits = state["retrieved_hits"]
        context = "\n\n".join(
            f"[{index}] {hit.title} | {hit.section} | {hit.text}"
            for index, hit in enumerate(hits, start=1)
        )
        generation = await self.provider.generate(
            GenerationRequest(
                user_message=state["message"],
                retrieved_context=context,
                confidence=state["confidence"],
                intent=state["intent"],
                mode=state.get("mode", "chat"),
                conversation_history=state.get("session_history", ""),
                image_context=state.get("image_context", ""),
                evidence_summary=state.get("evidence_summary", ""),
            )
        )
        return {
            "answer_text": generation.answer,
            "trace": self._append_trace(state, f"generate:{generation.provider}"),
        }

    def verify_citations(self, state: WorkflowState) -> WorkflowState:
        hits = state.get("retrieved_hits") or []
        verification_status, verification_summary = self._verify_answer_support(
            state.get("answer_text", ""),
            hits,
        )

        confidence = state["confidence"]
        answer_text = state["answer_text"]
        if verification_status == "partial" and confidence == "high":
            confidence = "medium"
        elif verification_status == "weak":
            confidence = "low" if confidence in {"medium", "low"} else "medium"
            if "i can answer part of that" not in answer_text.lower():
                answer_text = (
                    "I can answer part of that clearly, but not all of it yet. "
                    f"{answer_text}"
                )

        return {
            "verification_status": verification_status,
            "verification_summary": verification_summary,
            "confidence": confidence,
            "answer_text": answer_text,
            "trace": self._append_trace(state, f"verify_citations:{verification_status}:{confidence}"),
        }

    def format_response(self, state: WorkflowState) -> WorkflowState:
        hits = state.get("retrieved_hits") or []
        citations = [
            Citation(
                id=hit.id,
                title=f"{hit.title} - {hit.section}",
                source=f"{hit.source} ({hit.year})",
                snippet=hit.snippet,
                href=hit.href,
            )
            for hit in hits
        ]

        if state["intent"] == "dermatology_qa" and state["confidence"] == "low":
            follow_ups = [
                "Would you like me to narrow this to warning signs, triage, or follow-up?",
                "Do you want the supporting evidence snippets shown first?",
            ]
        elif state.get("mode") == "image_follow_up":
            follow_ups = [
                "What features usually justify urgent in-person review?",
                "What should be monitored if the image result stays uncertain?",
            ]
        elif state["intent"] == "dermatology_qa":
            follow_ups = [
                "Do you want a short checklist version of this answer?",
                "Would you like the key supporting evidence summarized?",
            ]
        else:
            follow_ups = state.get("follow_ups") or []

        return {
            "citations": citations if hits else state.get("citations", []),
            "follow_ups": follow_ups,
            "disclaimer": state.get(
                "disclaimer",
                "DermAI is informational only and does not replace a licensed dermatologist.",
            ),
            "trace": self._append_trace(state, "format"),
        }

    def _build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("respond_greeting", self.respond_greeting)
        graph.add_node("respond_product_help", self.respond_product_help)
        graph.add_node("respond_off_topic", self.respond_off_topic)
        graph.add_node("respond_emergency", self.respond_emergency)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("assess_evidence", self.assess_evidence)
        graph.add_node("prepare_retry", self.prepare_retry)
        graph.add_node("respond_no_hits", self.respond_no_hits)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("verify_citations", self.verify_citations)
        graph.add_node("format_response", self.format_response)

        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self.route_intent,
            {
                "greeting": "respond_greeting",
                "product_help": "respond_product_help",
                "off_topic": "respond_off_topic",
                "emergency": "respond_emergency",
                "dermatology_qa": "rewrite_query",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "assess_evidence")
        graph.add_conditional_edges(
            "assess_evidence",
            self.route_evidence,
            {
                "no_hits": "respond_no_hits",
                "retry": "prepare_retry",
                "generate": "generate_answer",
            },
        )
        graph.add_edge("prepare_retry", "retrieve")
        graph.add_edge("generate_answer", "verify_citations")
        graph.add_edge("verify_citations", "format_response")
        graph.add_edge("respond_greeting", END)
        graph.add_edge("respond_product_help", END)
        graph.add_edge("respond_off_topic", END)
        graph.add_edge("respond_emergency", END)
        graph.add_edge("respond_no_hits", END)
        graph.add_edge("format_response", END)
        return graph.compile(checkpointer=self.checkpointer)
