"""Answer generation for the RAG pipeline.

Primary path: Groq's free-tier, OpenAI-compatible chat models via ``langchain-groq``
(replacing the prototype's OpenAI GPT dependency). If no ``GROQ_API_KEY`` is set
(or the call fails), it falls back to a deterministic extractive answer built from
the retrieved passages, so the app always returns something grounded.
"""

from __future__ import annotations

from dataclasses import dataclass

from dermai.config import GROQ_MODEL, GROQ_TEMPERATURE, groq_enabled

SYSTEM_PROMPT = (
    "You are DermAI, a warm, helpful dermatology assistant for a general audience. "
    "Respond naturally, like a normal conversation. For greetings, thanks, or "
    "small talk, reply briefly and warmly and invite a dermatology question — do "
    "NOT say you lack information about greetings. For dermatology questions, base "
    "your factual claims primarily on the provided context; if the context does "
    "not cover something specific, say what you reasonably can and suggest seeing a "
    "dermatologist rather than refusing. Be clear and concise, use plain language, "
    "and never give a definitive diagnosis. Only add the reminder to consult a "
    "licensed dermatologist when the message is actually about a medical concern."
)

# Conversational messages that should skip retrieval entirely.
SMALLTALK_PROMPT = (
    "You are DermAI, a friendly dermatology assistant. The user sent a "
    "conversational/greeting message. Reply in one or two warm, natural sentences "
    "and gently invite a skin or dermatology question. Do not mention missing "
    "context, do not add medical disclaimers, and do not give medical advice."
)

_SMALLTALK_TERMS = {
    "hi", "hii", "hello", "hey", "heya", "yo", "hiya", "howdy", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you?", "how's it going", "hows it going",
    "what's up", "whats up", "thanks", "thank you", "thankyou", "ty",
    "ok", "okay", "cool", "nice", "great", "bye", "goodbye", "see you",
    "who are you", "what can you do", "what do you do", "help",
}


def is_smalltalk(message: str) -> bool:
    """True for greetings / thanks / chit-chat that shouldn't hit retrieval."""
    normalized = message.strip().lower().rstrip("!.")
    if not normalized:
        return True
    if normalized in _SMALLTALK_TERMS:
        return True
    # Short openers like "hi there", "hey!" — treat very short greetings loosely.
    words = normalized.split()
    if len(words) <= 3 and words[0] in {
        "hi", "hii", "hello", "hey", "heya", "yo", "hiya", "howdy", "thanks", "thank",
    }:
        return True
    return False


def smalltalk_answer(message: str, chat_history: list | None = None) -> "Answer":
    """Friendly conversational reply that bypasses retrieval."""
    if not groq_enabled():
        return Answer(
            text=(
                "Hi! I'm DermAI. Ask me anything about skin conditions, lesions, "
                "or upload an image and I'll help interpret it."
            ),
            provider="smalltalk-fallback",
        )
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=GROQ_MODEL, temperature=0.4)
        history_text = ""
        for user, bot in (chat_history or [])[-4:]:
            history_text += f"User: {user}\nDermAI: {bot}\n"
        prompt = (
            f"{SMALLTALK_PROMPT}\n\n"
            f"Conversation so far:\n{history_text or 'None'}\n\n"
            f"User: {message}\nDermAI:"
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        return Answer(text=content, provider="groq")
    except Exception:
        return Answer(
            text=(
                "Hi! I'm DermAI. Ask me anything about skin conditions, lesions, "
                "or upload an image and I'll help interpret it."
            ),
            provider="smalltalk-fallback",
        )


@dataclass
class Answer:
    text: str
    provider: str


def _format_context(documents) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "source")
        blocks.append(f"[{i}] ({source}) {doc.page_content.strip()}")
    return "\n\n".join(blocks)


def _extractive_answer(question: str, documents) -> Answer:
    if not documents:
        return Answer(
            text=(
                "I don't have grounded material on that yet. Try asking about a "
                "specific lesion type, warning sign, or skin-cancer category. "
                "Please consult a licensed dermatologist for medical decisions."
            ),
            provider="fallback",
        )
    snippets = " ".join(doc.page_content.strip().replace("\n", " ") for doc in documents[:3])
    snippets = snippets[:900]
    text = (
        f"Based on the curated dermatology sources: {snippets}\n\n"
        "This is informational only — please consult a licensed dermatologist "
        "for diagnosis or treatment."
    )
    return Answer(text=text, provider="extractive")


def generate_answer(question: str, documents, chat_history: list | None = None) -> Answer:
    """Generate a grounded answer from retrieved documents."""
    if not groq_enabled():
        return _extractive_answer(question, documents)

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=GROQ_MODEL, temperature=GROQ_TEMPERATURE)
        context = _format_context(documents)
        history_text = ""
        for user, bot in (chat_history or [])[-4:]:
            history_text += f"User: {user}\nDermAI: {bot}\n"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Conversation so far:\n{history_text or 'None'}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        return Answer(text=content, provider="groq")
    except Exception:
        return _extractive_answer(question, documents)
