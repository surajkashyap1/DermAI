"use client";

import type {
  ChatMessage,
  ChatTurn,
  ClassificationResponse,
} from "@dermai/shared";
import Image from "next/image";
import { useState } from "react";
import { ApiError, classifyImage, sendChat } from "../lib/api";

export function DemoShell() {
  const [message, setMessage] = useState("");
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [loadingChat, setLoadingChat] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  async function handleChatSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextMessage = message.trim();
    if (!nextMessage) return;

    setLoadingChat(true);
    setChatError(null);
    setMessage("");
    const userTurn: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: nextMessage,
    };
    setConversation((current) => [...current, userTurn]);

    try {
      const response = await sendChat({ message: nextMessage, history });
      setConversation((current) => [
        ...current,
        { id: `assistant-${Date.now()}`, role: "assistant", content: response.answer },
      ]);
      setHistory((current) => [...current, { user: nextMessage, assistant: response.answer }]);
    } catch (error) {
      setMessage(nextMessage);
      setConversation((current) => current.filter((item) => item.id !== userTurn.id));
      setChatError(
        error instanceof ApiError ? error.message : "DermAI could not complete the chat request."
      );
    } finally {
      setLoadingChat(false);
    }
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setLocalPreview(URL.createObjectURL(file));
    setLoadingUpload(true);
    setUploadError(null);
    try {
      const response = await classifyImage(file);
      setResult(response);
    } catch (error) {
      setResult(null);
      setUploadError(
        error instanceof ApiError ? error.message : "DermAI could not analyze that image."
      );
    } finally {
      setLoadingUpload(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.6fr_0.9fr]">
      {/* Chat */}
      <section className="glass-card flex min-h-[78vh] flex-col rounded-[2.25rem] p-6 md:p-8">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">DermAI</h1>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--muted)] md:text-base">
            Grounded dermatology chat (LangChain + FAISS RAG) and HAM10000 skin-lesion
            classification with Grad-CAM explainability.
          </p>
        </div>

        {chatError ? (
          <div className="mt-4 rounded-[1.5rem] border border-[rgba(255,182,189,0.18)] bg-[var(--danger-soft)] p-4 text-sm leading-6 text-[var(--danger-fg)]">
            {chatError}
          </div>
        ) : null}

        <div className="mt-4 flex flex-1 flex-col gap-4">
          <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-[rgba(10,21,27,0.92)] p-4 md:p-5">
            <div className="flex h-full flex-col justify-end">
              <div className="space-y-4">
                {conversation.length === 0 ? (
                  <div className="rounded-[1.25rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-5 text-sm leading-7 text-[var(--muted)]">
                    Say hi, or ask about skin cancer, lesion warning signs, or your uploaded
                    image result.
                  </div>
                ) : (
                  conversation.map((entry) => (
                    <div
                      key={entry.id}
                      className={
                        entry.role === "user"
                          ? "ml-auto w-fit max-w-[85%] whitespace-pre-wrap rounded-[1.4rem] rounded-br-md bg-[var(--accent)] px-4 py-3 text-sm leading-7 text-[#041015] md:text-base"
                          : "mr-auto max-w-[92%] whitespace-pre-wrap rounded-[1.4rem] rounded-bl-md border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm leading-7 text-[var(--foreground)] md:text-base"
                      }
                    >
                      {entry.content}
                    </div>
                  ))
                )}
                {loadingChat ? (
                  <div className="mr-auto max-w-[92%] rounded-[1.4rem] rounded-bl-md border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm text-[var(--muted)]">
                    Thinking...
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <form onSubmit={handleChatSubmit}>
            <div className="rounded-[1.75rem] border border-white/10 bg-[var(--surface)] p-3 shadow-[0_24px_60px_rgba(0,0,0,0.28)]">
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                className="min-h-28 w-full resize-none bg-transparent px-2 py-2 text-base text-[var(--foreground)] outline-none placeholder:text-[var(--muted)]/80"
                placeholder="Ask a dermatology question..."
              />
              <div className="flex items-center justify-between gap-3 border-t border-white/10 px-2 pt-3">
                <p className="text-xs text-[var(--muted)]">Press Enter to send.</p>
                <button
                  type="submit"
                  disabled={loadingChat || !message.trim()}
                  className="rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-[#041015] transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loadingChat ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          </form>
        </div>
      </section>

      {/* Image classification */}
      <aside className="space-y-6">
        <section className="glass-card rounded-[2.25rem] p-6">
          <div className="mb-4">
            <p className="text-sm font-semibold text-[var(--foreground)]">Classify a lesion image</p>
            <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
              7-class HAM10000 CNN with a Grad-CAM explanation.
            </p>
          </div>

          <label className="flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-[1.75rem] border border-dashed border-[var(--accent)]/35 bg-[var(--surface-soft)] px-6 py-8 text-center transition hover:border-[var(--accent)]">
            <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            <p className="text-base font-semibold">Drop in an image or click to upload</p>
            <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--muted)]">
              JPEG, PNG, or WEBP.
            </p>
          </label>

          <div className="mt-4 rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4 text-sm">
            <p className="font-semibold">Status</p>
            <p className="mt-2 text-[var(--muted)]">
              {loadingUpload
                ? "Running inference and Grad-CAM..."
                : result
                  ? "Analysis complete."
                  : "No image uploaded yet."}
            </p>
          </div>

          {uploadError ? (
            <div className="mt-4 rounded-[1.5rem] border border-[rgba(255,182,189,0.18)] bg-[var(--danger-soft)] p-4 text-sm leading-6 text-[var(--danger-fg)]">
              {uploadError}
            </div>
          ) : null}

          {result ? (
            <div className="mt-4 space-y-4">
              <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Diagnosis
                </p>
                <p className="mt-2 text-base font-semibold">{result.predictedClass}</p>
                <p className="mt-1 text-xs text-[var(--accent)]">{result.malignancy}</p>
                <div className="mt-3 rounded-2xl border border-white/10 bg-[var(--surface-soft)] px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Confidence
                  </p>
                  <p className="mt-1 text-sm">{(result.confidence * 100).toFixed(2)}%</p>
                </div>
                <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{result.description}</p>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Original
                  </p>
                  {localPreview ? (
                    <Image
                      src={localPreview}
                      alt="Uploaded lesion"
                      width={320}
                      height={320}
                      className="h-auto w-full rounded-[1rem] object-cover"
                      unoptimized
                    />
                  ) : null}
                </div>
                <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Grad-CAM
                  </p>
                  {result.gradcamImageDataUrl ? (
                    <Image
                      src={result.gradcamImageDataUrl}
                      alt="Grad-CAM overlay"
                      width={320}
                      height={320}
                      className="h-auto w-full rounded-[1rem] object-cover"
                      unoptimized
                    />
                  ) : (
                    <p className="text-sm text-[var(--muted)]">Overlay unavailable.</p>
                  )}
                </div>
              </div>

              <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Class probabilities
                </p>
                <div className="mt-3 space-y-2">
                  {[...result.probabilities]
                    .sort((a, b) => b.probability - a.probability)
                    .map((p) => (
                      <div key={p.code}>
                        <div className="flex items-center justify-between text-xs text-[var(--muted)]">
                          <span>{p.name}</span>
                          <span>{(p.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-[var(--surface-soft)]">
                          <div
                            className="h-full rounded-full bg-[var(--accent)]"
                            style={{ width: `${Math.max(p.probability * 100, 1)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              <p className="text-xs leading-6 text-[var(--muted)]">
                Automated pattern classification, not a diagnosis. Consult a licensed
                dermatologist.
              </p>
            </div>
          ) : null}
        </section>
      </aside>
    </div>
  );
}
