"use client";

import type { ChatResponse, HealthResponse, UploadImageResponse, VersionResponse } from "@dermai/shared";
import Image from "next/image";
import { useEffect, useState } from "react";
import { ApiError, getHealthStatus, getVersion, sendChat, uploadImage } from "../lib/api";

const starterQuestions = [
  "What are warning signs that a pigmented lesion needs urgent review?",
  "How do dermatologists distinguish benign nevi from melanoma clinically?",
  "What follow-up would you recommend after a low-confidence classifier result?",
];

const imageFollowUpQuestions = [
  "What does this image result mean?",
  "When should a lesion like this be reviewed urgently?",
  "What follow-up is reasonable if the image result stays uncertain?",
];

export function DemoShell() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [version, setVersion] = useState<VersionResponse | null>(null);
  const [message, setMessage] = useState("");
  const [chatResult, setChatResult] = useState<ChatResponse | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadImageResponse | null>(null);
  const [loadingChat, setLoadingChat] = useState(false);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const suggestedQuestions = uploadResult ? imageFollowUpQuestions : starterQuestions;

  useEffect(() => {
    void (async () => {
      try {
        const [nextHealth, nextVersion] = await Promise.all([getHealthStatus(), getVersion()]);
        setHealth(nextHealth);
        setVersion(nextVersion);
      } catch {
        setHealth(null);
        setVersion(null);
      }
    })();
  }, []);

  useEffect(() => {
    return () => {
      if (localPreview) {
        URL.revokeObjectURL(localPreview);
      }
    };
  }, [localPreview]);

  async function handleChatSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!message.trim()) {
      return;
    }

    setLoadingChat(true);
    setChatError(null);

    try {
      const result = await sendChat({
        message,
        mode: uploadResult ? "image_follow_up" : "chat",
        sessionId: uploadResult?.sessionId ?? chatResult?.sessionId,
      });

      setChatResult(result);
    } catch (error) {
      if (error instanceof ApiError) {
        const requestId = error.requestId ? ` Request ID: ${error.requestId}` : "";
        setChatError(`${error.message}.${requestId}`);
      } else {
        setChatError("DermAI could not complete the chat request.");
      }
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

    const formData = new FormData();
    formData.append("file", file);
    setLocalPreview(URL.createObjectURL(file));

    setLoadingUpload(true);
    setUploadError(null);
    try {
      const result = await uploadImage(formData);
      setUploadResult(result);
    } catch (error) {
      if (error instanceof ApiError) {
        const requestId = error.requestId ? ` Request ID: ${error.requestId}` : "";
        setUploadError(`${error.message}.${requestId}`);
      } else {
        setUploadError("DermAI could not analyze that image.");
      }
    } finally {
      setLoadingUpload(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.35fr_0.95fr]">
      <section className="glass-card rounded-[2rem] p-6">
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <span className="eyebrow">Dermatology Chat</span>
          <span className="rounded-full border border-black/10 px-3 py-1 text-xs text-[var(--muted)]">
            {health ? `API ${health.status}` : "API unavailable"}
          </span>
          <span className="rounded-full border border-black/10 px-3 py-1 text-xs text-[var(--muted)]">
            {version ? version.version : "Version unknown"}
          </span>
        </div>

        <div className="space-y-4">
          <div className="rounded-[1.5rem] border border-black/10 bg-white/70 p-5">
            <p className="mb-2 text-sm font-semibold">Ask a dermatology question</p>
            <p className="text-sm leading-6 text-[var(--muted)]">
              The current demo supports grounded dermatology chat, image upload, and same-session
              multimodal follow-up. The vision model remains a cautious heuristic preview.
            </p>
          </div>

          <form onSubmit={handleChatSubmit} className="space-y-4">
            <div className="rounded-[1.5rem] border border-black/10 bg-white p-3 shadow-[0_14px_40px_rgba(23,31,26,0.06)]">
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                className="min-h-28 w-full resize-none bg-transparent px-2 py-2 text-base outline-none"
                placeholder="Ask about melanoma risk factors, lesion appearance, or follow-up guidance."
              />

              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-black/8 px-2 pt-3">
                <div className="space-y-1 text-sm text-[var(--muted)]">
                  <p>
                    {uploadResult
                      ? "Multimodal follow-up is active for this session."
                      : "Workflow-backed dermatology chat is live."}
                  </p>
                  <p className="text-xs">Press Enter to send. Use Shift+Enter for a new line.</p>
                </div>
                <button
                  type="submit"
                  disabled={loadingChat || !message.trim()}
                  className="rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-white transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loadingChat ? "Sending..." : "Send"}
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {suggestedQuestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => setMessage(question)}
                  className="rounded-full border border-black/10 bg-white px-3 py-2 text-xs text-[var(--muted)] transition hover:border-[var(--accent)] hover:text-[var(--foreground)]"
                >
                  {question}
                </button>
              ))}
            </div>
          </form>

          {uploadResult?.imageAnalysis ? (
            <div className="rounded-[1.5rem] border border-black/10 bg-[var(--accent-soft)]/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                Active Image Context
              </p>
              <p className="mt-2 text-sm leading-6 text-[var(--foreground)]">
                Follow-up questions in this session will use the uploaded image result as non-diagnostic context.
                Current visual pattern: {uploadResult.imageAnalysis.predictedClass}.
              </p>
            </div>
          ) : null}

          {chatError ? (
            <div className="rounded-[1.5rem] border border-[#d6432b]/20 bg-[#fff3ef] p-4 text-sm leading-6 text-[#8f2f1f]">
              {chatError}
            </div>
          ) : null}

          <div className="rounded-[1.5rem] border border-black/10 bg-[#f8faf8] p-5">
            <p className="mb-2 text-sm font-semibold">Assistant response</p>
            <p className="text-sm leading-6 text-[var(--foreground)]">
              {chatResult?.answer ?? "No answer yet. Submit a dermatology question to exercise the current grounded workflow."}
            </p>

            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div className="rounded-2xl border border-black/8 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Confidence
                </p>
                <p className="mt-2 text-sm">{chatResult?.confidence ?? "Pending"}</p>
              </div>
              <div className="rounded-2xl border border-black/8 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Safety
                </p>
                <p className="mt-2 text-sm">
                  {chatResult?.disclaimer ?? "DermAI will attach a medical disclaimer to grounded answers."}
                </p>
              </div>
            </div>

            {(chatResult?.followUps?.length ?? 0) > 0 ? (
              <div className="mt-4 rounded-[1.25rem] border border-black/8 bg-white px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Suggested follow-ups
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {chatResult?.followUps.map((followUp) => (
                    <button
                      key={followUp}
                      type="button"
                      onClick={() => setMessage(followUp)}
                      className="rounded-full border border-black/10 px-3 py-2 text-xs text-[var(--muted)] transition hover:border-[var(--accent)] hover:text-[var(--foreground)]"
                    >
                      {followUp}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <aside className="space-y-6">
        <section className="glass-card rounded-[2rem] p-6">
          <div className="mb-4 flex items-center justify-between">
            <span className="eyebrow">Vision Panel</span>
            <span className="text-xs text-[var(--muted)]">Demo heuristic vision</span>
          </div>

          <label className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-[1.75rem] border border-dashed border-[var(--accent)]/35 bg-white/65 px-6 py-8 text-center transition hover:border-[var(--accent)]">
            <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            <div className="mb-3 rounded-full bg-[var(--accent-soft)] px-3 py-1 text-xs font-semibold text-[var(--accent)]">
              Lesion Upload
            </div>
            <p className="text-base font-semibold">Drop in a lesion image or click to upload</p>
            <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--muted)]">
              Upload returns a demo heuristic lesion-analysis result with an overlay preview, then unlocks
              same-session dermatology follow-up questions.
            </p>
          </label>

          <div className="mt-4 rounded-[1.5rem] border border-black/10 bg-white p-4 text-sm">
            <p className="font-semibold">Upload status</p>
            <p className="mt-2 text-[var(--muted)]">
              {loadingUpload
                ? "Running vision analysis..."
                : uploadResult?.message ?? "No image uploaded yet."}
            </p>
          </div>

          {uploadError ? (
            <div className="mt-4 rounded-[1.5rem] border border-[#d6432b]/20 bg-[#fff3ef] p-4 text-sm leading-6 text-[#8f2f1f]">
              {uploadError}
            </div>
          ) : null}

          {uploadResult?.imageAnalysis ? (
            <div className="mt-4 space-y-4">
              <div className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Vision Result
                </p>
                <p className="mt-2 text-base font-semibold">{uploadResult.imageAnalysis.predictedClass}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                  {uploadResult.imageAnalysis.summary}
                </p>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div className="rounded-2xl border border-black/8 bg-[#f9f7f2] px-4 py-3">
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                      Confidence
                    </p>
                    <p className="mt-2 text-sm">
                      {uploadResult.imageAnalysis.confidenceBand} ({uploadResult.imageAnalysis.confidence})
                    </p>
                  </div>
                  <div className="rounded-2xl border border-black/8 bg-[#f9f7f2] px-4 py-3">
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                      Model
                    </p>
                    <p className="mt-2 text-sm">{uploadResult.imageAnalysis.modelName}</p>
                  </div>
                </div>
                <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
                  {uploadResult.imageAnalysis.caution}
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {imageFollowUpQuestions.map((followUp) => (
                    <button
                      key={followUp}
                      type="button"
                      onClick={() => setMessage(followUp)}
                      className="rounded-full border border-black/10 px-3 py-2 text-xs text-[var(--muted)] transition hover:border-[var(--accent)] hover:text-[var(--foreground)]"
                    >
                      {followUp}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Original
                  </p>
                  {localPreview ? (
                    <Image
                      src={localPreview}
                      alt="Uploaded lesion"
                      width={uploadResult.imageAnalysis.width}
                      height={uploadResult.imageAnalysis.height}
                      className="h-auto w-full rounded-[1rem] object-cover"
                      unoptimized
                    />
                  ) : null}
                </div>
                <div className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Overlay
                  </p>
                  <Image
                    src={uploadResult.imageAnalysis.overlayImageDataUrl}
                    alt="Lesion overlay"
                    width={uploadResult.imageAnalysis.width}
                    height={uploadResult.imageAnalysis.height}
                    className="h-auto w-full rounded-[1rem] object-cover"
                    unoptimized
                  />
                </div>
              </div>

              <div className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Top Predictions
                </p>
                <div className="mt-3 space-y-3">
                  {uploadResult.imageAnalysis.topPredictions.map((prediction) => (
                    <div key={prediction.label} className="rounded-[1rem] border border-black/8 bg-[#f9f7f2] px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold">{prediction.label}</p>
                        <p className="text-xs text-[var(--muted)]">{prediction.confidence}</p>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{prediction.rationale}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Image Quality
                </p>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <div className="rounded-[1rem] border border-black/8 bg-[#f9f7f2] px-4 py-3 text-sm">
                    Contrast: {uploadResult.imageAnalysis.quality.contrast}
                  </div>
                  <div className="rounded-[1rem] border border-black/8 bg-[#f9f7f2] px-4 py-3 text-sm">
                    Sharpness: {uploadResult.imageAnalysis.quality.sharpness}
                  </div>
                  <div className="rounded-[1rem] border border-black/8 bg-[#f9f7f2] px-4 py-3 text-sm">
                    Coverage: {uploadResult.imageAnalysis.quality.lesionCoverage}
                  </div>
                  <div className="rounded-[1rem] border border-black/8 bg-[#f9f7f2] px-4 py-3 text-sm">
                    Asymmetry: {uploadResult.imageAnalysis.quality.asymmetry}
                  </div>
                </div>
                {(uploadResult.imageAnalysis.quality.issues ?? []).length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {uploadResult.imageAnalysis.quality.issues.map((issue) => (
                      <p key={issue} className="text-sm leading-6 text-[var(--muted)]">
                        {issue}
                      </p>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}
        </section>

        <section className="glass-card rounded-[2rem] p-6">
          <div className="mb-4 flex items-center justify-between">
            <span className="eyebrow">Evidence Panel</span>
            <span className="text-xs text-[var(--muted)]">
              {uploadResult ? "Image-conditioned retrieval live" : "Grounded retrieval live"}
            </span>
          </div>

          <div className="space-y-3">
            {(chatResult?.citations ?? []).length === 0 ? (
              <div className="rounded-[1.5rem] border border-black/10 bg-white p-4 text-sm text-[var(--muted)]">
                Retrieved evidence cards will appear here after a chat request returns grounded support.
              </div>
            ) : (
              chatResult?.citations.map((citation) => (
                <div key={citation.id} className="rounded-[1.5rem] border border-black/10 bg-white p-4">
                  <p className="text-sm font-semibold">{citation.title}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.12em] text-[var(--muted)]">
                    {citation.source}
                  </p>
                  <p className="mt-3 text-sm text-[var(--muted)]">{citation.snippet}</p>
                  {citation.href ? (
                    <a
                      href={citation.href}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-3 inline-flex text-xs font-semibold text-[var(--accent)]"
                    >
                      Open source link
                    </a>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </section>
      </aside>
    </div>
  );
}
