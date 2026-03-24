"use client";

import type { SessionMessage, UploadImageResponse } from "@dermai/shared";
import Image from "next/image";
import { useEffect, useState } from "react";
import { ApiError, sendChat, uploadImage } from "../lib/api";

export function DemoShell() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversation, setConversation] = useState<SessionMessage[]>([]);
  const [uploadResult, setUploadResult] = useState<UploadImageResponse | null>(null);
  const [loadingChat, setLoadingChat] = useState(false);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (localPreview) {
        URL.revokeObjectURL(localPreview);
      }
    };
  }, [localPreview]);

  async function handleChatSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextMessage = message.trim();
    if (!nextMessage) {
      return;
    }

    setLoadingChat(true);
    setChatError(null);
    setMessage("");

    const userTurn: SessionMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: nextMessage,
    };
    setConversation((current) => [...current, userTurn]);

    try {
      const result = await sendChat({
        message: nextMessage,
        mode: uploadResult ? "image_follow_up" : "chat",
        sessionId: uploadResult?.sessionId ?? sessionId ?? undefined,
      });
      setSessionId(result.sessionId);
      setConversation((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.answer,
        },
      ]);
    } catch (error) {
      setMessage(nextMessage);
      setConversation((current) => current.filter((item) => item.id !== userTurn.id));
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
      setSessionId(result.sessionId);
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
    <div className="grid gap-6 lg:grid-cols-[1.7fr_0.85fr]">
      <section className="glass-card flex min-h-[78vh] flex-col rounded-[2.25rem] p-6 md:p-8">
        <div className="space-y-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">DermAI</h1>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--muted)] md:text-base">
              Ask about skin cancer, lesion warning signs, or upload an image and continue the conversation with visual context.
            </p>
          </div>

          {uploadResult?.imageAnalysis ? (
            <div className="rounded-[1.5rem] border border-white/10 bg-[var(--accent-soft)] p-4">
              <p className="text-sm leading-6 text-[var(--foreground)]">
                Image context attached. Current visual pattern: {uploadResult.imageAnalysis.predictedClass}.
              </p>
            </div>
          ) : null}

          {chatError ? (
            <div className="rounded-[1.5rem] border border-[rgba(255,182,189,0.18)] bg-[var(--danger-soft)] p-4 text-sm leading-6 text-[var(--danger-fg)]">
              {chatError}
            </div>
          ) : null}
        </div>

        <div className="mt-4 flex flex-1 flex-col gap-4">
          <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-[rgba(10,21,27,0.92)] p-4 md:p-5">
            <div className="flex h-full flex-col justify-end">
              <div className="space-y-4">
                {conversation.length === 0 ? (
                  <div className="rounded-[1.25rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-5 text-sm leading-7 text-[var(--muted)]">
                    Start the conversation by asking a question. Your previous messages and answers will stay here.
                  </div>
                ) : (
                  conversation.map((entry) => (
                    <div
                      key={entry.id}
                      className={
                        entry.role === "user"
                          ? "ml-auto w-fit max-w-[85%] rounded-[1.4rem] rounded-br-md bg-[var(--accent)] px-4 py-3 text-sm leading-7 text-[#041015] md:text-base"
                          : "mr-auto max-w-[92%] rounded-[1.4rem] rounded-bl-md border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm leading-7 text-[var(--foreground)] md:text-base"
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
                placeholder="Type your dermatology question here."
              />

              <div className="flex items-center justify-between gap-3 border-t border-white/10 px-2 pt-3">
                <p className="text-xs text-[var(--muted)]">
                  {uploadResult ? "Image context is attached to this session." : "Press Enter to send."}
                </p>
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

      <aside className="space-y-6">
        <section className="glass-card rounded-[2.25rem] p-6">
          <div className="mb-4">
            <p className="text-sm font-semibold text-[var(--foreground)]">Add an image</p>
            <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
              Optional visual context for the chat session.
            </p>
          </div>

          <label className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-[1.75rem] border border-dashed border-[var(--accent)]/35 bg-[var(--surface-soft)] px-6 py-8 text-center transition hover:border-[var(--accent)]">
            <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            <div className="mb-3 rounded-full bg-[var(--accent-soft)] px-3 py-1 text-xs font-semibold text-[var(--accent)]">
              Image Upload
            </div>
            <p className="text-base font-semibold">Drop in an image or click to upload</p>
            <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--muted)]">
              The uploaded image stays attached to the chat so you can ask follow-up questions with visual context.
            </p>
          </label>

          <div className="mt-4 rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4 text-sm">
            <p className="font-semibold">Upload status</p>
            <p className="mt-2 text-[var(--muted)]">
              {loadingUpload
                ? "Running vision analysis..."
                : uploadResult?.message ?? "No image uploaded yet."}
            </p>
          </div>

          {uploadError ? (
            <div className="mt-4 rounded-[1.5rem] border border-[rgba(255,182,189,0.18)] bg-[var(--danger-soft)] p-4 text-sm leading-6 text-[var(--danger-fg)]">
              {uploadError}
            </div>
          ) : null}

          {uploadResult?.imageAnalysis ? (
            <div className="mt-4 space-y-4">
              <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Vision Result
                </p>
                <p className="mt-2 text-base font-semibold">{uploadResult.imageAnalysis.predictedClass}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                  {uploadResult.imageAnalysis.summary}
                </p>
                <div className="mt-4 rounded-2xl border border-white/10 bg-[var(--surface-soft)] px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    Confidence
                  </p>
                  <p className="mt-2 text-sm">
                    {uploadResult.imageAnalysis.confidenceBand} ({uploadResult.imageAnalysis.confidence})
                  </p>
                </div>
                <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
                  {uploadResult.imageAnalysis.caution}
                </p>
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
                      width={uploadResult.imageAnalysis.width}
                      height={uploadResult.imageAnalysis.height}
                      className="h-auto w-full rounded-[1rem] object-cover"
                      unoptimized
                    />
                  ) : null}
                </div>
                <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
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

              <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Top Predictions
                </p>
                <div className="mt-3 space-y-3">
                  {uploadResult.imageAnalysis.topPredictions.map((prediction) => (
                    <div key={prediction.label} className="rounded-[1rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold">{prediction.label}</p>
                        <p className="text-xs text-[var(--muted)]">{prediction.confidence}</p>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{prediction.rationale}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[1.5rem] border border-white/10 bg-[var(--surface)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Image Quality
                </p>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <div className="rounded-[1rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm">
                    Contrast: {uploadResult.imageAnalysis.quality.contrast}
                  </div>
                  <div className="rounded-[1rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm">
                    Sharpness: {uploadResult.imageAnalysis.quality.sharpness}
                  </div>
                  <div className="rounded-[1rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm">
                    Coverage: {uploadResult.imageAnalysis.quality.lesionCoverage}
                  </div>
                  <div className="rounded-[1rem] border border-white/10 bg-[var(--surface-soft)] px-4 py-3 text-sm">
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
      </aside>
    </div>
  );
}
