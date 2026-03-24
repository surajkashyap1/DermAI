import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  UploadImageResponse,
  VersionResponse,
} from "@dermai/shared";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  requestId?: string;
  status?: number;

  constructor(message: string, status?: number, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

async function safeFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    cache: "no-store",
  });

  const requestId = response.headers.get("x-request-id") ?? undefined;

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {}

    throw new ApiError(message, response.status, requestId);
  }

  return (await response.json()) as T;
}

export async function getHealthStatus() {
  return safeFetch<HealthResponse>("/health");
}

export async function getVersion() {
  return safeFetch<VersionResponse>("/version");
}

export async function sendChat(payload: ChatRequest) {
  return safeFetch<ChatResponse>("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function uploadImage(formData: FormData) {
  return safeFetch<UploadImageResponse>("/upload-image", {
    method: "POST",
    body: formData,
  });
}
