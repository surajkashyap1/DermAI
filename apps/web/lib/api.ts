import type {
  ChatRequest,
  ChatResponse,
  ClassificationResponse,
} from "@dermai/shared";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(response: Response): Promise<string> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {}
  return message;
}

export async function sendChat(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${apiBaseUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!response.ok) throw new ApiError(await parseError(response), response.status);
  return (await response.json()) as ChatResponse;
}

export async function classifyImage(file: File): Promise<ClassificationResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${apiBaseUrl}/classify`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!response.ok) throw new ApiError(await parseError(response), response.status);
  return (await response.json()) as ClassificationResponse;
}
