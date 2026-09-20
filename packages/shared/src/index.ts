export interface HealthResponse {
  status: string;
  version: string;
  modelAvailable: boolean;
  llmBackend: string;
}

export interface ChatTurn {
  user: string;
  assistant: string;
}

export interface ChatRequest {
  message: string;
  history: ChatTurn[];
}

export interface SourceRef {
  source: string;
  page?: number | null;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  provider: string;
  sources: SourceRef[];
}

export interface ClassProbability {
  code: string;
  name: string;
  probability: number;
}

export interface ClassificationResponse {
  predictedClass: string;
  code: string;
  malignancy: string;
  description: string;
  confidence: number;
  gradcamImageDataUrl?: string | null;
  probabilities: ClassProbability[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}
