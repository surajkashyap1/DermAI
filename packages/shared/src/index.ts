export type AppEnvironment = "development" | "staging" | "production";

export interface HealthResponse {
  status: "ok";
  service: string;
  environment: AppEnvironment | string;
}

export interface VersionResponse {
  name: string;
  version: string;
  commitSha: string;
  apiBasePath: string;
}

export interface ChatRequest {
  sessionId?: string;
  message: string;
  mode: "chat" | "image_follow_up";
}

export interface ChatResponse {
  sessionId: string;
  answer: string;
}

export interface VisionPrediction {
  label: string;
  confidence: number;
  rationale: string;
}

export interface VisionQuality {
  usable: boolean;
  issues: string[];
  contrast: number;
  sharpness: number;
  lesionCoverage: number;
  asymmetry: number;
}

export interface ImageAnalysis {
  predictedClass: string;
  confidence: number;
  confidenceBand: "low" | "medium" | "high";
  summary: string;
  caution: string;
  topPredictions: VisionPrediction[];
  quality: VisionQuality;
  overlayImageDataUrl: string;
  width: number;
  height: number;
}

export interface UploadImageResponse {
  sessionId: string;
  status: "pending" | "completed";
  message: string;
  imageAnalysis?: ImageAnalysis | null;
}

export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}
