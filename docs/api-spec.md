# DermAI API Spec

## GET /health

Response:

```json
{
  "status": "ok",
  "service": "dermai-api",
  "environment": "development"
}
```

## GET /version

Response:

```json
{
  "name": "DermAI API",
  "version": "0.1.0",
  "commitSha": "dev",
  "apiBasePath": "/"
}
```

## POST /chat

Request:

```json
{
  "sessionId": "optional-session-id",
  "message": "What are warning signs of melanoma?",
  "mode": "chat"
}
```

Response:

```json
{
  "sessionId": "generated-session-id",
  "answer": "Grounded answer generated from retrieved dermatology evidence.",
  "citations": [
    {
      "id": "melanoma-warning-signs-5b4268b1823c",
      "title": "Melanoma Warning Signs - Risk context",
      "source": "DermAI Clinical Notes (2026)",
      "snippet": "Melanoma concern is higher when...",
      "href": "https://dermai.local/sources/melanoma-warning-signs"
    }
  ],
  "confidence": "medium",
  "disclaimer": "DermAI is informational only and does not replace a licensed dermatologist.",
  "followUps": [
    "Do you want a short checklist version of this answer?"
  ]
}
```

## POST /upload-image

Multipart form upload with file field `file`.

Response:

```json
{
  "sessionId": "generated-session-id",
  "status": "completed",
  "message": "Image analysis completed. You can now ask a follow-up question in the same session.",
  "imageAnalysis": {
    "predictedClass": "benign_pattern",
    "confidence": 0.96,
    "confidenceBand": "high",
    "summary": "The classifier scored this image closer to a benign-pattern class than a malignant one.",
    "caution": "This image result is best used as visual context for the chat.",
    "overlayImageDataUrl": "data:image/png;base64,..."
  }
}
```

## Error Handling

- API responses include an `x-request-id` header.
- Unhandled server failures return JSON with:

```json
{
  "detail": "DermAI hit an unexpected server error.",
  "requestId": "uuid"
}
```
