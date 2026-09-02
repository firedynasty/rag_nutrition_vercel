# Fix: Stanley's Key → OpenAI via /api/chat

## Problem

`ReportChat.js` had a localhost shortcut that bypassed the backend:

```javascript
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

if (isLocalhost && !useSharedKey) {
  // Direct browser → OpenAI call (no API route)
  // Works fine on localhost with own key
} else {
  // Calls /api/chat (requires vercel dev or Vercel deployment)
}
```

When using Stanley's key on localhost:
- `useSharedKey = true` → `!useSharedKey = false`
- Falls into `else` branch → calls `/api/chat`
- `npm start` (CRA dev server) does NOT serve `/api/*` routes
- Result: **404 / network error**

## Reference: Working Pattern (react-chess-analysis_vercel)

The chess project works because it has NO localhost shortcut.
Every request always goes through the backend `/api/rag-chess`:

```javascript
// chess: /api/rag-chess.py — key resolution (server-side)
if api_key:
    llm_key = api_key                         # user's own key
elif access_code:
    valid_code = os.environ.get("ACCESS_CODE")
    if access_code != valid_code:
        return 401
    llm_key = os.environ.get("OPENAI_API_KEY")  # Stanley's key
else:
    return 400
```

Frontend always sends to backend:
```javascript
fetch('/api/rag-chess', {
  body: JSON.stringify({
    apiKey: apiKey || undefined,
    accessCode: useSharedKey ? accessCode : undefined,
  })
})
```

No browser-to-OpenAI direct calls. No localhost special case.

## Fix Applied

Removed the `isLocalhost` branch from `ReportChat.js` `sendMessage()`.
All OpenAI calls now always route through `/api/chat`.

Before (lines ~318-413):
```javascript
const isLocalhost = ...;
if (isLocalhost && !useSharedKey) {
  // direct fetch to https://api.openai.com/...
} else {
  fetch('/api/chat', { ... })
}
```

After:
```javascript
// Always go through backend
fetch('/api/chat', { ... })
```

## Local Dev

Use `vercel dev` instead of `npm start` to serve both the React app
and `/api/*` serverless functions locally:

```bash
vercel dev
```

Env vars are loaded from `.env.local` automatically.
