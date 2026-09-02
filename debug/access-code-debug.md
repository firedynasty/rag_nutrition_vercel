# Debug: Invalid Access Code

## Symptoms

Both `/api/rag` and `/api/chat` returned 401:

```
Error: Invalid access code
/api/rag:1  Failed to load resource: the server responded with a status of 401 ()
RAG retrieval failed: Invalid access code
/api/chat:1  Failed to load resource: the server responded with a status of 401 ()
Error: Error: Invalid access code
```

## Root Causes Found

### 1. Wrong Vercel project linked

The CLI had lost the link to the correct project (`rag-nutrition-vercel`). When `vercel --prod` was run, it created a **new orphan project** (`react-chat-rag-nutrition`) with **no env vars at all** — so `process.env.ACCESS_CODE` was `undefined` and every request failed.

**Fix:** Relink to the correct project:
```bash
vercel link --project rag-nutrition-vercel --yes
```

Verify env vars are present:
```bash
vercel env ls
# Should show: OPENAI_API_KEY, ACCESS_CODE, QDRANT_API_KEY, QDRANT_URL
```

### 2. User was entering 15 chars, expected 16

After relinking, the error persisted. Added debug logging to narrow it down:

```javascript
// api/chat.js — temporary debug (now removed)
console.log(`[debug] received accessCode length: ${accessCode.length}`);
console.log(`[debug] env ACCESS_CODE length: ${validAccessCode?.length}`);
if (accessCode.trim() !== validAccessCode?.trim()) {
  return res.status(401).json({
    error: `Invalid access code (received ${accessCode.trim().length} chars, expected ${validAccessCode?.trim().length} chars)`
  });
}
```

Error returned: `Invalid access code (received 15 chars, expected 16 chars)`

**Fix:** User was missing the last character when typing the access code in the browser prompt dialog. Check the exact value with:
```bash
grep ACCESS_CODE .env.local
```

## Vercel Env Var Notes

- `vercel env pull .env.local` pulls **development** env vars only — production values may differ.
- Changing an env var in the Vercel dashboard takes effect on the **next function invocation** (no redeploy needed for Node.js functions).
- After `vercel link` to a new project, always run `vercel env ls` to confirm vars are present before deploying.

## Debug Technique

When access code mismatches are suspected, add this to the backend temporarily to compare lengths without exposing values:

```javascript
const validAccessCode = process.env.ACCESS_CODE;
console.log(`[debug] received: ${accessCode.trim().length} chars, expected: ${validAccessCode?.trim().length} chars, match: ${accessCode.trim() === validAccessCode?.trim()}`);
if (accessCode.trim() !== validAccessCode?.trim()) {
  return res.status(401).json({
    error: `Invalid access code (received ${accessCode.trim().length} chars, expected ${validAccessCode?.trim().length ?? 'undefined'} chars)`
  });
}
```

Remove after confirming the fix.
