# AGENTS.md - AI Agent Guidelines

## Core Philosophy

**Minimal, surgical, precise.** Always write the least code necessary to achieve the goal. Prefer small targeted changes over large rewrites.

## Change Principles

### Surgical Edits
- Modify only what is strictly necessary
- Insert code at the most precise location possible
- Never refactor working code unless explicitly requested
- If a 3-line fix works, don't write 30 lines
- Preserve existing code structure and patterns

### Before Making Changes
1. Read the surrounding context (not just the target lines)
2. Understand existing patterns and conventions in the file
3. Check if similar functionality already exists elsewhere
4. Verify imports/dependencies are already available before adding new ones

### When Writing Code
- Follow existing naming conventions exactly
- Match the code style of the file (indentation, quotes, etc.)
- Reuse existing utilities and helpers before creating new ones
- Don't add comments unless explicitly requested
- Don't add emojis unless explicitly requested

## Project Structure

This is a **Zotero 7 plugin** with two main parts:

### Zotero Extension (`/` root)
- `manifest.json` — WebExtension manifest (Zotero 7, MV2)
- `chrome/` — UI overlays (XUL/XHTML)
- `modules/` — JavaScript modules (`.sys.mjs` ES modules)
- `locale/` — i18n strings
- `defaults/` — default preferences

### Python Webapp (`/webapp/`)
- Flask-based backend app
- AI provider integrations (OpenRouter, OpenAI, Google)
- RAG engine with vector store and reranker
- Zotero reader integration

## Language-Specific Rules

### JavaScript/TypeScript (`.sys.mjs`)
- ES modules format (`.sys.mjs`)
- Follow Zotero 7 extension conventions
- Use existing Zotero APIs and services

### Python (`/webapp/`)
- Flask patterns for routes and views
- Keep functions focused and small
- Use existing provider/store abstractions

## Decision Tree

```
User request → Can it be done with an existing function? → YES → Use it
                    ↓ NO
               Can it be done with a small insertion? → YES → Insert surgically
                    ↓ NO
               Does it require a new file? → YES → Create minimal file
                    ↓ NO
               Refactor only if explicitly requested
```

## Anti-Patterns (NEVER Do These)

- Do NOT rewrite entire files to add a small feature
- Do NOT change working code "to improve it" without being asked
- Do NOT add boilerplate or scaffolding beyond what's needed
- Do NOT introduce new dependencies unless necessary
- Do NOT rename variables/functions for style preferences
- Do NOT reformat code (preserve existing formatting)
- Do NOT add documentation files unless explicitly requested

## Good Practices

- Test your changes mentally by tracing the code flow
- Ensure imports are correct and paths resolve
- Keep error handling consistent with surrounding code
- Prefer composition over inheritance
- Fail fast, handle errors gracefully
- Security: never expose secrets, keys, or credentials in code

## Communication

- Answer concisely
- Show only the changed code or file paths when explaining
- Reference files with `path:line_number` format
- No preamble, no postamble — just the solution
