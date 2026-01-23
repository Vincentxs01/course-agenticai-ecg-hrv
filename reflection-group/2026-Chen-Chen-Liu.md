# Reflection: Agent vs Chat-Based ECG Pomodoro

**Group:** 2026-Chen-Chen-Liu  
**Authors:** Chen, Chen, Liu  
**License:** CC-BY-4.0

## Tools

- Our AI-agent-style system: React frontend + ECG FastAPI service + AI FastAPI service. The frontend generates synthetic ECG segments and calls the backend APIs automatically.
- Chat-based baseline: a pure chat session (e.g., ChatGPT/Claude) where the user manually asks for interpretation and manually executes any suggested steps.

## Task Description

We compared how to complete one Pomodoro work session analysis:

1. Generate/collect ECG segments during work.
2. Produce a session summary (HR/HRV-like metrics).
3. Produce user-facing advice for rest/next actions.

## Agent Approach

In our implementation, the frontend periodically generates ECG segments (synthetic) during work and buffers them. When work ends and the app enters rest mode, it sends a `pomodoro-work/v1` request to `POST /ecg/pomodoro/end` to obtain a `PomodoroSummary`, then sends the summary to `POST /ai/pomodoro/advice` to obtain structured advice for the user.

The AI service can optionally use Gemini via `GEMINI_API_KEY`, with a rule-based fallback when the API call fails or is not configured.

## Chat Approach

With chat-only, the user must describe the session context and ECG-related metrics (or paste raw data), then repeatedly ask for: (a) how to compute metrics, (b) how to interpret them, and (c) what suggestions to provide. Execution and verification (running code, debugging, consolidating results) are manual steps outside the chat.

## Performance Comparison

Qualitatively, the agent approach reduces “context switching” because data generation and API calls are integrated into the workflow and triggered by UI state transitions (WORK → REST).

Chat-only is flexible for open-ended reasoning, but it does not automatically run the multi-step pipeline; the user must orchestrate each step, which increases time and opportunities for mismatch (wrong schema, missing fields, inconsistent assumptions).

## Analysis

The performance difference mainly comes from workflow automation and interface contracts. Our system encodes a fixed API contract (segments → summary → advice) and executes it consistently, while chat-only relies on the user to maintain structure and pass correct inputs between steps.

However, our “agent” is only as reliable as the surrounding integration: without real ECG data, the pipeline demonstrates architecture more than physiological validity, and AI output quality depends on whether Gemini is configured or fallback is used.

## Lessons Learned

- Agent-style designs are most valuable for repetitive multi-step tasks with stable inputs/outputs and clear API boundaries.
- A chat-only approach is useful for exploration, but is inefficient for end-to-end execution unless the user is willing to manually run code and keep context synchronized.
- Even a simple fallback strategy (rule-based) improves robustness when external LLM calls fail, but it must be clearly communicated to avoid over-trusting the results.
