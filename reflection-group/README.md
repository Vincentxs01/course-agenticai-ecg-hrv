# Reflection - Group Submission

This folder contains group reflection documents comparing AI agent performance to chat-based solutions.

## Requirements

### File Format
- **Format:** Markdown (`.md`)
- **Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3.md` (alphabetical order, ASCII only)
- **License:** Include license declaration (CC-BY-4.0 recommended)
- **Length:** Maximum 1 A4 page equivalent (~500 words)

### Content Requirements

The reflection document analyzes your group's experience comparing:
- **AI Agent approach** (your project implementation)
- **Pure Chat-based approach** (e.g., ChatGPT, Claude without agent features)

Your reflection must address:

| Section | Description |
|---------|-------------|
| **Tools** | Which tools did you use? |
| **Task Description** | What task did you compare? |
| **Agent Approach** | How did your AI agent solve the task? |
| **Chat Approach** | How did you attempt the same task with chat-only? |
| **Performance Comparison** | Quantitative and/or qualitative differences |
| **Analysis** | Why do you think the performance differs? |
| **Lessons Learned** | Key insights from the comparison |

---

## Grading Criteria

Deliverable scored as passed (1) if handed in with acceptable quality before the end of the course.

---

## Example Reflection

```markdown
# Reflection: Agent vs Chat-Based HRV Analysis

**Group:** 2026-Chen-Lin-Wang
**License:** CC-BY-4.0

## Tools

Antigravity (Gemini 3.0 pro) vs. ChatGPT-4

## Task Description

We compared approaches for analyzing a 6-minute ECG recording to extract
HRV metrics and generate a fatigue assessment report.

## Agent Approach

Our AI agent system:
1. Automatically loaded the ECG data file
2. Applied bandpass filtering (0.5-40 Hz)
3. Detected R-peaks using Pan-Tompkins algorithm
4. Calculated time-domain HRV metrics (SDNN, RMSSD, pNN50)
5. Generated a structured report with visualizations

The agent completed the full pipeline in ~30 seconds with no manual intervention
after initial setup and creation of a sufficiently detailed prompt.

## Chat Approach

Using ChatGPT-4:
1. We uploaded the data file and asked for HRV analysis
2. ChatGPT provided Python code snippets
3. We manually copied, ran, and debugged code
4. We asked follow-up questions for each metric
5. We manually compiled results into a report

Total time: ~45 minutes with multiple iterations.

## Performance Comparison

| Metric | Agent | Chat-based |
|--------|-------|------------|
| Time to complete | 30 sec | 45 min |
| Manual steps required | 1 (initiation) | 15+ |
| Errors encountered | 0 | 3 (code bugs) |
| Output consistency | 100% reproducible | Varied |

## Analysis

The performance difference occurs because:

1. **Workflow automation:** The agent executes a predefined pipeline that it
   developped, while chat requires manual orchestration of each step.

2. **Tool integration:** The agent directly interfaces with Python libraries
   and file systems. Chat can only generate code that humans must execute.

3. **State management:** The agent maintains context across the entire
   pipeline. In chat, we had to re-explain context for each step.

4. **Error handling:** The agent has built-in validation. Chat-generated
   code often had minor errors requiring debugging.

## Lessons Learned

- Agentic AI excels at **repetitive, multi-step workflows** with clear structure
- Chat-based AI is better for **exploratory, one-off tasks** where flexibility matters
- The agent's value increases with task complexity and repetition
- Initial agent setup requires more effort, but amortizes over multiple runs
```

---

## How to Write an Effective Reflection

### Do:
- Be specific with examples and numbers
- Honestly assess both approaches' strengths and weaknesses
- Connect observations to course concepts (agents, workflows, tools)
- Show genuine learning and insight

### Don't:
- Simply state "the agent was better"
- Make vague claims without evidence
- Ignore limitations of your agent approach
- Exceed the length limit

---

## Submission Checklist

Before submitting, verify:

- [ ] File named correctly: `YYYY-Name1-Name2-Name3.md` (alphabetical)
- [ ] Only ASCII characters in filename
- [ ] License declaration included
- [ ] All required sections present
- [ ] Maximum 1 A4 page (~500 words)
- [ ] No personally identifiable information (PII)
- [ ] All group members contributed and are listed
