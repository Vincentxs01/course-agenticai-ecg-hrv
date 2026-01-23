# Reflection: Agent vs Chat-Based ECG Analysis for Drowsiness Detection

**Group:** 2026-Wei-Wu-Zheng
**Authors:** Wei JungYing, Wu KunChe, Zheng YiKai
**License:** CC-BY-4.0

## Tools

- **Agentic AI:** Custom three-agent system with Streamlit UI (Python, scipy, numpy)
- **Chat-based:** ChatGPT-4 for code generation and analysis guidance

## Task Description

We compared two approaches for analyzing a 30-second ECG recording to extract HRV metrics, assess drowsiness risk, and generate a comprehensive report with recommendations.

## Agent Approach

Our AI agent system executes a fully automated pipeline:

1. User uploads ECG CSV file through Streamlit interface
2. **Agent 1** automatically applies bandpass filtering (0.5-40 Hz), notch filtering (50 Hz), and normalization
3. **Agent 2** detects R-peaks, calculates RR intervals, and computes HR, SDNN, and RMSSD
4. **Agent 3** queries MCP tools (weather, time risk, driving duration), calculates multi-factor risk score, and generates a detailed analysis report

Total processing time: **< 2 seconds** with no manual intervention after file upload.

## Chat Approach

Using ChatGPT-4 for the same task:

1. Uploaded ECG data and requested analysis
2. ChatGPT provided Python code snippets for filtering
3. Manually copied code, ran it, and debugged errors
4. Asked follow-up questions for R-peak detection
5. Requested HRV calculation code, manually executed
6. Asked for interpretation of results
7. Manually compiled findings into a report

Total time: **~40 minutes** with 12+ manual interactions and 2 code debugging cycles.

## Performance Comparison

| Metric | Agent | Chat-based |
|--------|-------|------------|
| Time to complete | 2 seconds | 40 minutes |
| Manual steps required | 1 (upload) | 12+ |
| Code errors encountered | 0 | 2 |
| Output consistency | 100% reproducible | Variable |
| Contextual integration | Automatic (MCP) | Manual prompting |
| Real-time capable | Yes | No |

## Analysis

The performance difference stems from four key factors:

1. **Workflow Automation:** The agent executes a predefined pipeline autonomously, while chat requires manual orchestration of each step with copy-paste between environments.

2. **Tool Integration:** The agent directly interfaces with Python libraries and MCP tools. Chat can only generate code that humans must execute and debug separately.

3. **State Management:** The agent maintains context across the entire pipeline. In chat, we repeatedly re-explained context (sampling rate, data format) for each step.

4. **Error Handling:** The agent has built-in validation and graceful degradation. Chat-generated code often had minor bugs (wrong variable names, missing imports) requiring debugging.

## Challenges Encountered

Our main challenges were:

1. **Agent architecture design:** Understanding how to properly divide responsibilities between agents was not intuitive at first. We had to iterate on which agent handles which task.

2. **Git workflow:** As beginners to version control, we were often uncertain about the correct commands for staging, committing, and pushing files. This added friction to the development process.

## Lessons Learned

- **Learning to drive AI:** The most valuable skill was learning how to effectively instruct AI agents to help complete a project—asking the right questions and providing clear context.

- **Git fundamentals:** Though still not fully proficient, we learned the basics of version control, which will be essential for future collaborative projects.

- **Each AI approach has its place:** Agentic AI excels at repetitive, structured workflows. Chat-based AI is better for exploration and learning new concepts. A hybrid approach is optimal.

- **MCP tools add significant value** by integrating contextual information (time, weather, duration) that would require multiple separate prompts in a chat-based approach.

- **Future improvements needed:** Real API integration for weather/location and real-time streaming support would make this system production-ready.
