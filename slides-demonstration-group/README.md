# Presentation Slides - Group Submission

This folder contains group presentation slides for the final demonstration.

## Requirements

### File Format
- **Format:** LaTeX Beamer (`.tex`)
- **Template:** NordlingLab Beamer 16:9 template (required)
- **Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3.tex` (alphabetical order, ASCII only)
- **License:** Include license declaration in the document (CC-BY-4.0 recommended)

### Why Beamer/LaTeX?

- Professional, consistent formatting
- Version control friendly (text-based)
- Mathematical notation support
- Reproducible builds
- Industry-standard for technical presentations

---

## Content Requirements

Your presentation should cover:

| Section | Suggested Slides |
|---------|------------------|
| Title slide | 1 |
| Problem (What?) & Motivation (Why?) | 1-2 |
| System Architecture | 2-3 |
| Demo / Results | 2-3 |
| Challenges & Lessons Learned | 1-2 |
| Conclusion & Q&A | 1 |

**Total:** Approximately 8-12 slides

---

## Grading Criteria

The oral presentation is scored based on:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Presentation Delivery & Organization | 20% | Clear structure, good pacing, confident delivery |
| Technical Depth & Soundness | 20% | Engineering design, accurate technical content, demonstrates understanding |
| Evaluation & Validation | 20% | Results, testing, evidence that the system works |
| Contribution & Impact | 10% | Significance of the problem and solution |
| Novelty / Creativity | 10% | Original ideas, creative approaches |
| Team Work | 10% | Evidence of collaboration, balanced contribution |
| Q & A Performance and Reflection | 10% | Thoughtful responses to questions |

---

## NordlingLab Beamer Template

Rewrite based on example in L2_hrv_ecg/main.tex
Include guide on how to setup template from https://bitbucket.org/nordlinglab/nordlinglab-template-beamer/

### Basic Template Structure

```latex
\documentclass[aspectratio=169]{beamer}

% NordlingLab theme settings
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]

% Packages
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

% Title information
\title{Your Project Title}
\subtitle{Agentic AI Course - Final Presentation}
\author{FamilyName1, FirstName1 \and FamilyName2, FirstName2 \and FamilyName3, FirstName3}
\institute{National Cheng Kung University}
\date{January 2026}

% License
% SPDX-License-Identifier: CC-BY-4.0

\begin{document}

% Title slide
\begin{frame}
    \titlepage
\end{frame}

% Table of contents (optional)
\begin{frame}{Outline}
    \tableofcontents
\end{frame}

% Content slides
\section{Problem \& Motivation}
\begin{frame}{Problem Statement}
    \begin{itemize}
        \item What problem are you solving?
        \item Why does it matter?
        \item Who are the stakeholders?
    \end{itemize}
\end{frame}

\section{System Architecture}
\begin{frame}{Architecture Overview}
    % Include your architecture diagram
    % \includegraphics[width=\textwidth]{architecture.png}

    \begin{itemize}
        \item Component 1: Description
        \item Component 2: Description
        \item Component 3: Description
    \end{itemize}
\end{frame}

\section{Demo \& Results}
\begin{frame}{Demonstration}
    % Screenshots, results, performance metrics
    \begin{itemize}
        \item Key result 1
        \item Key result 2
        \item Performance metrics
    \end{itemize}
\end{frame}

\section{Lessons Learned}
\begin{frame}{Challenges \& Insights}
    \begin{columns}
        \column{0.5\textwidth}
        \textbf{Challenges:}
        \begin{itemize}
            \item Challenge 1
            \item Challenge 2
        \end{itemize}

        \column{0.5\textwidth}
        \textbf{Lessons:}
        \begin{itemize}
            \item Lesson 1
            \item Lesson 2
        \end{itemize}
    \end{columns}
\end{frame}

\section{Conclusion}
\begin{frame}{Conclusion}
    \begin{itemize}
        \item Summary of achievements
        \item Future work possibilities
        \item Key takeaways
    \end{itemize}

    \vspace{1em}
    \centering
    \Large{Thank you! Questions?}
\end{frame}

\end{document}
```

---

## How to Compile LaTeX

### Option 1: Overleaf (Recommended for Beginners)

1. Go to https://www.overleaf.com
2. Create free account
3. New Project → Upload Project or Blank Project
4. Paste the template code
5. Click "Recompile" to see the PDF
6. Download the `.tex` file for submission

### Option 2: Local Installation

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download MiKTeX: https://miktex.org/download

**Linux:**
```bash
sudo apt install texlive-full
```

**Compile:**
```bash
xelatex 2026-Chen-Lin-Wang.tex
```

---

## Tips for Effective Presentations

### Design
- Use large fonts (minimum 20pt for body text)
- One main idea per slide
- Use visually clear diagrams and charts
- Limit bullet points (max 5-6 per slide)
- Include slide numbers

### Delivery
- Practice your timing
- Know your slides without reading them
- Make eye contact with the audience
- Speak clearly and at moderate pace
- Be prepared for questions

### Technical Content
- Start with the "why" before the "how"
- Use diagrams to explain architecture
- Show real results and tests, not just theory
- Be honest about limitations
- Demonstrate the system if possible

---

## Including Images

Place images in the same directory and reference them:

```latex
\begin{frame}{System Architecture}
    \centering
    \includegraphics[width=0.8\textwidth]{architecture-diagram.pdf}
\end{frame}
```

**Supported formats:** PNG, JPG, PDF (vector graphics preferred)

**Naming convention for images:**
`YYYY-FamilyName1-FamilyName2-FamilyName3-figurename.pdf`

---

## Submission Checklist

Before submitting, verify:

- [ ] File named correctly: `YYYY-Name1-Name2-Name3.tex` (alphabetical)
- [ ] Only ASCII characters in filename
- [ ] License declaration included in document
- [ ] Uses 16:9 aspect ratio (`aspectratio=169`)
- [ ] All images included with correct names
- [ ] Compiles without errors
- [ ] PDF renders correctly
- [ ] No personally identifiable information (PII)
- [ ] All group members listed as authors
