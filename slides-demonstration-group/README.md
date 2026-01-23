# Presentation Slides - Group Submission

This folder contains group presentation slides for the final demonstration.

## Requirements

### File Format

- **Format:** LaTeX Beamer (`.tex`) with Figures (`.pdf`,`.png`, or `.jpg`)
- **Template:** NordlingLab Beamer 16:9 template (required)
- **Compiler:** XeLaTeX (required for the template)
- **Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3.tex` (alphabetical order, ASCII only). If you have less than three figures, then include them named `YYYY-FamilyName1-FamilyName2-FamilyName3-FigureX.pdf`, otherwise include all your figures in a folder named `YYYY-FamilyName1-FamilyName2-FamilyName3-Figures`.
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
| Demo / Results & Evaluation | 2-3 |
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

You **must** use the official NordlingLab Beamer template for your presentation. The template provides consistent branding and professional formatting, like your future employer will require.

### Template Repository

**Source:** https://bitbucket.org/nordlinglab/nordlinglab-template-beamer/

### Template Installation

Do not add the template files to any folder in this repository. The templates belong in their own external repository and should not be added to this one.

#### macOS

```bash
# 1. Find your TeX home directory
kpsewhich -expand-var '$TEXMFHOME'

# 2. Create the latex directory (if needed) and navigate to it
mkdir -p $(kpsewhich -expand-var '$TEXMFHOME')/tex/latex
cd $(kpsewhich -expand-var '$TEXMFHOME')/tex/latex

# 3. Clone the template repository
git clone https://bitbucket.org/nordlinglab/nordlinglab-template-beamer.git

# 4. Update the TeX database
texhash
```

#### Windows 10 (MiKTeX)

```cmd
# 1. Create local texmf directory structure
mkdir C:\localtexmf\tex\latex
cd C:\localtexmf\tex\latex

# 2. Clone the template repository
git clone https://bitbucket.org/nordlinglab/nordlinglab-template-beamer.git

# 3. Configure MiKTeX:
#    - Open "MiKTeX Settings (Admin)"
#    - Go to "Roots" tab
#    - Add "C:\localtexmf" as a root
#    - Go to "General" tab
#    - Click "Refresh FNDB"
```

#### Linux (Ubuntu/Debian)

```bash
# 1. Find and navigate to your TeX home directory
cd $(kpsewhich -var-value $TEXMFHOME)

# 2. Create the latex directory (may need sudo)
sudo mkdir -p tex/latex
cd tex/latex

# 3. Clone the template repository
sudo git clone https://bitbucket.org/nordlinglab/nordlinglab-template-beamer.git

# 4. Update the TeX database
sudo texhash
```

#### Alternative: Use TEXINPUTS (No Installation)

If you cannot install the template system-wide, you can use the `TEXINPUTS` environment variable:

```bash
# Clone the template to a local directory
git clone https://bitbucket.org/nordlinglab/nordlinglab-template-beamer.git

# Compile with TEXINPUTS pointing to the template
TEXINPUTS=./nordlinglab-template-beamer//: xelatex your-presentation.tex
```

---

## Template Structure

The NordlingLab template provides several theme files:

| File | Purpose |
|------|---------|
| `beamerthemeNordlingLab.sty` | Main theme (4:3 aspect ratio) |
| `beamerthemeNordlingLab169.sty` | 16:9 widescreen theme |
| `beamercolorthemeNordlingLab.sty` | Color definitions |
| `beamerfontthemeNordlingLab.sty` | Font settings |
| `beamerinnerthemeNordlingLab.sty` | Inner theme elements |
| `beamerouterthemeNordlingLab.sty` | Outer theme (4:3) |
| `beamerouterthemeNordlingLab169.sty` | Outer theme (16:9) |
| `Figures/` | Logo files and graphics |

---

## Basic Template Usage

### Minimal Example

```latex
% !TEX program = xelatex
% !TEX encoding = UTF-8 Unicode
%
% SPDX-License-Identifier: CC-BY-4.0
% Your Name, Your Group Members

% Use 16:9 aspect ratio (required for this course)
\documentclass[aspectratio=169]{beamer}
\mode<presentation>{
    \usetheme{NordlingLab169}
}

% Load packages
\usepackage[english]{babel}
\let\latinencoding\relax
\usepackage{xltxtra}
% Use a system font (Candara is the official font, but may not be available)
% Alternatives: Helvetica Neue (macOS), DejaVu Sans (Linux), Calibri (Windows)
\setsansfont{Helvetica Neue}[
    BoldFont = Helvetica Neue Bold,
    ItalicFont = Helvetica Neue Italic
]
\usepackage{graphicx}
\usepackage[yyyymmdd]{datetime}
\renewcommand{\dateseparator}{--}

% Presentation metadata
\title[Short Title]{Your Project Title}
\subtitle{Agentic AI Course - Final Presentation}
\author[Names]{FamilyName1,~FirstName1 \and FamilyName2,~FirstName2 \and FamilyName3,~FirstName3}
\institute[NCKU]{
    National Cheng Kung University\\
    Agentic AI Course
}
\date{Spring 2026}
\subject{Your presentation topic}

\begin{document}

% === TITLE SLIDE ===
\setbeamertemplate{background}[NLTitle]
\setbeamertemplate{footline}[NLTitle]
\begin{frame}
    \titlepage
\end{frame}

% === OUTLINE (with CC license footer) ===
\setbeamertemplate{background}[NLCC]
\setbeamertemplate{footline}[NLCC]
\begin{frame}{Outline}
    \tableofcontents
\end{frame}

% === CONTENT SLIDES ===
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
    \begin{itemize}
        \item Component 1: Description
        \item Component 2: Description
        \item Component 3: Description
    \end{itemize}
    % Include your architecture diagram:
    % \includegraphics[width=0.8\textwidth]{your-diagram.pdf}
\end{frame}

\section{Demo \& Results}

\begin{frame}{Results}
    \begin{itemize}
        \item Key result 1
        \item Key result 2
        \item Performance metrics
    \end{itemize}
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

## Background and Footline Templates

The NordlingLab theme provides different background and footline styles for different slide types:

### Title Slide
```latex
\setbeamertemplate{background}[NLTitle]
\setbeamertemplate{footline}[NLTitle]
```
Use for the title page. Shows NordlingLab branding without footer.

### Creative Commons Licensed Content
```latex
\setbeamertemplate{background}[NLCC]
\setbeamertemplate{footline}[NLCC]
```
Use for content under CC license. Shows CC logo and license URL in footer.

### Standard Slides (Non-CC Content)
```latex
\setbeamertemplate{background}[NL]
\setbeamertemplate{footline}[NL][Custom footer text]
```
Use for slides with content that is not under CC license (e.g., third-party images). Requires source attribution.

### Example: Switching Between Styles

```latex
% Title slide
\setbeamertemplate{background}[NLTitle]
\setbeamertemplate{footline}[NLTitle]
\begin{frame}
    \titlepage
\end{frame}

% CC licensed content
\setbeamertemplate{background}[NLCC]
\setbeamertemplate{footline}[NLCC]
\begin{frame}{Your Content}
    % Your CC-licensed content here
\end{frame}

% Slide with third-party image (not CC)
\setbeamertemplate{background}[NL]
\setbeamertemplate{footline}[NL][Source: Author Name, Year]
\begin{frame}{External Image}
    \includegraphics[width=\textwidth]{third-party-image.png}
\end{frame}

% Back to CC licensed content
\setbeamertemplate{background}[NLCC]
\setbeamertemplate{footline}[NLCC]
\begin{frame}{More Content}
    % Continue with your content
\end{frame}
```

---

## How to Compile

### Option 1: Overleaf (Recommended for Beginners)

1. Go to https://www.overleaf.com
2. Create free account
3. New Project → Upload Project
4. Upload all template files (`.sty` files and `Figures/` folder) along with your `.tex` file
5. Set compiler to **XeLaTeX** (Menu → Compiler → XeLaTeX)
6. Click "Recompile" to see the PDF
7. Download the `.tex` file for submission

**Important:** Overleaf requires you to upload the template files since they are not installed system-wide.

### Option 2: Local Compilation (After Template Installation)

```bash
# Using xelatex directly
xelatex 2026-Chen-Lin-Wang.tex

# If you have bibliography
xelatex 2026-Chen-Lin-Wang.tex
biber 2026-Chen-Lin-Wang
xelatex 2026-Chen-Lin-Wang.tex
xelatex 2026-Chen-Lin-Wang.tex
```

### Option 3: Using TEXINPUTS (Without Installation)

```bash
# Set TEXINPUTS to include template directory
export TEXINPUTS=./nordlinglab-template-beamer//:

# Then compile
xelatex 2026-Chen-Lin-Wang.tex
```

### Installing XeLaTeX

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download MiKTeX: https://miktex.org/download

**Linux:**
```bash
sudo apt install texlive-full
# or minimal:
sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-extra
```

---

## Useful Packages

The following packages are commonly used with the template:

```latex
% Mathematics
\usepackage{amsmath}
\usepackage{amssymb}

% SI units
\usepackage{siunitx}

% Code listings
\usepackage{listings}
\usepackage{xcolor}

% Tables
\usepackage{booktabs}

% Bibliography (if needed)
\usepackage[style=numeric,sorting=none,backend=biber]{biblatex}
\addbibresource{references.bib}

% Hyperlinks
\usepackage{hyperref}
```

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

## Submission Checklist

Before submitting, verify:

- [ ] File named correctly: `YYYY-Name1-Name2-Name3.tex` (alphabetical)
- [ ] Only ASCII characters in filename
- [ ] License declaration included in document (SPDX header)
- [ ] Uses NordlingLab169 theme with 16:9 aspect ratio
- [ ] Uses correct background/footline templates (NLTitle, NLCC, NL)
- [ ] All images included with correct names
- [ ] None of the Nordling Lab Beamer template files are included
- [ ] Compiles without errors using LuaLaTeX or XeLaTeX
- [ ] PDF renders correctly
- [ ] No personally identifiable information (PII)
- [ ] All group members listed as authors
