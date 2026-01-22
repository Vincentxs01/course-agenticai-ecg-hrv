# SPDX-License-Identifier: Apache-2.0
"""Report generation with AI-powered interpretation using Claude Opus 4.5."""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Optional imports for PDF generation (Removed reportlab, now generating Markdown)
REPORTLAB_AVAILABLE = False # No longer attempting to import

# Optional import for Claude API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def generate_interpretation(
    features: dict,
    pass_rate: Optional[float] = None, # New argument
    evaluation_summary: str = "N/A",   # New argument
    model: str = "claude-opus-4-5-20251101"
) -> dict:
    """
    Use Claude Opus 4.5 to generate interpretive text for the report.

    Args:
        features: Dictionary of HRV features
        pass_rate: Overall pass rate from rule-based evaluation.
        evaluation_summary: A text summary of the rule-based evaluation.
        model: Claude model to use

    Returns:
        dict: Contains 'discussion' and 'conclusion' sections
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "discussion": "AI interpretation unavailable (anthropic package not installed).",
            "conclusion": "Please install the anthropic package for AI-generated interpretations.",
        }

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "discussion": "AI interpretation unavailable (ANTHROPIC_API_KEY not set).",
            "conclusion": "Please set the ANTHROPIC_API_KEY environment variable.",
        }

    client = anthropic.Anthropic()

    # Modified prompt to reflect rule-based evaluation
    prompt = f"""Analyze these HRV metrics and baseline evaluation results.
Write a Discussion and Conclusion section for a clinical HRV analysis report.

HRV Metrics:
- SDNN: {features.get('sdnn', 'N/A'):.2f} ms (normal range: 50-100 ms)
- RMSSD: {features.get('rmssd', 'N/A'):.2f} ms (normal range: 20-50 ms)
- pNN50: {features.get('pnn50', 'N/A'):.2f}% (normal range: 10-25%)
- Mean HR: {features.get('mean_hr', 'N/A'):.1f} bpm
- LF Power: {features.get('lf_power', 'N/A'):.2f} ms²
- HF Power: {features.get('hf_power', 'N/A'):.2f} ms²
- LF/HF Ratio: {features.get('lf_hf_ratio', 'N/A'):.2f} (normal range: 1.0-2.0)

Baseline Evaluation Results:
- Overall Pass Rate: {pass_rate:.1%}
- Summary: {evaluation_summary}

Provide:
1. **Discussion** (2-3 paragraphs): Interpret the HRV metrics in clinical context.
   Explain what the values indicate about autonomic nervous system balance.
   Discuss the overall pass rate and what the evaluation summary implies about physiological consistency.

2. **Conclusion** (1 paragraph): Summarize findings and provide recommendations
   for the individual (e.g., suggestions based on consistency with baseline).

Use professional medical report language. Be specific about the metrics and the rule-based evaluation.
Format with clear section headers."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text

        # Parse response into sections
        if "Conclusion" in text:
            parts = text.split("Conclusion", 1)
            discussion = parts[0].replace("Discussion", "").replace("**", "").strip()
            conclusion = parts[1].replace("**", "").strip()
            # Remove leading colon or punctuation
            if conclusion.startswith(":"):
                conclusion = conclusion[1:].strip()
        else:
            discussion = text
            conclusion = ""

        return {
            "discussion": discussion,
            "conclusion": conclusion,
        }

    except Exception as e:
        return {
            "discussion": f"AI interpretation failed: {str(e)}",
            "conclusion": "Please review the HRV metrics manually.",
        }


def create_visualizations(
    ecg_data: dict,
    processed: dict,
    features: dict,
    pass_rate: Optional[float] = None, # New argument
    evaluation_summary: str = "N/A"    # New argument
) -> bytes:
    """
    Create visualization plots for the report.

    Args:
        ecg_data: Raw ECG data dictionary
        processed: Processed signal data
        features: HRV features
        pass_rate: Overall pass rate from rule-based evaluation.
        evaluation_summary: A text summary of the rule-based evaluation.

    Returns:
        bytes: PNG image data
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('HRV Analysis Results', fontsize=14, fontweight='bold')

    # Plot 1: ECG Signal with R-peaks
    ax1 = axes[0, 0]
    signal = processed.get('filtered_signal', ecg_data.get('signal', []))
    fs = processed.get('sampling_rate', 500)
    r_peaks = processed.get('r_peaks', [])

    # Show first 10 seconds
    n_samples = min(len(signal), int(10 * fs))
    time = np.arange(n_samples) / fs

    ax1.plot(time, signal[:n_samples], 'b-', linewidth=0.5, label='ECG')
    peak_mask = r_peaks < n_samples
    if np.any(peak_mask):
        peak_times = r_peaks[peak_mask] / fs
        peak_vals = signal[r_peaks[peak_mask]]
        ax1.scatter(peak_times, peak_vals, c='red', s=30, label='R-peaks', zorder=5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('ECG Signal (first 10s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: HRV Time-Domain Features
    ax2 = axes[0, 1]
    feature_names = ['SDNN', 'RMSSD', 'pNN50']
    feature_values = [
        features.get('sdnn', 0),
        features.get('rmssd', 0),
        features.get('pnn50', 0)
    ]
    normal_ranges = [(50, 100), (20, 50), (10, 25)]

    x_pos = np.arange(len(feature_names))
    bars = ax2.bar(x_pos, feature_values, color=['#3498db', '#2ecc71', '#9b59b6'])

    # Add normal range indicators
    for i, (low, high) in enumerate(normal_ranges):
        ax2.axhline(y=low, xmin=i/3, xmax=(i+1)/3, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=high, xmin=i/3, xmax=(i+1)/3, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names)
    ax2.set_ylabel('Value (ms / %)')
    ax2.set_title('Time-Domain HRV Features')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Frequency Domain Power
    ax3 = axes[1, 0]
    powers = [
        features.get('lf_power', 0),
        features.get('hf_power', 0)
    ]
    labels = ['LF Power\n(0.04-0.15 Hz)', 'HF Power\n(0.15-0.4 Hz)']
    colors_freq = ['#e74c3c', '#3498db']

    ax3.bar(labels, powers, color=colors_freq)
    ax3.set_ylabel('Power (ms²)')
    ax3.set_title(f'Frequency-Domain Power (LF/HF = {features.get("lf_hf_ratio", 0):.2f})')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Rule-Based Evaluation Summary
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.7, "Rule-Based Evaluation", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.5, f"Overall Pass Rate:", ha='center', va='center', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.5, 0.3, f"{pass_rate:.1%}" if pass_rate is not None else "N/A", ha='center', va='center', fontsize=18, fontweight='bold', color='darkgreen' if (pass_rate or 0) > 0.8 else 'darkred', transform=ax4.transAxes)
    ax4.axis('off') # Hide axes for text display
    ax4.set_title(evaluation_summary, fontsize=9, wrap=True) # Use evaluation_summary as title for the subplot

    plt.tight_layout()

    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)

    return img_buffer.getvalue()


def generate_report(
    ecg_data: dict,
    processed: dict,
    features: dict,
    output_path: Union[str, Path],
    pass_rate: Optional[float] = None,
    evaluation_summary: str = "N/A",
    include_ai_interpretation: bool = True
) -> str:
    """
    Generate a complete Markdown report with HRV analysis results.

    Args:
        ecg_data: Raw ECG data dictionary
        processed: Processed signal data
        features: HRV features dictionary
        output_path: Path for the output Markdown file (e.g., .md)
        pass_rate: Overall pass rate from rule-based evaluation.
        evaluation_summary: A text summary of the rule-based evaluation.
        include_ai_interpretation: Whether to include Claude-generated text

    Returns:
        str: Path to the generated report
    """
    output_path = Path(output_path).with_suffix('.md') # Ensure .md suffix
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate AI interpretation if requested
    if include_ai_interpretation:
        interpretation = generate_interpretation(
            features,
            pass_rate=pass_rate,
            evaluation_summary=evaluation_summary
        )
    else:
        interpretation = {
            "discussion": "AI interpretation not requested.",
            "conclusion": "Please review the HRV metrics above.",
        }

    # Generate visualizations (still generates PNG, will be embedded in MD)
    img_data = create_visualizations(
        ecg_data,
        processed,
        features,
        pass_rate=pass_rate,
        evaluation_summary=evaluation_summary
    )

    # Save visualization image alongside the MD report
    img_path = output_path.with_suffix('.png')
    with open(img_path, 'wb') as f:
        f.write(img_data)

    # Build Markdown report
    markdown_content = f"""# HRV Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. HRV Features Summary

| Metric      | Value           | Normal Range |
|-------------|-----------------|--------------|
| SDNN (ms)   | {features.get('sdnn', 0):.2f}    | 50-100       |
| RMSSD (ms)  | {features.get('rmssd', 0):.2f}   | 20-50        |
| pNN50 (%)   | {features.get('pnn50', 0):.2f}    | 10-25        |
| Mean HR (bpm)| {features.get('mean_hr', 0):.1f} | 60-100       |
| LF/HF Ratio | {features.get('lf_hf_ratio', 0):.2f} | 1.0-2.0      |

## 2. Evaluation Summary

**Overall Pass Rate:** {f"{pass_rate:.1%}" if pass_rate is not None else "N/A"}
**Summary:** {evaluation_summary}

## 3. Visualizations

![HRV Analysis Visualizations]({img_path.name})

## 4. Discussion

{interpretation['discussion']}

## 5. Conclusion

{interpretation['conclusion']}
"""
    # Write Markdown content to file
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(markdown_content)

    return str(output_path)
