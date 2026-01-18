# SPDX-License-Identifier: Apache-2.0
"""Report generation with AI-powered interpretation using Claude Opus 4.5."""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Optional imports for PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Optional import for Claude API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def generate_interpretation(
    features: dict,
    prediction: dict,
    model: str = "claude-opus-4-5-20251101"
) -> dict:
    """
    Use Claude Opus 4.5 to generate interpretive text for the report.

    Args:
        features: Dictionary of HRV features
        prediction: Dictionary of prediction results
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

    prompt = f"""Analyze these HRV metrics and stress classification results.
Write a Discussion and Conclusion section for a clinical HRV analysis report.

HRV Metrics:
- SDNN: {features.get('sdnn', 'N/A'):.2f} ms (normal range: 50-100 ms)
- RMSSD: {features.get('rmssd', 'N/A'):.2f} ms (normal range: 20-50 ms)
- pNN50: {features.get('pnn50', 'N/A'):.2f}% (normal range: 10-25%)
- Mean HR: {features.get('mean_hr', 'N/A'):.1f} bpm
- LF Power: {features.get('lf_power', 'N/A'):.2f} ms²
- HF Power: {features.get('hf_power', 'N/A'):.2f} ms²
- LF/HF Ratio: {features.get('lf_hf_ratio', 'N/A'):.2f} (normal range: 1.0-2.0)

Classification Result:
- Prediction: {prediction.get('prediction', 'N/A')}
- Confidence: {prediction.get('confidence', 0):.1%}
- Stress Probability: {prediction.get('stress_probability', 0):.1%}

Provide:
1. **Discussion** (2-3 paragraphs): Interpret the HRV metrics in clinical context.
   Explain what the values indicate about autonomic nervous system balance.
   Discuss the classification result and its implications.

2. **Conclusion** (1 paragraph): Summarize findings and provide recommendations
   for the individual (rest, stress management, follow-up, etc.).

Use professional medical report language. Be specific about the metrics.
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
    prediction: dict
) -> bytes:
    """
    Create visualization plots for the report.

    Args:
        ecg_data: Raw ECG data dictionary
        processed: Processed signal data
        features: HRV features
        prediction: Classification result

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

    # Plot 4: Classification Result
    ax4 = axes[1, 1]
    probs = [
        prediction.get('baseline_probability', 0.5),
        prediction.get('stress_probability', 0.5)
    ]
    labels_pred = ['Baseline', 'Stressed']
    colors_pred = ['#2ecc71', '#e74c3c']

    wedges, texts, autotexts = ax4.pie(
        probs,
        labels=labels_pred,
        colors=colors_pred,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05)
    )
    ax4.set_title(f'Classification: {prediction.get("prediction", "Unknown")}\n'
                  f'(Confidence: {prediction.get("confidence", 0):.1%})')

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
    prediction: dict,
    output_path: Union[str, Path],
    include_ai_interpretation: bool = True
) -> str:
    """
    Generate a complete PDF report with HRV analysis results.

    Args:
        ecg_data: Raw ECG data dictionary
        processed: Processed signal data
        features: HRV features dictionary
        prediction: Classification result dictionary
        output_path: Path for the output PDF
        include_ai_interpretation: Whether to include Claude-generated text

    Returns:
        str: Path to the generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate AI interpretation if requested
    if include_ai_interpretation:
        interpretation = generate_interpretation(features, prediction)
    else:
        interpretation = {
            "discussion": "AI interpretation not requested.",
            "conclusion": "Please review the HRV metrics above.",
        }

    # Generate visualizations
    img_data = create_visualizations(ecg_data, processed, features, prediction)

    if not REPORTLAB_AVAILABLE:
        # Fallback: save as text + image
        text_path = output_path.with_suffix('.txt')
        img_path = output_path.with_suffix('.png')

        with open(img_path, 'wb') as f:
            f.write(img_data)

        with open(text_path, 'w') as f:
            f.write("HRV ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("HRV FEATURES\n")
            f.write("-" * 30 + "\n")
            for key, value in features.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\nCLASSIFICATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Prediction: {prediction.get('prediction', 'N/A')}\n")
            f.write(f"Confidence: {prediction.get('confidence', 0):.1%}\n")
            f.write("\nDISCUSSION\n")
            f.write("-" * 30 + "\n")
            f.write(interpretation['discussion'] + "\n\n")
            f.write("CONCLUSION\n")
            f.write("-" * 30 + "\n")
            f.write(interpretation['conclusion'] + "\n")

        return str(text_path)

    # Generate PDF with ReportLab
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=1  # Center
    )
    heading_style = styles['Heading2']
    body_style = styles['Normal']

    story = []

    # Title
    story.append(Paragraph("HRV Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        body_style
    ))
    story.append(Spacer(1, 20))

    # Visualizations
    img_buffer = io.BytesIO(img_data)
    img = Image(img_buffer, width=6.5*inch, height=5.2*inch)
    story.append(img)
    story.append(Spacer(1, 20))

    # HRV Features Table
    story.append(Paragraph("HRV Features Summary", heading_style))
    table_data = [
        ['Metric', 'Value', 'Normal Range'],
        ['SDNN (ms)', f"{features.get('sdnn', 0):.2f}", '50-100'],
        ['RMSSD (ms)', f"{features.get('rmssd', 0):.2f}", '20-50'],
        ['pNN50 (%)', f"{features.get('pnn50', 0):.2f}", '10-25'],
        ['Mean HR (bpm)', f"{features.get('mean_hr', 0):.1f}", '60-100'],
        ['LF/HF Ratio', f"{features.get('lf_hf_ratio', 0):.2f}", '1.0-2.0'],
    ]
    table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Classification Result
    story.append(Paragraph("Classification Result", heading_style))
    story.append(Paragraph(
        f"<b>Prediction:</b> {prediction.get('prediction', 'N/A')}",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Confidence:</b> {prediction.get('confidence', 0):.1%}",
        body_style
    ))
    story.append(Spacer(1, 15))

    # Discussion
    story.append(Paragraph("Discussion", heading_style))
    story.append(Paragraph(interpretation['discussion'], body_style))
    story.append(Spacer(1, 15))

    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    story.append(Paragraph(interpretation['conclusion'], body_style))

    # Build PDF
    doc.build(story)

    return str(output_path)
