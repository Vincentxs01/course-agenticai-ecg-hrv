# SPDX-License-Identifier: Apache-2.0
"""Unit tests for report generator non-visual functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import report_generator
from src.tools.report_generator import (
    generate_interpretation,
    generate_report,
    REPORTLAB_AVAILABLE,
    ANTHROPIC_AVAILABLE,
)

@pytest.fixture(autouse=True)
def mock_matplotlib():
    """Mock matplotlib.pyplot to prevent GUI windows during tests."""
    mock_ax = MagicMock()
    # Configure mock_ax's methods to return mock_ax itself or appropriate mocks
    mock_ax.plot.return_value = MagicMock()
    mock_ax.scatter.return_value = MagicMock()
    mock_ax.bar.return_value = MagicMock()
    mock_ax.pie.return_value = (MagicMock(), MagicMock(), MagicMock()) # Returns 3 mocks for unpacking
    mock_ax.set_xlabel.return_value = MagicMock()
    mock_ax.set_ylabel.return_value = MagicMock()
    mock_ax.set_title.return_value = MagicMock()
    mock_ax.legend.return_value = MagicMock()
    mock_ax.grid.return_value = MagicMock()
    mock_ax.axhline.return_value = MagicMock()
    mock_ax.set_xticks.return_value = MagicMock()
    mock_ax.set_xticklabels.return_value = MagicMock()


    mock_fig = MagicMock()
    # Create a mock for the axes array that returns mock_ax when indexed
    mock_axes_array = MagicMock(spec=np.ndarray)
    mock_axes_array.__getitem__.side_effect = lambda idx: mock_ax
    mock_fig.subplots.return_value = (mock_fig, mock_axes_array) # Return fig and array-like mock for axes

    with patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_axes_array)) as mock_subplots, \
         patch('matplotlib.pyplot.savefig') as mock_savefig_patch, \
         patch('matplotlib.pyplot.close') as mock_close, \
         patch('matplotlib.pyplot.figure', return_value=mock_fig) as mock_figure, \
         patch('reportlab.lib.utils.ImageReader') as mock_reportlab_image_reader: # Patch reportlab.lib.utils.ImageReader

        # Configure the mock ImageReader to return a mock image object
        mock_image_instance = MagicMock()
        mock_image_instance.as_PIL.return_value = MagicMock(size=(1,1), mode='RGBA') # For split() to work
        mock_reportlab_image_reader.return_value = mock_image_instance

        def mock_savefig_side_effect(buffer_obj, *args, **kwargs):
            # Simulate writing some valid PNG data to the buffer
            buffer_obj.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\xda\xed\xc1\x01\x01\x00\x00\x00\xc2\xa0\xf7Om\x00\x00\x00\x00IEND\xaeB`\x82')
        mock_savefig_patch.side_effect = mock_savefig_side_effect
        yield
class TestAvailabilityConstants:
    """Tests for availability constants."""

    def test_reportlab_available_is_boolean(self):
        """Test that REPORTLAB_AVAILABLE is a boolean."""
        assert isinstance(REPORTLAB_AVAILABLE, bool)

    def test_anthropic_available_is_boolean(self):
        """Test that ANTHROPIC_AVAILABLE is a boolean."""
        assert isinstance(ANTHROPIC_AVAILABLE, bool)


class TestGenerateInterpretation:
    """Tests for generate_interpretation function."""

    def test_interpretation_without_anthropic(self):
        """Test interpretation when anthropic package is not available."""
        features = {"sdnn": 50.0, "rmssd": 30.0}
        pass_rate = 0.8
        evaluation_summary = "All good"

        with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
            result = generate_interpretation(features, pass_rate=pass_rate, evaluation_summary=evaluation_summary)

        assert "discussion" in result
        assert "conclusion" in result
        assert "anthropic package not installed" in result["discussion"]

    def test_interpretation_without_api_key(self):
        """Test interpretation when API key is not set."""
        features = {"sdnn": 50.0, "rmssd": 30.0}
        pass_rate = 0.8
        evaluation_summary = "All good"

        # Temporarily unset API key
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        try:
            with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', True):
                result = generate_interpretation(features, pass_rate=pass_rate, evaluation_summary=evaluation_summary)

            assert "discussion" in result
            assert "conclusion" in result
            assert "ANTHROPIC_API_KEY not set" in result["discussion"]
        finally:
            # Restore original key
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_interpretation_returns_dict(self):
        """Test that interpretation always returns a dict with required keys."""
        features = {"sdnn": 50.0, "rmssd": 30.0, "pnn50": 15.0, "mean_hr": 70.0}
        pass_rate = 0.8
        evaluation_summary = "All good"

        with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
            result = generate_interpretation(features, pass_rate=pass_rate, evaluation_summary=evaluation_summary)

        assert isinstance(result, dict)
        assert "discussion" in result
        assert "conclusion" in result
        assert isinstance(result["discussion"], str)
        assert isinstance(result["conclusion"], str)


    @pytest.fixture
    def sample_data(self):
        """Create sample data for report generation."""
        np.random.seed(42)
        signal = np.random.randn(5000)
        return {
            "ecg_data": {
                "signal": signal,
                "sampling_rate": 500,
                "duration_sec": 10.0,
            },
            "processed": {
                "filtered_signal": signal,
                "sampling_rate": 500,
                "r_peaks": np.array([100, 600, 1100, 1600, 2100]),
                "rr_intervals": np.array([1000, 1000, 1000, 1000]),
            },
            "features": {
                "sdnn": 45.5,
                "rmssd": 32.1,
                "pnn50": 12.3,
                "mean_hr": 72.5,
                "lf_power": 1200.0,
                "hf_power": 800.0,
                "lf_hf_ratio": 1.5,
            },
            "pass_rate": 0.85,
            "evaluation_summary": "All baseline criteria passed.",
        }

    def test_text_fallback_creates_files(self, sample_data):
        """Test that text fallback creates .md and .png files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            with patch.object(report_generator, 'REPORTLAB_AVAILABLE', False):
                with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
                    result = generate_report(
                        sample_data["ecg_data"],
                        sample_data["processed"],
                        sample_data["features"],
                        output_path,
                        pass_rate=sample_data["pass_rate"],
                        evaluation_summary=sample_data["evaluation_summary"],
                        include_ai_interpretation=False,
                    )

            # Should return path to .md file
            assert result.endswith('.md')
            assert Path(result).exists()

            # Check .png file was also created
            png_path = Path(result).with_suffix('.png')
            assert png_path.exists()

    def test_text_fallback_content(self, sample_data):
        """Test that text fallback report contains expected sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            with patch.object(report_generator, 'REPORTLAB_AVAILABLE', False):
                with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
                    result = generate_report(
                        sample_data["ecg_data"],
                        sample_data["processed"],
                        sample_data["features"],
                        output_path,
                        pass_rate=sample_data["pass_rate"],
                        evaluation_summary=sample_data["evaluation_summary"],
                        include_ai_interpretation=False,
                    )

            with open(result, 'r') as f:
                content = f.read()

            assert "# HRV Analysis Report" in content
            assert "## 1. HRV Features Summary" in content
            assert "## 4. Discussion" in content
            assert "## 5. Conclusion" in content
            assert "| SDNN (ms)   | 45.50    | 50-100       |" in content # Assert content of the table
            assert "Overall Pass Rate:" in content

    def test_text_fallback_feature_values(self, sample_data):
        """Test that text fallback contains actual feature values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            with patch.object(report_generator, 'REPORTLAB_AVAILABLE', False):
                with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
                    result = generate_report(
                        sample_data["ecg_data"],
                        sample_data["processed"],
                        sample_data["features"],
                        output_path,
                        pass_rate=sample_data["pass_rate"],
                        evaluation_summary=sample_data["evaluation_summary"],
                        include_ai_interpretation=False,
                    )

            with open(result, 'r') as f:
                content = f.read()

            # Check specific values are in the report
            assert "45.50" in content  # sdnn
            assert "32.10" in content  # rmssd
            assert "85.0%" in content  # pass_rate (confidence changed to pass_rate)


class TestGenerateReportWithoutAI:
    """Tests for report generation without AI interpretation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for report generation."""
        np.random.seed(42)
        signal = np.random.randn(5000)
        return {
            "ecg_data": {
                "signal": signal,
                "sampling_rate": 500,
                "duration_sec": 10.0,
            },
            "processed": {
                "filtered_signal": signal,
                "sampling_rate": 500,
                "r_peaks": np.array([100, 600, 1100, 1600, 2100]),
                "rr_intervals": np.array([1000, 1000, 1000, 1000]),
            },
            "features": {
                "sdnn": 45.5,
                "rmssd": 32.1,
                "pnn50": 12.3,
                "mean_hr": 72.5,
                "lf_power": 1200.0,
                "hf_power": 800.0,
                "lf_hf_ratio": 1.5,
            },
            "pass_rate": 0.75,
            "evaluation_summary": "Some windows failed baseline criteria.",
        }

    def test_report_without_ai_interpretation(self, sample_data):
        """Test report generation with AI interpretation disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            with patch.object(report_generator, 'REPORTLAB_AVAILABLE', False):
                result = generate_report(
                    sample_data["ecg_data"],
                    sample_data["processed"],
                    sample_data["features"],
                    output_path,
                    pass_rate=sample_data["pass_rate"],
                    evaluation_summary=sample_data["evaluation_summary"],
                    include_ai_interpretation=False,
                )

            with open(result, 'r') as f:
                content = f.read()

            assert "AI interpretation not requested" in content

    def test_report_creates_parent_directory(self, sample_data):
        """Test that report generation creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "deep" / "report.pdf"

            with patch.object(report_generator, 'REPORTLAB_AVAILABLE', False):
                with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
                    result = generate_report(
                        sample_data["ecg_data"],
                        sample_data["processed"],
                        sample_data["features"],
                        output_path,
                        pass_rate=sample_data["pass_rate"],
                        evaluation_summary=sample_data["evaluation_summary"],
                        include_ai_interpretation=False,
                    )

            assert Path(result).exists()
            assert Path(result).parent.name == "deep"


class TestGenerateReportPDF:
    """Tests for PDF report generation when reportlab is available."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for report generation."""
        np.random.seed(42)
        signal = np.random.randn(5000)
        return {
            "ecg_data": {
                "signal": signal,
                "sampling_rate": 500,
                "duration_sec": 10.0,
            },
            "processed": {
                "filtered_signal": signal,
                "sampling_rate": 500,
                "r_peaks": np.array([100, 600, 1100, 1600, 2100]),
                "rr_intervals": np.array([1000, 1000, 1000, 1000]),
            },
            "features": {
                "sdnn": 45.5,
                "rmssd": 32.1,
                "pnn50": 12.3,
                "mean_hr": 72.5,
                "lf_power": 1200.0,
                "hf_power": 800.0,
                "lf_hf_ratio": 1.5,
            },
            "pass_rate": 0.85,
            "evaluation_summary": "All baseline criteria passed.",
        }

    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")
    @patch('reportlab.lib.utils.ImageReader')
    def test_pdf_generation(self, mock_image_reader, sample_data):
        """Test PDF generation when reportlab is available."""
        # Configure the mock ImageReader to return a mock image object
        mock_image_instance = MagicMock()
        mock_image_instance.as_PIL.return_value = MagicMock(size=(1,1), mode='RGBA') # For split() to work
        mock_image_instance.getSize.return_value = (1, 1) # Return dummy size for reportlab's img.getSize() call
        mock_image_reader.return_value = mock_image_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            with patch.object(report_generator, 'ANTHROPIC_AVAILABLE', False):
                result = generate_report(
                    sample_data["ecg_data"],
                    sample_data["processed"],
                    sample_data["features"],
                    output_path,
                    pass_rate=sample_data["pass_rate"],
                    evaluation_summary=sample_data["evaluation_summary"],
                    include_ai_interpretation=False,
                )

            assert result.endswith('.pdf')
            assert Path(result).exists()
            # PDF should have some content
            assert Path(result).stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
