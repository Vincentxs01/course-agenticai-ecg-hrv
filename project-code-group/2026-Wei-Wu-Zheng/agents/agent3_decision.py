"""
Agent 3: Decision Agent (Rule-based + MCP Tool Integration)
Responsible for comprehensive analysis and decision making
"""

import sys
import os

# Ensure tools can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from tools.mcp_tools import MCPTools
except ImportError:
    from mcp_tools import MCPTools

class DecisionAgent:
    """Agent 3: Smart decision making using rule engine with MCP tool integration"""

    def __init__(self):
        self.name = "Decision Agent (Rule-based + MCP)"
        self.status = "Standby"
        self.tools = MCPTools()
        self.decision_log = []
        self.baseline = None

    def set_baseline(self, features):
        """
        Set personalized baseline

        Establish baseline during first 10-15 minutes of driving

        Args:
            features: Feature dictionary
        """
        self.baseline = {
            "heart_rate": features['heart_rate'],
            "hrv_sdnn": features['hrv_sdnn'],
            "hrv_rmssd": features['hrv_rmssd']
        }
        self.decision_log.append(f"[OK] Baseline established: HR={self.baseline['heart_rate']} bpm, SDNN={self.baseline['hrv_sdnn']} ms")

    def analyze(self, features, driving_duration_minutes=0):
        """
        Analyze physiological features and make decisions

        Decision logic:
        1. Query MCP tools for contextual information
        2. Calculate multi-factor risk score
        3. Determine risk level based on total score
        4. Generate analysis report

        Args:
            features: dict, feature data from Agent 2
            driving_duration_minutes: Driving duration (minutes)

        Returns:
            decision: dict containing risk assessment and recommendations
        """
        self.status = "Analyzing..."
        self.decision_log = []

        # Initialize risk score
        risk_score = 0
        reasons = []
        tool_results = {}

        # === 1. Query MCP Tools ===
        self.decision_log.append("--- Querying Context Information ---")

        # 1.1 Query weather
        weather = self.tools.get_weather()
        tool_results['weather'] = weather
        self.decision_log.append(f"Weather: {weather['description']}")

        if weather['fatigue_factor'] == "High":
            risk_score += 15
            reasons.append(f"Weather factor: {weather['condition']}, {weather['temperature']}C, hot environment increases fatigue")
        elif weather['fatigue_factor'] == "Medium":
            risk_score += 5
            reasons.append(f"Weather factor: {weather['condition']}, slight impact")

        # 1.2 Query time risk
        time_risk = self.tools.get_time_risk()
        tool_results['time_risk'] = time_risk
        self.decision_log.append(f"Time: {time_risk['current_time']} ({time_risk['risk_level']})")

        risk_score += time_risk['risk_score']
        if time_risk['risk_score'] > 0:
            reasons.append(f"Time factor: {time_risk['reason']}")

        # 1.3 Query driving duration risk
        if driving_duration_minutes > 0:
            duration_risk = self.tools.get_driving_duration_risk(driving_duration_minutes)
            tool_results['duration_risk'] = duration_risk
            self.decision_log.append(f"Driving duration: {duration_risk['duration_hours']:.1f} hours ({duration_risk['risk_level']})")

            risk_score += duration_risk['risk_score']
            if duration_risk['risk_score'] > 0:
                reasons.append(f"Driving duration: {duration_risk['reason']}")

        # === 2. Analyze Physiological Metrics ===
        self.decision_log.append("\n--- Analyzing Physiological Metrics ---")

        hr = features['heart_rate']
        sdnn = features['hrv_sdnn']
        rmssd = features['hrv_rmssd']

        self.decision_log.append(f"Heart Rate: {hr} bpm")
        self.decision_log.append(f"HRV (SDNN): {sdnn} ms")
        self.decision_log.append(f"HRV (RMSSD): {rmssd} ms")

        # 2.1 Heart rate assessment
        if hr < 60:
            risk_score += 30
            reasons.append(f"Low heart rate ({hr} bpm), may indicate relaxation or fatigue")
            # Query medical knowledge
            med_info = self.tools.get_medical_info("low_hr")
            self.decision_log.append(f"   Medical info: {med_info['meaning']}")
        elif hr > 90:
            risk_score -= 10  # High HR may indicate alertness
            self.decision_log.append("   [OK] Heart rate slightly elevated, maintaining alertness")

        # 2.2 HRV assessment
        if sdnn > 80:
            risk_score += 35
            reasons.append(f"High HRV (SDNN={sdnn} ms), parasympathetic active, fatigue sign")
            # Query medical knowledge
            med_info = self.tools.get_medical_info("high_hrv")
            self.decision_log.append(f"   Medical info: {med_info['meaning']}")
        elif sdnn < 30:
            risk_score -= 5
            self.decision_log.append("   [OK] Low HRV, may indicate alert state")

        # 2.3 Compare with baseline (if set)
        if self.baseline:
            self.decision_log.append("\n--- Comparing with Personal Baseline ---")

            hr_diff = hr - self.baseline['heart_rate']
            sdnn_diff = sdnn - self.baseline['hrv_sdnn']

            if hr_diff < -10:
                risk_score += 15
                reasons.append(f"Heart rate {abs(hr_diff):.1f} bpm below baseline, deviation from personal normal")
                self.decision_log.append(f"   [WARN] Heart rate {abs(hr_diff):.1f} bpm below baseline")

            if sdnn_diff > 20:
                risk_score += 15
                reasons.append(f"HRV {sdnn_diff:.1f} ms above baseline, increased variability")
                self.decision_log.append(f"   [WARN] HRV {sdnn_diff:.1f} ms above baseline")

        # === 3. Determine Risk Level ===
        self.decision_log.append(f"\n--- Risk Score: {risk_score} ---")

        if risk_score >= 70:
            risk_level = "Very High"
            alert_needed = True
        elif risk_score >= 50:
            risk_level = "High"
            alert_needed = True
        elif risk_score >= 30:
            risk_level = "Medium"
            alert_needed = False
        else:
            risk_level = "Low"
            alert_needed = False

        # === 4. Generate Analysis Report ===
        analysis = self._generate_report(
            features,
            tool_results,
            reasons,
            risk_level,
            risk_score
        )

        self.status = "Done"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "alert_needed": alert_needed,
            "analysis": analysis,
            "reasons": reasons,
            "tool_results": tool_results
        }

    def _generate_report(self, features, tool_results, reasons, risk_level, risk_score):
        """
        Generate detailed analysis report (AI-style)

        Args:
            features: Physiological features
            tool_results: Tool query results
            reasons: Risk factor list
            risk_level: Risk level
            risk_score: Risk score

        Returns:
            str: Analysis report
        """
        report = "## Multi-Modal Analysis Report\n\n"

        # 1. Physiological metrics assessment
        report += "### 1. Physiological Metrics Assessment\n\n"
        report += f"- **Heart Rate**: {features['heart_rate']} bpm\n"
        report += f"- **HRV (SDNN)**: {features['hrv_sdnn']} ms\n"
        report += f"- **HRV (RMSSD)**: {features['hrv_rmssd']} ms\n"
        report += f"- **Detected Beats**: {features['num_beats']}\n\n"

        # 2. Environmental context analysis
        report += "### 2. Environmental Context Analysis\n\n"

        if 'weather' in tool_results:
            weather = tool_results['weather']
            report += f"**Weather Condition**: {weather['description']}\n"
            report += f"- Fatigue Impact Factor: {weather['fatigue_factor']}\n\n"

        if 'time_risk' in tool_results:
            time_risk = tool_results['time_risk']
            report += f"**Time Risk**: {time_risk['current_time']}\n"
            report += f"- Risk Level: {time_risk['risk_level']}\n"
            report += f"- Description: {time_risk['reason']}\n\n"

        if 'duration_risk' in tool_results:
            duration_risk = tool_results['duration_risk']
            report += f"**Driving Duration**: {duration_risk['duration_hours']:.1f} hours\n"
            report += f"- Risk Level: {duration_risk['risk_level']}\n"
            report += f"- Description: {duration_risk['reason']}\n\n"

        # 3. Risk factor identification
        report += "### 3. Risk Factor Identification\n\n"
        if reasons:
            for reason in reasons:
                report += f"- {reason}\n"
        else:
            report += "- No significant risk factors identified\n"

        report += "\n"

        # 4. Comprehensive assessment
        report += f"### 4. Comprehensive Assessment\n\n"
        report += f"**Risk Score**: {risk_score} / 100\n\n"
        report += f"**Risk Level**: {risk_level}\n\n"

        # 5. Recommended actions
        report += "### 5. Recommended Actions\n\n"

        if risk_level == "Very High":
            report += "**IMMEDIATE ACTION REQUIRED!**\n\n"
            report += "1. Find a safe place to stop immediately\n"
            report += "2. Rest for at least 20-30 minutes\n"
            report += "3. Get out and stretch your body\n"
            report += "4. Stay hydrated, avoid excessive caffeine\n"
            report += "5. If fatigue persists, find a place to sleep\n\n"

            # Provide nearest rest areas
            if 'weather' in tool_results:
                rest_areas = self.tools.get_rest_area()
                report += "**Nearest Rest Areas**:\n"
                for area in rest_areas[:2]:
                    report += f"- {area['name']} ({area['distance']}, approx. {area['eta']})\n"

        elif risk_level == "High":
            report += "**ATTENTION NEEDED!**\n\n"
            report += "1. Stay highly alert\n"
            report += "2. Consider stopping at the next rest area\n"
            report += "3. Stay hydrated\n"
            report += "4. Adjust car environment (music, temperature, ventilation)\n"
            report += "5. If traveling with others, consider switching drivers\n"

        elif risk_level == "Medium":
            report += "**STAY ALERT**\n\n"
            report += "1. Pay attention to road conditions, avoid distractions\n"
            report += "2. If driving over 2 hours, consider taking a break\n"
            report += "3. Ensure comfortable car temperature\n"

        else:
            report += "**CURRENT STATUS: GOOD**\n\n"
            report += "Continue safe driving. Recommended to take a break every 2 hours.\n"

        return report

    def get_log(self):
        """Get decision log"""
        return self.decision_log

# Test program
if __name__ == "__main__":
    print("=" * 50)
    print("Agent 3 Test - Decision Agent (Rule-based + MCP)")
    print("=" * 50)
    print()

    # Create test features (simulating drowsy state)
    test_features_drowsy = {
        "heart_rate": 58,
        "hrv_sdnn": 85,
        "hrv_rmssd": 45,
        "num_beats": 120,
        "mean_rr": 1034,
        "min_hr": 55,
        "max_hr": 62
    }

    # Create test features (simulating normal state)
    test_features_normal = {
        "heart_rate": 75,
        "hrv_sdnn": 50,
        "hrv_rmssd": 30,
        "num_beats": 150,
        "mean_rr": 800,
        "min_hr": 70,
        "max_hr": 80
    }

    # Test Agent
    agent3 = DecisionAgent()

    print("=== Test Case 1: Drowsy State ===\n")
    decision_drowsy = agent3.analyze(test_features_drowsy, driving_duration_minutes=150)

    print("Decision log:")
    for log in agent3.get_log():
        print(f"  {log}")

    print(f"\nRisk Level: {decision_drowsy['risk_level']}")
    print(f"Risk Score: {decision_drowsy['risk_score']}")
    print(f"Alert Needed: {'Yes' if decision_drowsy['alert_needed'] else 'No'}")

    print("\n" + "=" * 50)
    print("\n=== Test Case 2: Normal State ===\n")

    agent3_normal = DecisionAgent()
    decision_normal = agent3_normal.analyze(test_features_normal, driving_duration_minutes=30)

    print(f"Risk Level: {decision_normal['risk_level']}")
    print(f"Risk Score: {decision_normal['risk_score']}")
    print(f"Alert Needed: {'Yes' if decision_normal['alert_needed'] else 'No'}")

    print("\n" + "=" * 50)
    print("[OK] Agent 3 test complete!")
    print("=" * 50)
