"""
MCP Tools Collection
Provides multi-modal context query functions
"""

from datetime import datetime
import random

class MCPTools:
    """Model Context Protocol Tool Set"""

    @staticmethod
    def get_weather(location="Tainan"):
        """
        Query weather conditions (affects driving fatigue)

        In real applications, this would connect to a weather API
        Currently uses simulated data

        Args:
            location: Location name

        Returns:
            dict: Weather information
        """
        # Simulate different weather conditions
        conditions = [
            {"condition": "Sunny", "temp": 28, "humidity": 65, "fatigue_factor": "Medium"},
            {"condition": "Cloudy", "temp": 25, "humidity": 70, "fatigue_factor": "Low"},
            {"condition": "Hot & Humid", "temp": 33, "humidity": 85, "fatigue_factor": "High"},
            {"condition": "Cool", "temp": 22, "humidity": 60, "fatigue_factor": "Low"},
        ]

        # Randomly select one (in real applications, this would be an actual query)
        weather = random.choice(conditions)

        return {
            "location": location,
            "temperature": weather["temp"],
            "condition": weather["condition"],
            "humidity": weather["humidity"],
            "fatigue_factor": weather["fatigue_factor"],
            "description": f"{location} currently {weather['condition']}, {weather['temp']}C, humidity {weather['humidity']}%"
        }

    @staticmethod
    def get_time_risk():
        """
        Assess fatigue risk based on current time

        Late night (02:00-05:00) is highest risk
        Afternoon (14:00-16:00) is second highest

        Returns:
            dict: Time risk information
        """
        current_hour = datetime.now().hour
        current_time = datetime.now().strftime("%H:%M")

        # Determine risk based on time
        if 2 <= current_hour <= 5:
            risk = "Very High"
            reason = "Late night period (02:00-05:00), circadian low point, prone to fatigue"
            risk_score = 40
        elif 23 <= current_hour or current_hour <= 1:
            risk = "High"
            reason = "Late night period (23:00-01:00), recommended to rest"
            risk_score = 30
        elif 14 <= current_hour <= 16:
            risk = "High"
            reason = "Afternoon period (14:00-16:00), prone to fatigue"
            risk_score = 25
        elif 22 <= current_hour:
            risk = "Medium"
            reason = "Evening period, stay alert"
            risk_score = 15
        elif 12 <= current_hour <= 13:
            risk = "Medium"
            reason = "Post-lunch period, maintain focus"
            risk_score = 10
        else:
            risk = "Normal"
            reason = "Normal daytime period"
            risk_score = 0

        return {
            "current_time": current_time,
            "hour": current_hour,
            "risk_level": risk,
            "reason": reason,
            "risk_score": risk_score
        }

    @staticmethod
    def get_rest_area(location="Tainan"):
        """
        Query nearest rest areas

        In real applications, this would connect to Google Maps API
        Currently uses simulated data

        Args:
            location: Current location

        Returns:
            list: Rest area information list
        """
        # Simulated data
        rest_areas = [
            {
                "name": "Rende Service Area",
                "distance": "5.2 km",
                "eta": "7 min",
                "facilities": ["Restaurant", "Gas Station", "Rest Room", "Restroom"]
            },
            {
                "name": "Xinhua Rest Stop",
                "distance": "12.8 km",
                "eta": "15 min",
                "facilities": ["Convenience Store", "Restroom", "Parking"]
            },
            {
                "name": "Shanhua Service Area",
                "distance": "18.5 km",
                "eta": "22 min",
                "facilities": ["Restaurant", "Convenience Store", "Gas Station", "Rest Room"]
            }
        ]

        return rest_areas

    @staticmethod
    def get_medical_info(symptom):
        """
        Query medical knowledge base

        Provides medical information about HRV and fatigue

        Args:
            symptom: Symptom or metric keyword

        Returns:
            dict: Medical information
        """
        medical_db = {
            "high_hrv": {
                "description": "High HRV usually indicates parasympathetic nervous system activation",
                "meaning": "In driving context, may indicate body entering relaxation or fatigue state",
                "note": "Should be combined with other metrics (like heart rate) for comprehensive assessment"
            },
            "low_hr": {
                "description": "Decreased heart rate may indicate body entering rest mode",
                "meaning": "If accompanied by elevated HRV, likely a sign of fatigue",
                "note": "Normal resting heart rate varies by individual, requires personalized baseline"
            },
            "hrv_increase": {
                "description": "HRV increase indicates greater heart rate interval variability",
                "meaning": "Parasympathetic nervous system (relaxation system) becoming dominant",
                "note": "This is a danger signal while driving, indicates decreased alertness"
            },
            "baseline_deviation": {
                "description": "Deviation from personal baseline",
                "meaning": "Physiological state has changed",
                "note": "Even if values are within normal range, deviation from baseline may be significant"
            }
        }

        info = medical_db.get(symptom, {
            "description": "No related information",
            "meaning": "Please query other keywords",
            "note": ""
        })

        return info

    @staticmethod
    def get_driving_duration_risk(duration_minutes):
        """
        Assess risk based on driving duration

        Args:
            duration_minutes: Driving duration in minutes

        Returns:
            dict: Duration risk information
        """
        if duration_minutes < 60:
            risk = "Low"
            reason = "Short driving duration"
            risk_score = 0
        elif 60 <= duration_minutes < 120:
            risk = "Medium"
            reason = "Driving over 1 hour, consider taking a break"
            risk_score = 10
        elif 120 <= duration_minutes < 180:
            risk = "High"
            reason = "Driving over 2 hours, strongly recommended to rest"
            risk_score = 25
        else:
            risk = "Very High"
            reason = "Driving over 3 hours, must rest!"
            risk_score = 40

        return {
            "duration_minutes": duration_minutes,
            "duration_hours": duration_minutes / 60,
            "risk_level": risk,
            "reason": reason,
            "risk_score": risk_score
        }

# Test program
if __name__ == "__main__":
    print("=" * 50)
    print("MCP Tools Test")
    print("=" * 50)
    print()

    # Test weather query
    print("1. Weather Query:")
    weather = MCPTools.get_weather("Tainan")
    print(f"   {weather['description']}")
    print(f"   Fatigue Impact Factor: {weather['fatigue_factor']}")
    print()

    # Test time risk
    print("2. Time Risk Assessment:")
    time_risk = MCPTools.get_time_risk()
    print(f"   Current Time: {time_risk['current_time']}")
    print(f"   Risk Level: {time_risk['risk_level']}")
    print(f"   Reason: {time_risk['reason']}")
    print()

    # Test rest area query
    print("3. Nearest Rest Areas:")
    rest_areas = MCPTools.get_rest_area()
    for area in rest_areas[:2]:
        print(f"   {area['name']} - {area['distance']} ({area['eta']})")
    print()

    # Test medical knowledge base
    print("4. Medical Knowledge Query:")
    info = MCPTools.get_medical_info("high_hrv")
    print(f"   {info['description']}")
    print(f"   Meaning: {info['meaning']}")
    print()

    print("=" * 50)
    print("[OK] All tools test complete!")
    print("=" * 50)
