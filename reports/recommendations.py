def get_recommendation(flood_percentage):

    if flood_percentage < 5:

        severity = "LOW"

        recommendation = [

            "No immediate action required.",

            "Continue monitoring.",

            "Inspect drainage systems."

        ]

    elif flood_percentage < 20:

        severity = "MODERATE"

        recommendation = [

            "Inspect vulnerable roads.",

            "Prepare emergency teams.",

            "Monitor water levels."

        ]

    elif flood_percentage < 40:

        severity = "HIGH"

        recommendation = [

            "Deploy emergency response.",

            "Inspect bridges.",

            "Warn nearby communities.",

            "Protect critical infrastructure."

        ]

    else:

        severity = "SEVERE"

        recommendation = [

            "Immediate evacuation.",

            "Deploy rescue teams.",

            "Close affected roads.",

            "Establish emergency shelters.",

            "Monitor continuously."

        ]

    return severity, recommendation

if __name__ == "__main__":

    percentages = [2, 12, 27, 55]

    for p in percentages:

        severity, rec = get_recommendation(p)

        print("=" * 40)
        print(f"Flood Coverage : {p}%")
        print(f"Severity       : {severity}")
        print()

        for r in rec:
            print(f"• {r}")

        print()