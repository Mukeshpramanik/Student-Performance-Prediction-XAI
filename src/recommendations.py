"""
Rule-based academic intervention suggestions and supportive recommendations.
"""

from typing import List, Dict, Any


def generate_recommendations(student_data: Dict[str, Any], predicted_grade: float, at_risk_prob: float) -> List[Dict[str, str]]:
    """
    Generate transparent, non-judgmental academic support recommendations based on empirical indicators.
    All suggestions are framed supportively as advisory insights for educators and mentors.
    """
    recommendations = []

    # 1. Attendance & Absenteeism Review
    absences = student_data.get("absences", 0)
    if absences >= 10:
        recommendations.append({
            "category": "Attendance & Engagement",
            "priority": "High",
            "observation": f"Elevated absence count recorded ({absences} days).",
            "suggestion": "Review underlying attendance drivers with the academic mentor. Consider structured catch-up sessions or recorded course review materials to close knowledge gaps."
        })
    elif absences >= 5:
        recommendations.append({
            "category": "Attendance & Engagement",
            "priority": "Moderate",
            "observation": f"Moderate absence count observed ({absences} days).",
            "suggestion": "Encourage regular class presence and verify whether any lecture notes or seminar exercises were missed during past absences."
        })

    # 2. Independent Study Time Allocation
    studytime = student_data.get("studytime", 2)
    if studytime == 1:  # < 2 hours per week
        recommendations.append({
            "category": "Study Habits & Routine",
            "priority": "High",
            "observation": "Weekly independent study time is currently under 2 hours.",
            "suggestion": "Help the student establish a structured weekly study timetable targeting 4 to 6 hours of focused, distributed practice rather than cramming."
        })
    elif studytime == 2:  # 2 to 5 hours per week
        recommendations.append({
            "category": "Study Habits & Routine",
            "priority": "Moderate",
            "observation": "Study time is between 2 and 5 hours weekly.",
            "suggestion": "Consider incrementally increasing self-directed problem-solving practice by 1 to 2 hours per week, particularly on foundational mathematical topics."
        })

    # 3. Prior Academic Failure History
    failures = student_data.get("failures", 0)
    if failures >= 1:
        recommendations.append({
            "category": "Curriculum Foundation",
            "priority": "High",
            "observation": f"History indicates {failures} prior course failure(s).",
            "suggestion": "Conduct a diagnostic evaluation on core prerequisite concepts. Coordinate one-on-one peer tutoring or faculty office hour sessions to rebuild foundational confidence."
        })

    # 4. Interim Examination Performance (Mid-term)
    g1 = student_data.get("G1", None)
    g2 = student_data.get("G2", None)
    if g1 is not None and g1 < 10:
        recommendations.append({
            "category": "Assessment Milestone",
            "priority": "High",
            "observation": f"First evaluation score (G1 = {g1}/20) is below passing threshold.",
            "suggestion": "Offer targeted remediation on the specific modules covered in the first period examination before upcoming assessments compound difficulty."
        })
    if g2 is not None and g2 < 10:
        recommendations.append({
            "category": "Assessment Milestone",
            "priority": "High",
            "observation": f"Second evaluation score (G2 = {g2}/20) is below passing threshold.",
            "suggestion": "Initiate focused revision clinics and practice mock problem sets to reinforce comprehension ahead of final examinations."
        })

    # 5. Institutional Support Services
    schoolsup = str(student_data.get("schoolsup", "no")).lower()
    if schoolsup == "no" and (predicted_grade < 10 or at_risk_prob > 0.4):
        recommendations.append({
            "category": "Institutional Support",
            "priority": "Moderate",
            "observation": "Student is not currently enrolled in school-based extra educational support.",
            "suggestion": "Explore enrollment in department-sponsored academic tutoring clinics or remedial workshop programs."
        })

    # 6. Overall Positive Encouragement if Performing Well
    if not recommendations or (predicted_grade >= 14 and at_risk_prob < 0.2):
        recommendations.append({
            "category": "Academic Enrichment",
            "priority": "Low",
            "observation": "Indicators demonstrate solid academic engagement and steady progress.",
            "suggestion": "Maintain continuous study discipline and encourage participation in advanced problem-solving challenges, project work, or peer mentoring roles."
        })

    return recommendations
