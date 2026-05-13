RUBRIC = {
    "feasibility": {
        1: "Unrealistic or impossible with current resources/technology.",
        2: "Very hard to implement; major gaps in practicality.",
        3: "Possible in principle, but incomplete or underspecified.",
        4: "Reasonably implementable with known tools and effort.",
        5: "Highly practical and clearly implementable."
    },
    "technical_correctness": {
        1: "Technically incorrect or contradictory.",
        2: "Contains major technical issues or weak logic.",
        3: "Broadly plausible, but some technical uncertainty remains.",
        4: "Mostly technically sound and logically valid.",
        5: "Strong technical logic and clear implementation sense."
    },
    "business_viability": {
        1: "No clear users, value, or market relevance.",
        2: "Weak market fit or unclear user benefit.",
        3: "Some practical value, but business case is limited.",
        4: "Good user value and realistic adoption potential.",
        5: "Strong value proposition and clear business potential."
    }
}


def explain_score(metric: str, score: int) -> str:
    score = int(round(score))
    score = max(1, min(5, score))
    return RUBRIC[metric][score]