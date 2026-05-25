"""Confidence score interpretation helper.

Maps (verdict, confidence) pairs to human-readable interpretations.
"""

from typing import Dict, Tuple


# Confidence interpretation mappings for each verdict type
CONFIDENCE_INTERPRETATIONS: Dict[str, Dict[Tuple[int, int], str]] = {
    "SUPPORTS": {
        (9, 10): "Extremely confident the claim is true - overwhelming, consistent evidence from high-quality sources",
        (7, 8): "Highly confident the claim is true - strong, clear evidence with minimal contradictions",
        (5, 6): "Moderately confident the claim is true - good evidence but some limitations or minor contradictions",
        (3, 4): "Somewhat confident the claim is true - suggestive evidence but notable uncertainties",
        (1, 2): "Low confidence the claim is true - weak or very limited supporting evidence",
    },
    "REFUTES": {
        (9, 10): "Extremely confident the claim is false - overwhelming, consistent evidence contradicting it",
        (7, 8): "Highly confident the claim is false - strong, clear evidence refuting it",
        (5, 6): "Moderately confident the claim is false - good refuting evidence but some uncertainties",
        (3, 4): "Somewhat confident the claim is false - suggestive refuting evidence but notable gaps",
        (1, 2): "Low confidence the claim is false - weak or very limited refuting evidence",
    },
    "INSUFFICIENT_EVIDENCE": {
        (
            9,
            10,
        ): "Extremely confident evidence is insufficient - thoroughly searched, found genuinely mixed/inconclusive results",
        (7, 8): "Highly confident evidence is insufficient - good search yielded conflicting or unclear findings",
        (
            5,
            6,
        ): "Moderately confident evidence is insufficient - found some mixed evidence but search may be incomplete",
        (3, 4): "Somewhat confident evidence is insufficient - limited search or unclear if more evidence exists",
        (1, 2): "Low confidence evidence is insufficient - very limited search, likely missing relevant evidence",
    },
}


def get_confidence_interpretation(verdict: str, confidence: int) -> str:
    """Get human-readable interpretation of confidence score.

    Args:
        verdict: Verification verdict (SUPPORTS, REFUTES, INSUFFICIENT_EVIDENCE)
        confidence: Confidence score (1-10)

    Returns:
        Human-readable interpretation string

    Examples:
        >>> get_confidence_interpretation("SUPPORTS", 8)
        'Highly confident the claim is true - strong, clear evidence with minimal contradictions'

        >>> get_confidence_interpretation("REFUTES", 3)
        'Somewhat confident the claim is false - suggestive refuting evidence but notable gaps'

        >>> get_confidence_interpretation("INSUFFICIENT_EVIDENCE", 5)
        'Moderately confident evidence is insufficient - found some mixed evidence but search may be incomplete'
    """
    # Normalize verdict (handle case variations)
    verdict = verdict.upper().strip()

    # Handle unknown verdicts
    if verdict not in CONFIDENCE_INTERPRETATIONS:
        return f"Unknown verdict: {verdict}"

    # Get interpretation mapping for this verdict
    interpretations = CONFIDENCE_INTERPRETATIONS[verdict]

    # Find the appropriate range
    for (min_conf, max_conf), interpretation in interpretations.items():
        if min_conf <= confidence <= max_conf:
            return interpretation

    # Fallback for out-of-range confidence scores
    if confidence < 1:
        return f"Confidence score too low: {confidence} (expected 1-10)"
    elif confidence > 10:
        return f"Confidence score too high: {confidence} (expected 1-10)"
    else:
        # Should not reach here, but provide fallback
        return f"No interpretation available for confidence {confidence}"


def get_confidence_level(confidence: int) -> str:
    """Get confidence level name (for styling/display).

    Args:
        confidence: Confidence score (1-10)

    Returns:
        Level name: "extremely-high", "high", "moderate", "low", "very-low"
    """
    if 9 <= confidence <= 10:
        return "extremely-high"
    elif 7 <= confidence <= 8:
        return "high"
    elif 5 <= confidence <= 6:
        return "moderate"
    elif 3 <= confidence <= 4:
        return "low"
    elif 1 <= confidence <= 2:
        return "very-low"
    else:
        return "unknown"


if __name__ == "__main__":
    # Demo
    print("=== Confidence Interpretation Demo ===\n")

    test_cases = [
        ("SUPPORTS", 9),
        ("SUPPORTS", 7),
        ("SUPPORTS", 5),
        ("SUPPORTS", 3),
        ("SUPPORTS", 1),
        ("REFUTES", 8),
        ("REFUTES", 4),
        ("INSUFFICIENT_EVIDENCE", 6),
        ("INSUFFICIENT_EVIDENCE", 2),
    ]

    for verdict, confidence in test_cases:
        interpretation = get_confidence_interpretation(verdict, confidence)
        level = get_confidence_level(confidence)
        print(f"{verdict} @ {confidence}/10 [{level}]:")
        print(f"  {interpretation}\n")
