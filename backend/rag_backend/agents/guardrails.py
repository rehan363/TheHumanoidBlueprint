"""
Guardrails for RAG Agent System

Input guardrails run BEFORE agent execution (blocking mode)
Output guardrails run AFTER agent execution (parallel mode)
"""

import re
import logging
from typing import Optional
from agents import input_guardrail, output_guardrail, GuardrailFunctionOutput

logger = logging.getLogger(__name__)


# ===== INPUT GUARDRAILS (Blocking Mode) =====

@input_guardrail
async def check_relevance(ctx, agent, input: str) -> GuardrailFunctionOutput:
    """
    Check if query is relevant to Physical AI/Robotics topics.

    Blocking guardrail that prevents wasted API calls on off-topic queries.
    """
    logger.info(f"check_relevance: Checking query: '{input[:50]}...'")

    # List of relevant keywords
    robotics_keywords = [
        "robot", "ros", "ai", "sensor", "actuator", "navigation",
        "perception", "control", "physical", "autonomous", "dds",
        "middleware", "lidar", "camera", "imu", "motor", "servo",
        "path planning", "slam", "localization", "mapping", "vision",
        "kinematics", "dynamics", "manipulation", "gripper", "arm",
        "mobile", "humanoid", "week", "chapter", "textbook"
    ]

    # Quick keyword check
    input_lower = input.lower()
    if any(keyword in input_lower for keyword in robotics_keywords):
        logger.info("Relevance check: PASSED (keywords found)")
        return GuardrailFunctionOutput(
            output_info={"relevant": True, "reason": "Contains robotics keywords"},
            tripwire_triggered=False
        )

    # Check for common off-topic patterns
    off_topic_patterns = [
        r"\b(weather|stock|movie|music|recipe|sports|celebrity)\b",
        r"\bwhat.*(time|date|day)\b",
        r"\bhow.*(cook|bake|recipe)\b"
    ]

    for pattern in off_topic_patterns:
        if re.search(pattern, input_lower):
            logger.warning(f"Relevance check: FAILED (off-topic pattern: {pattern})")
            return GuardrailFunctionOutput(
                output_info={
                    "relevant": False,
                    "reason": "Off-topic query detected",
                    "detected_pattern": pattern
                },
                tripwire_triggered=True
            )

    # If no clear signals, allow (benefit of the doubt)
    logger.info("Relevance check: PASSED (no off-topic signals)")
    return GuardrailFunctionOutput(
        output_info={"relevant": True, "reason": "No off-topic signals detected"},
        tripwire_triggered=False
    )


@input_guardrail
async def check_language(ctx, agent, input: str) -> GuardrailFunctionOutput:
    """
    Ensure query is in English (textbook is English-only).

    Blocking guardrail for language validation.
    """
    logger.info(f"check_language: Checking query language")

    # Simple heuristic: check for common non-English characters
    # This is basic; a real implementation would use langdetect library

    # Check for non-Latin characters (except common punctuation)
    non_latin_pattern = r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F]+'

    if re.search(non_latin_pattern, input):
        logger.warning("Language check: FAILED (non-Latin characters detected)")
        return GuardrailFunctionOutput(
            output_info={"language": "non-english"},
            tripwire_triggered=True
        )

    logger.info("Language check: PASSED")
    return GuardrailFunctionOutput(
        output_info={"language": "english"},
        tripwire_triggered=False
    )


@input_guardrail
async def check_safety(ctx, agent, input: str) -> GuardrailFunctionOutput:
    """
    Filter harmful or academic dishonesty queries.

    Blocking guardrail for content safety.
    """
    logger.info(f"check_safety: Checking for safety issues")

    # Check for academic dishonesty signals
    dishonesty_patterns = [
        r"\b(solve|answer|complete).*(homework|assignment|exam|test|quiz)\b",
        r"\bgive me.*(answer|solution|code)\b",
        r"\bdo my (homework|assignment|project)\b",
        r"\bcheat\b",
        r"\banswer key\b"
    ]

    input_lower = input.lower()

    for pattern in dishonesty_patterns:
        if re.search(pattern, input_lower):
            logger.warning(f"Safety check: FAILED (academic dishonesty: {pattern})")
            return GuardrailFunctionOutput(
                output_info={
                    "safe": False,
                    "issue": "academic_dishonesty",
                    "pattern": pattern
                },
                tripwire_triggered=True
            )

    # Check for inappropriate content
    inappropriate_patterns = [
        r"\b(hack|exploit|attack|ddos|malware)\b",
        r"\bhow to (break|crack|bypass)\b"
    ]

    for pattern in inappropriate_patterns:
        if re.search(pattern, input_lower):
            logger.warning(f"Safety check: FAILED (inappropriate: {pattern})")
            return GuardrailFunctionOutput(
                output_info={
                    "safe": False,
                    "issue": "inappropriate_content",
                    "pattern": pattern
                },
                tripwire_triggered=True
            )

    logger.info("Safety check: PASSED")
    return GuardrailFunctionOutput(
        output_info={"safe": True},
        tripwire_triggered=False
    )


# ===== OUTPUT GUARDRAILS (Parallel Mode) =====

@output_guardrail
async def validate_citations(ctx, agent, output: str) -> GuardrailFunctionOutput:
    """
    Ensure response has proper citations when providing factual information.

    Parallel guardrail that checks output quality.
    """
    logger.info("validate_citations: Checking for citations in output")

    # Check for citation pattern: [Week X, ...]
    citation_pattern = r'\[Week \d+[^\]]*\]'
    citations = re.findall(citation_pattern, output)

    has_citations = len(citations) > 0

    # If response is long (>100 chars) and factual, it should have citations
    if len(output) > 100 and not has_citations:
        # Check if it's a factual response (not a clarification or meta-response)
        factual_indicators = ["according to", "explained in", "described in", "the textbook"]

        if any(indicator in output.lower() for indicator in factual_indicators):
            logger.warning("Citations validation: FAILED (long factual answer without citations)")
            return GuardrailFunctionOutput(
                output_info={
                    "citations_found": 0,
                    "warning": "Long factual response missing citations"
                },
                tripwire_triggered=True
            )

    logger.info(f"Citations validation: PASSED ({len(citations)} citations found)")
    return GuardrailFunctionOutput(
        output_info={"citations_found": len(citations)},
        tripwire_triggered=False
    )


@output_guardrail
async def check_response_length(ctx, agent, output: str) -> GuardrailFunctionOutput:
    """
    Ensure response is not too short or too long.

    Parallel guardrail for response length validation.
    """
    logger.info("check_response_length: Validating response length")

    word_count = len(output.split())

    too_short = word_count < 10
    too_long = word_count > 500

    if too_short:
        logger.warning(f"Response length: TOO SHORT ({word_count} words)")
        return GuardrailFunctionOutput(
            output_info={"word_count": word_count, "issue": "too_short"},
            tripwire_triggered=True
        )

    if too_long:
        logger.warning(f"Response length: TOO LONG ({word_count} words)")
        return GuardrailFunctionOutput(
            output_info={"word_count": word_count, "issue": "too_long"},
            tripwire_triggered=True
        )

    logger.info(f"Response length: PASSED ({word_count} words)")
    return GuardrailFunctionOutput(
        output_info={"word_count": word_count},
        tripwire_triggered=False
    )


@output_guardrail
async def detect_hallucination(ctx, agent, output: str) -> GuardrailFunctionOutput:
    """
    Basic hallucination detection.

    Checks for common hallucination signals in the output.
    """
    logger.info("detect_hallucination: Checking for hallucination signals")

    # Common hallucination patterns
    hallucination_signals = [
        r"Week (14|15|16|17|18|19|20)",  # Textbook only has 13 weeks
        r"Chapter \d{2,}",  # Too many chapters
        r"according to my (knowledge|training|understanding)",  # External knowledge
        r"i (think|believe|assume)",  # Uncertain language
        r"probably|maybe|might be|could be|possibly",  # Hedge words (excessive use)
    ]

    output_lower = output.lower()
    detected_signals = []

    for pattern in hallucination_signals:
        if re.search(pattern, output_lower):
            detected_signals.append(pattern)

    if detected_signals:
        logger.warning(f"Hallucination detection: POSSIBLE ({len(detected_signals)} signals)")
        return GuardrailFunctionOutput(
            output_info={
                "likely_hallucination": True,
                "signals": detected_signals
            },
            tripwire_triggered=True
        )

    logger.info("Hallucination detection: PASSED")
    return GuardrailFunctionOutput(
        output_info={"likely_hallucination": False},
        tripwire_triggered=False
    )


# ===== GUARDRAIL RESPONSE MESSAGES =====

def get_guardrail_response(guardrail_name: str, guardrail_output: GuardrailFunctionOutput) -> str:
    """
    Get appropriate response message when a guardrail is triggered.

    Args:
        guardrail_name: Name of the triggered guardrail
        guardrail_output: Output from the guardrail

    Returns:
        User-friendly error message
    """

    if guardrail_name == "check_relevance":
        return (
            "I'm specialized in Physical AI and Robotics topics from the textbook. "
            "Your question seems to be outside this scope. "
            "Could you rephrase to focus on robotics, ROS 2, sensors, or autonomous systems?"
        )

    elif guardrail_name == "check_language":
        return (
            "I currently only support questions in English. "
            "Please rephrase your question in English, and I'll be happy to help!"
        )

    elif guardrail_name == "check_safety":
        issue = guardrail_output.output_info.get("issue")

        if issue == "academic_dishonesty":
            return (
                "I'm here to help you learn, not complete assignments. "
                "I can explain concepts, provide examples, and clarify topics. "
                "How can I help you understand this material better?"
            )
        else:
            return (
                "I can't help with that type of request. "
                "Please ask a question about Physical AI or Robotics concepts."
            )

    elif guardrail_name == "validate_citations":
        return (
            "\n\n⚠️ _Note: This response may lack proper source citations. "
            "Please verify information with the textbook._"
        )

    elif guardrail_name == "check_response_length":
        issue = guardrail_output.output_info.get("issue")

        if issue == "too_short":
            return (
                "\n\n⚠️ _The response seems brief. Would you like more detail?_"
            )
        else:  # too_long
            return (
                "\n\n⚠️ _This is a detailed response. "
                "Feel free to ask for specific parts to be clarified._"
            )

    elif guardrail_name == "detect_hallucination":
        return (
            "\n\n⚠️ _This response may contain unverified information. "
            "Please cross-reference with the textbook sections cited._"
        )

    else:
        return "A validation check was triggered. Please rephrase your question."
