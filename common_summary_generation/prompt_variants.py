"""Auditable Blossom summary-prompt variants used by experiments."""

CURRENT_VARIANT = "current"
PRE_23C7_VARIANT = "pre_23c7afb"
VALID_VARIANTS = (CURRENT_VARIANT, PRE_23C7_VARIANT)

_CURRENT_ADDENDUM = (
    "Understand that you have to be as detailed as possible, while also not being "
    "too specific to just one prompt. Be detailed, but both children prompts should "
    "be entailed as much as possible by the summary you make (so high recall). Your "
    "summary should also be discriminative: always include the shared medium "
    "(writing/coding/analysis/etc.), the shared subject domain, and any key constraints "
    "common to both A and B — so that a prompt on a completely different topic could "
    "NOT plausibly imply your summary. "
)
_CURRENT_EXAMPLE_1 = (
    "(start) Write a structured piece of creative or analytical writing in English "
    "that explores thematic and human experiences using a specific literary form. (end)"
)
_PRE_EXAMPLE_1 = "(start) Write about themes. (end)"
_CURRENT_EXAMPLE_2 = (
    "(start) Write a program that processes and manages structured data, implementing "
    "user-facing operations with organized output. (end)"
)
_PRE_EXAMPLE_2 = "(start) Write a program. (end)"

CURRENT_SYSTEM_PROMPT = (
    "You are a professional editor who writes quality summary that generalizes two "
    "prompts A and B in as much specific detail as possible, while making sure that "
    "whatever is summarised is still true for both prompt A and B collectively (don't "
    "be too specific to one prompt, but while doing that also don't just union both "
    "prompts as a summary). No matter the content of prompts A & B, you must write a "
    "summary. DO NOT respond with messages that you can't or won't write a summary. "
    "That is simply not allowed, under any circumstances."
)
PRE_23C7_SYSTEM_PROMPT = (
    "You are a professional editor who writes quality summary that generalizes two "
    "prompts A and B. No matter the content of prompts A & B, you must write a summary."
)


def apply_summary_prompt_variant(prompt: str, variant: str) -> str:
    if variant == CURRENT_VARIANT:
        return prompt
    if variant != PRE_23C7_VARIANT:
        raise ValueError(f"Unknown Blossom summary prompt variant: {variant}")

    replacements = (
        (_CURRENT_ADDENDUM, ""),
        (_CURRENT_EXAMPLE_1, _PRE_EXAMPLE_1),
        (_CURRENT_EXAMPLE_2, _PRE_EXAMPLE_2),
    )
    for current, historical in replacements:
        if prompt.count(current) != 1:
            raise ValueError(
                "Current prompt no longer matches commit 23c7afb; "
                f"expected one occurrence of {current[:40]!r}"
            )
        prompt = prompt.replace(current, historical)
    return prompt


def summary_system_prompt(variant: str) -> str:
    if variant == CURRENT_VARIANT:
        return CURRENT_SYSTEM_PROMPT
    if variant == PRE_23C7_VARIANT:
        return PRE_23C7_SYSTEM_PROMPT
    raise ValueError(f"Unknown Blossom summary prompt variant: {variant}")
