import unittest
from pathlib import Path

from common_summary_generation.prompt_variants import (
    CURRENT_SYSTEM_PROMPT,
    PRE_23C7_SYSTEM_PROMPT,
    apply_summary_prompt_variant,
    summary_system_prompt,
)
from dataset_prep.sample_wb_random import active_summary_prompt, demonstration_texts


class PromptVariantTests(unittest.TestCase):
    def test_pre_commit_variant_restores_changed_prompt_sections(self):
        current = active_summary_prompt(Path("common_summary_generation/summarize.py"))
        historical = apply_summary_prompt_variant(current, "pre_23c7afb")

        self.assertIn("(start) Write about themes. (end)", historical)
        self.assertIn("(start) Write a program. (end)", historical)
        self.assertNotIn("Understand that you have to be as detailed as possible", historical)
        self.assertEqual(len(demonstration_texts(current)), 45)
        self.assertEqual(len(demonstration_texts(historical)), 45)

    def test_system_prompt_variant(self):
        self.assertEqual(summary_system_prompt("current"), CURRENT_SYSTEM_PROMPT)
        self.assertEqual(summary_system_prompt("pre_23c7afb"), PRE_23C7_SYSTEM_PROMPT)
