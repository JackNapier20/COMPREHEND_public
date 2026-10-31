import unittest

from dataset_prep.sample_wb_random import demonstration_texts, leakage_reason, normalize


class WildBenchSamplingTests(unittest.TestCase):
    def test_extracts_all_three_demonstration_roles(self):
        prompt = """Example 1
Prompt A:
Alpha text
Prompt B:
Beta text
Prompt C:
(start) Gamma text (end)
Now carry out this task"""
        self.assertEqual(
            demonstration_texts(prompt),
            ["Alpha text", "Beta text", "(start) Gamma text (end)"],
        )

    def test_normalize_ignores_case_and_punctuation(self):
        self.assertEqual(normalize("Hello, WORLD!"), "hello world")

    def test_exact_demonstration_is_rejected(self):
        example = "Write a detailed technical comparison of two systems."
        self.assertIsNotNone(leakage_reason(example, example, [example]))

    def test_modified_near_verbatim_demonstration_is_rejected(self):
        example = (
            "Write a Python program that reads purchase data, calculates total sales, "
            "and outputs a structured summary report."
        )
        modified = (
            "Write a Python program that reads purchase data, calculates total sales, "
            "and produces a structured summary report."
        )
        self.assertIsNotNone(leakage_reason(modified, example, [example]))

    def test_unrelated_prompt_is_allowed(self):
        example = "Write a Python REST API for a task application."
        unrelated = "Explain the geological processes that form volcanic island chains."
        self.assertIsNone(leakage_reason(unrelated, example, [example]))


if __name__ == "__main__":
    unittest.main()
