import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "download_gemma4.sh"


class GemmaDownloadScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bin_dir = self.root / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        hf = bin_dir / "hf"
        hf.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$HF_TEST_LOG"\n'
            'exit "${HF_TEST_EXIT:-0}"\n', encoding="utf-8",
        )
        hf.chmod(0o755)
        self.log = self.root / "hf_args.txt"
        self.env = {
            **os.environ, "DOWNLOAD_ROOT": str(self.root / "models"),
            "DOWNLOAD_VENV": str(bin_dir.parent), "MODEL_REVISION": "pinned-sha",
            "HF_TEST_LOG": str(self.log), "HF_TOKEN": "secret-do-not-print",
            "MODEL_ID": "google/gemma-4-31B-it", "MODEL_DIR": "",
        }

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args], env=self.env, cwd=self.root,
            capture_output=True, text=True, check=False,
        )

    def test_default_download_uses_isolated_cli_and_pinned_revision(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "download", "google/gemma-4-31B-it", "--revision", "pinned-sha",
            "--local-dir", str(self.root / "models" / "gemma-4-31B-it"),
        ])
        self.assertNotIn(self.env["HF_TOKEN"], result.stdout + result.stderr)

    def test_custom_model_and_destination_with_spaces(self):
        destination = self.root / "model cache" / "gemma-4-E4B-it"
        result = self.run_script("google/gemma-4-E4B-it", str(destination))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.log.read_text().splitlines()
        self.assertEqual(args[1], "google/gemma-4-E4B-it")
        self.assertEqual(args[-1], str(destination))

    def test_download_failure_does_not_print_success(self):
        self.env["HF_TEST_EXIT"] = "9"
        result = self.run_script()
        self.assertEqual(result.returncode, 9)
        self.assertNotIn("Download complete", result.stdout)

    def test_help_and_invalid_arguments_make_no_download_calls(self):
        self.assertEqual(self.run_script("--help").returncode, 0)
        self.assertEqual(self.run_script("a", "b", "c").returncode, 2)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
