from __future__ import annotations

"""Unit tests for HuggingFaceExporter.push_to_hub and CustomTokenizer.push_to_hub."""

import contextlib
import os
import sys
import types
import unittest
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

from uniqtoken.hf_exporter import HuggingFaceExporter
from uniqtoken.tokenizer import CustomTokenizer


@contextlib.contextmanager
def _huggingface_hub_module() -> Iterator[Any]:
    """Yields an importable ``huggingface_hub`` module without leaking a stub into ``sys.modules``."""
    try:
        import huggingface_hub as real_hub

        yield real_hub
    except ImportError:
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = MagicMock  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            yield fake_hub


def _build_tiny_tokenizer() -> CustomTokenizer:
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "fast parallel batch encoding verification",
        "neural language model subword regularization",
    ]
    return CustomTokenizer.train_from_corpus(
        corpus, target_vocab_size=320, hex_literals=False, digit_chunking="greedy", verbose=False
    )


class PushToHubTests(unittest.TestCase):
    tokenizer: CustomTokenizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = _build_tiny_tokenizer()

    def test_push_to_hub_calls_create_repo_and_upload_folder(self) -> None:
        captured: dict[str, Any] = {}
        commit_url = "https://huggingface.co/test-user/test-repo/commit/abc123"

        def _upload_folder_side_effect(**kwargs: Any) -> Any:
            captured["kwargs"] = kwargs
            folder = kwargs["folder_path"]
            # Files must already be staged while the temp dir is alive.
            captured["staged_files"] = sorted(os.listdir(folder))
            with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
                captured["readme"] = f.read()
            # Mirrors huggingface_hub.CommitInfo: only `commit_url` is read by push_to_hub.
            return types.SimpleNamespace(commit_url=commit_url)

        with _huggingface_hub_module(), patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api = MagicMock()
            mock_api.upload_folder.side_effect = _upload_folder_side_effect
            mock_hf_api.return_value = mock_api

            result = HuggingFaceExporter.push_to_hub(self.tokenizer, "test-user/test-repo", token="hf_test_token")

            # The upload commit URL is returned for CI logs and verification.
            self.assertEqual(result, commit_url)
            # HfApi instantiated with the provided token.
            mock_hf_api.assert_called_once_with(token="hf_test_token")
            # Repo is created idempotently; visibility is set at creation time.
            mock_api.create_repo.assert_called_once_with(repo_id="test-user/test-repo", exist_ok=True, private=False)
            # Folder upload uses defaults for commit message and staging dir contents.
            mock_api.upload_folder.assert_called_once()
            _, upload_kwargs = mock_api.upload_folder.call_args
            self.assertEqual(upload_kwargs["repo_id"], "test-user/test-repo")
            self.assertEqual(upload_kwargs["commit_message"], "Upload UniqToken model")
            self.assertNotIn("private", upload_kwargs)
            self.assertIsInstance(upload_kwargs["folder_path"], str)
            self.assertTrue(len(upload_kwargs["folder_path"]) > 0)

        self.assertIn("tokenizer.json", captured["staged_files"])
        self.assertIn("tokenizer_config.json", captured["staged_files"])
        self.assertIn("README.md", captured["staged_files"])
        self.assertIn("test-user/test-repo", captured["readme"])
        self.assertIn(f"vocab_size: {self.tokenizer.vocab_size}", captured["readme"])
        self.assertIn(f"byte_fallback: {self.tokenizer.model.byte_fallback}", captured["readme"])
        self.assertIn(f"unk_token: {self.tokenizer.model.unk_token}", captured["readme"])

    def test_push_to_hub_forwards_commit_message_and_kwargs(self) -> None:
        with _huggingface_hub_module(), patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api = MagicMock()
            mock_hf_api.return_value = mock_api

            HuggingFaceExporter.push_to_hub(
                self.tokenizer,
                "org/model",
                token=None,
                commit_message="Custom message",
                private=True,
            )

            mock_hf_api.assert_called_once_with(token=None)
            # `private` is routed to create_repo; upload_folder accepts no such argument.
            mock_api.create_repo.assert_called_once_with(repo_id="org/model", exist_ok=True, private=True)
            _, upload_kwargs = mock_api.upload_folder.call_args
            self.assertEqual(upload_kwargs["repo_id"], "org/model")
            self.assertEqual(upload_kwargs["commit_message"], "Custom message")
            self.assertNotIn("private", upload_kwargs)

    def test_push_to_hub_rejects_invalid_repo_id(self) -> None:
        for bad_repo_id in ("", "noslash", "owner/", "/model", "owner/model/extra"):
            with self.assertRaises(ValueError):
                HuggingFaceExporter.push_to_hub(self.tokenizer, bad_repo_id)
        with self.assertRaises(ValueError):
            HuggingFaceExporter.push_to_hub(self.tokenizer, None)  # type: ignore[arg-type]

    def test_push_to_hub_rejects_run_as_future(self) -> None:
        # A background Future would read from the temporary staging directory after it is deleted.
        with _huggingface_hub_module(), patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api = MagicMock()
            mock_hf_api.return_value = mock_api
            with self.assertRaises(ValueError):
                HuggingFaceExporter.push_to_hub(self.tokenizer, "test-user/test-repo", run_as_future=True)
            mock_api.create_repo.assert_not_called()
            mock_api.upload_folder.assert_not_called()

    def test_push_to_hub_returns_pr_url_for_multi_commits(self) -> None:
        # multi_commits=True mode returns the PR URL string instead of a CommitInfo.
        pr_url = "https://huggingface.co/test-user/test-repo/discussions/1"
        with _huggingface_hub_module(), patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api = MagicMock()
            mock_api.upload_folder.return_value = pr_url
            mock_hf_api.return_value = mock_api

            result = HuggingFaceExporter.push_to_hub(self.tokenizer, "test-user/test-repo", multi_commits=True)

            self.assertEqual(result, pr_url)
            _, upload_kwargs = mock_api.upload_folder.call_args
            self.assertTrue(upload_kwargs["multi_commits"])

    def test_push_to_hub_missing_dependency_raises_helpful_import_error(self) -> None:
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaises(ImportError) as ctx:
                HuggingFaceExporter.push_to_hub(self.tokenizer, "test-user/test-repo")
        self.assertIn('pip install "uniqtoken[huggingface]"', str(ctx.exception))

    def test_custom_tokenizer_push_to_hub_delegates_to_exporter(self) -> None:
        with patch.object(HuggingFaceExporter, "push_to_hub", return_value="https://hub/commit/1") as mock_push:
            result = self.tokenizer.push_to_hub(
                "test-user/test-repo", token="hf_tok", commit_message="msg", private=True
            )
            mock_push.assert_called_once_with(
                self.tokenizer, "test-user/test-repo", token="hf_tok", commit_message="msg", private=True
            )
            self.assertEqual(result, "https://hub/commit/1")


if __name__ == "__main__":
    unittest.main()
