"""Tests for plugins/memory/openviking/__init__.py — URI normalization and payload handling."""

import json

from plugins.memory.openviking import OpenVikingMemoryProvider


class FakeVikingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append((path, params or {}))
        response = self.responses[(path, tuple(sorted((params or {}).items())))]
        if isinstance(response, Exception):
            raise response
        return response


class TestOpenVikingSummaryUriNormalization:
    def test_normalize_summary_uri_maps_pseudo_files_to_parent_directory(self):
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/.overview.md") == "viking://user/hermes"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://resources/.abstract.md") == "viking://resources"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://") == "viking://"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/memories/profile.md") == "viking://user/hermes/memories/profile.md"


class TestOpenVikingTurnConversion:
    def test_extract_current_turn_anchors_on_latest_matching_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        turn = OpenVikingMemoryProvider._extract_current_turn_messages(
            messages,
            "Please inspect the repository for assemble hooks.",
            "The current main does not expose assemble.",
        )

        assert turn == messages[2:]

    def test_messages_to_openviking_batch_coalesces_tool_results(self):
        turn = [
            {"role": "user", "content": "Please inspect the repository for assemble hooks."},
            {
                "role": "assistant",
                "content": "I will search the codebase.",
                "tool_calls": [
                    {
                        "id": "call_rg_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg assemble"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rg_1",
                "name": "shell_command",
                "content": "agent/context_engine.py: no preassemble hook",
            },
            {"role": "assistant", "content": "The current main does not expose assemble."},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert [message["role"] for message in batch] == ["user", "assistant", "user", "assistant"]
        assert batch[0]["parts"] == [
            {"type": "text", "text": "Please inspect the repository for assemble hooks."}
        ]
        assert batch[1]["parts"] == [
            {"type": "text", "text": "I will search the codebase."}
        ]
        assert batch[2]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_rg_1",
                "tool_name": "shell_command",
                "tool_input": {"command": "rg assemble"},
                "tool_output": "agent/context_engine.py: no preassemble hook",
                "tool_status": "completed",
            }
        ]
        assert batch[3]["parts"] == [
            {"type": "text", "text": "The current main does not expose assemble."}
        ]

    def test_messages_to_openviking_batch_marks_json_tool_error_results(self):
        turn = [
            {"role": "user", "content": "Check the file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "missing.md"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "name": "read_file",
                "content": json.dumps({"error": "File not found", "exit_code": 1}),
            },
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert batch[1]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_read_1",
                "tool_name": "read_file",
                "tool_input": {"path": "missing.md"},
                "tool_output": json.dumps({"error": "File not found", "exit_code": 1}),
                "tool_status": "error",
            }
        ]

    def test_messages_to_openviking_batch_keeps_pending_tool_call_without_result(self):
        turn = [
            {"role": "user", "content": "Start a long running check."},
            {
                "role": "assistant",
                "content": "Starting it now.",
                "tool_calls": [
                    {
                        "id": "call_long_1",
                        "type": "function",
                        "function": {
                            "name": "long_check",
                            "arguments": json.dumps({"target": "repo"}),
                        },
                    }
                ],
            },
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert batch[1]["parts"] == [
            {"type": "text", "text": "Starting it now."},
            {
                "type": "tool",
                "tool_id": "call_long_1",
                "tool_name": "long_check",
                "tool_input": {"target": "repo"},
                "tool_status": "pending",
            },
        ]

    def test_messages_to_openviking_batch_coalesces_adjacent_tool_results(self):
        turn = [
            {"role": "user", "content": "Run both tools."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "first_tool",
                            "arguments": json.dumps({"x": 1}),
                        },
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "second_tool",
                            "arguments": json.dumps({"y": 2}),
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "name": "first_tool", "content": "a"},
            {"role": "tool", "tool_call_id": "call_b", "name": "second_tool", "content": "b"},
            {"role": "assistant", "content": "Done."},
        ]

        batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

        assert [message["role"] for message in batch] == ["user", "user", "assistant"]
        assert batch[1]["parts"] == [
            {
                "type": "tool",
                "tool_id": "call_a",
                "tool_name": "first_tool",
                "tool_input": {"x": 1},
                "tool_output": "a",
                "tool_status": "completed",
            },
            {
                "type": "tool",
                "tool_id": "call_b",
                "tool_name": "second_tool",
                "tool_input": {"y": 2},
                "tool_output": "b",
                "tool_status": "completed",
            },
        ]

    def test_messages_to_openviking_batch_skips_openviking_recall_tool_results(self):
        for recall_tool_name in ("viking_search", "viking_read", "viking_browse"):
            turn = [
                {"role": "user", "content": "What did we decide about context assembly?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_recall_1",
                            "type": "function",
                            "function": {
                                "name": recall_tool_name,
                                "arguments": json.dumps({"query": "context assembly decision"}),
                            },
                        },
                        {
                            "id": "call_shell_1",
                            "type": "function",
                            "function": {
                                "name": "shell_command",
                                "arguments": json.dumps({"command": "rg preassemble"}),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_recall_1",
                    "name": recall_tool_name,
                    "content": json.dumps({
                        "results": [
                            {
                                "uri": "viking://user/hermes/memories/context",
                                "abstract": "Old OpenViking memory content",
                            }
                        ]
                    }),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_shell_1",
                    "name": "shell_command",
                    "content": "plugins/memory/openviking/__init__.py",
                },
                {"role": "assistant", "content": "We decided to keep sync_turn scoped to ingestion."},
            ]

            batch = OpenVikingMemoryProvider._messages_to_openviking_batch(turn)

            assert [message["role"] for message in batch] == ["user", "user", "assistant"]
            assert batch[1]["parts"] == [
                {
                    "type": "tool",
                    "tool_id": "call_shell_1",
                    "tool_name": "shell_command",
                    "tool_input": {"command": "rg preassemble"},
                    "tool_output": "plugins/memory/openviking/__init__.py",
                    "tool_status": "completed",
                }
            ]
            batch_text = json.dumps(batch)
            assert recall_tool_name not in batch_text
            assert "Old OpenViking memory content" not in batch_text


class TestOpenVikingRead:
    def test_overview_read_normalizes_uri_and_unwraps_result(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": {"content": "overview text"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["uri"] == "viking://user/hermes/.overview.md"
        assert result["resolved_uri"] == "viking://user/hermes"
        assert result["level"] == "overview"
        assert result["content"] == "overview text"
        assert provider._client.calls == [(
            "/api/v1/content/overview",
            {"uri": "viking://user/hermes"},
        )]

    def test_full_read_keeps_original_uri(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", "viking://user/hermes/memories/profile.md"),),
                ): {"result": "full text"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/memories/profile.md", "level": "full"}))

        assert result["uri"] == "viking://user/hermes/memories/profile.md"
        assert result["resolved_uri"] == "viking://user/hermes/memories/profile.md"
        assert result["level"] == "full"
        assert result["content"] == "full text"
        assert provider._client.calls == [(
            "/api/v1/content/read",
            {"uri": "viking://user/hermes/memories/profile.md"},
        )]

    def test_overview_file_uri_routes_straight_to_content_read_via_stat_probe(self):
        """Pre-check via fs/stat: file URIs skip the directory-only endpoint entirely."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): {"result": {"isDir": False}},
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["resolved_uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_overview_dir_uri_skips_stat_when_pseudo_summary(self):
        """Pseudo-URI path already resolves to dir, so no stat probe needed."""
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": "overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["content"] == "overview"
        # No fs/stat call — normalization already determined it's a directory.
        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]

    def test_overview_directory_uri_uses_stat_probe_then_overview(self):
        """Non-pseudo directory URI: stat → isDir=True → summary endpoint."""
        provider = OpenVikingMemoryProvider()
        dir_uri = "viking://user/hermes/memories"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", dir_uri),),
                ): {"result": {"isDir": True}},
                (
                    "/api/v1/content/overview",
                    (("uri", dir_uri),),
                ): {"result": "dir overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": dir_uri, "level": "overview"}))

        assert result["content"] == "dir overview"
        assert "fallback" not in result
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": dir_uri}),
            ("/api/v1/content/overview", {"uri": dir_uri}),
        ]

    def test_overview_file_uri_falls_back_via_exception_when_stat_indeterminate(self):
        """If fs/stat raises or returns unknown shape, legacy exception fallback still kicks in."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): RuntimeError("stat unavailable"),
                (
                    "/api/v1/content/overview",
                    (("uri", file_uri),),
                ): RuntimeError("500 Internal Server Error"),
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "fallback full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "fallback full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/overview", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_summary_uri_error_does_not_fallback_and_raises(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): RuntimeError("500 Internal Server Error"),
            }
        )

        try:
            provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"})
            assert False, "Expected summary endpoint error to be raised"
        except RuntimeError:
            pass

        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]


class TestOpenVikingBrowse:
    def test_list_browse_unwraps_and_normalizes_entry_shapes(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/ls",
                    (("uri", "viking://user/hermes"),),
                ): {
                    "result": {
                        "entries": [
                            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir"},
                            {"rel_path": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "isDir": False, "abstract": "Profile"},
                        ]
                    }
                },
            }
        )

        result = json.loads(provider._tool_browse({"action": "list", "path": "viking://user/hermes"}))

        assert result["path"] == "viking://user/hermes"
        assert result["entries"] == [
            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir", "abstract": ""},
            {"name": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "type": "file", "abstract": "Profile"},
        ]
        assert provider._client.calls == [(
            "/api/v1/fs/ls",
            {"uri": "viking://user/hermes"},
        )]
