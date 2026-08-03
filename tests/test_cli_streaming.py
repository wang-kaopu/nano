from unittest.mock import patch

from nano import cli
from nano.runtime.query_events import QueryEvent


class _StreamingAgent:
    """用于验证 CLI 流式输出的最小 Agent 替身。"""

    model_client = object()

    def ask(self, user_message, event_callback=None):
        """模拟逐段返回带协议标签的最终回答。"""
        assert user_message == "Say hello"
        event_callback(QueryEvent("model_requested"))
        event_callback(QueryEvent("text_delta", {"text": "<final>Hel"}))
        event_callback(QueryEvent("text_delta", {"text": "lo.</final>"}))
        event_callback(QueryEvent("final", {"answer": "Hello."}))
        return "Hello."


def test_cli_prints_text_deltas_without_protocol_tags(capsys):
    with patch("nano.cli.build_agent", return_value=_StreamingAgent()), patch("nano.cli.build_welcome", return_value="Welcome"):
        assert cli.main(["Say hello"]) == 0

    assert capsys.readouterr().out == "Welcome\n\nHello.\n"
