"""What a small model writes instead of what it should: tool calls as text, reasoning in the reply."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from robot.cognition.backend import ToolCall

_THINK_TAIL = re.compile(r"</t(?:h(?:i(?:n(?:k(?:>)?)?)?)?)?\s*$")
_CALL_TOKENS = re.compile(r"<\|tool_call_(?:start|end)\|>|</?think>|```(?:python|json)?")
# <tool_call><function=go_to><parameter=target>ball_1</parameter></function></tool_call>
_XML_CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
_XML_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)|<function=\w+>.*?(?:</function>|$)", re.S)


def strip_reasoning(content: str) -> tuple[str, bool]:
    """Return (reply, leaked): drop <think> blocks; a reply that is only reasoning counts as leaked."""
    text = content or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.S)
    if _THINK_TAIL.search(text):
        return "", True  # reasoning cut off right where the answer would have started
    leaked = bool(re.match(r"\s*(?:The user|User (?:said|asked|wants|introduced)|I should|Let me think)", text))
    return ("" if leaked else text.strip()), leaked


def _xml_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def strip_written_calls(text: str) -> str:
    """Remove tool calls written out as markup: they must never be spoken."""
    return _XML_BLOCK.sub("", text or "").strip()


def recover_text_tool_calls(content: str, tool_names: set[str]) -> list[ToolCall]:
    """Small models sometimes write a call as text ("observe()", or JSON) instead of a real call.

    Only a reply that is *entirely* calls to known tools is recovered; prose is left alone.
    """
    xml = _XML_CALL.findall(content or "")
    if xml and not strip_written_calls(re.sub(r"</?tool_call>", "", content or "")):
        return [ToolCall(f"recovered_{i}", name, {k: _xml_value(v) for k, v in _XML_PARAM.findall(body)})
                for i, (name, body) in enumerate(xml[:3]) if name in tool_names]
    text = _CALL_TOKENS.sub("", content or "").strip()
    if not text or len(text) > 400:
        return []
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except ValueError:
            return []
        name, args = raw.get("name"), raw.get("arguments", raw.get("parameters", {}))
        if name in tool_names and isinstance(args, dict):
            return [ToolCall("recovered_0", name, args)]
        return []
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return []
    nodes = tree.elts if isinstance(tree, ast.List) else [tree]
    calls = []
    for i, node in enumerate(nodes[:3]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in tool_names
                and not node.args):
            return []
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except (ValueError, TypeError):  # TypeError: unhashable literal keys such as {[1]: 2}
            return []
        calls.append(ToolCall(f"recovered_{i}", node.func.id, args))
    return calls
