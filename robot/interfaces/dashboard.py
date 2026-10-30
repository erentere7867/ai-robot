"""Zero-dependency async HTTP diagnostic dashboard and REST API server."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from robot.body.hal import BodyPose
from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth
from robot.world.spatial_map import CELL_MM

if TYPE_CHECKING:
    from robot.main import RobotHarness

logger = logging.getLogger(__name__)

# Events that explain a decision. They are tapped synchronously (cheap deque appends) so none is
# lost to the lossy dashboard queue, and so the camera frame of a view can be paired with it.
THINKING_PATTERNS = (
    "cognition.*",
    "skill.*",
    "perception.view",
    "memory.retrieved",
    "robot.response",
    "turn.cancelled",
    "reflex.triggered",
)
MAX_THINKING_ENTRIES = 240
KEPT_VIEW_FRAMES = 6
COVERAGE_DOWNSAMPLE = 2  # spatial-map cells per coverage cell sent to the browser


def _clip_text(value: Any, limit: int) -> str:
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip_json(value: Any, limit: int) -> Any:
    """A JSON-safe copy of `value`, or a truncated JSON string when its encoding is long."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    except (TypeError, ValueError):
        return _clip_text(value, limit)
    return json.loads(text) if len(text) <= limit else text[: limit - 1] + "…"


def _query_flag(query: dict[str, list[str]], key: str) -> bool:
    return (query.get(key) or [""])[0].strip().lower() in {"1", "true", "yes", "on"}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SPROCKET // EMBODIED AI HARNESS DASHBOARD</title>
<style>
  :root {
    --bg-dark: #070b13;
    --bg-card: rgba(14, 22, 38, 0.88);
    --bg-card-hover: rgba(20, 32, 54, 0.95);
    --bg-input: #090e1a;
    --border: #1a2744;
    --border-glow: rgba(0, 240, 255, 0.28);
    --text-main: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #00f0ff;
    --accent-glow: rgba(0, 240, 255, 0.25);
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.25);
    --amber: #f59e0b;
    --amber-glow: rgba(245, 158, 11, 0.25);
    --rose: #f43f5e;
    --rose-glow: rgba(244, 63, 94, 0.35);
    --violet: #8b5cf6;
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg-dark);
    background-image: radial-gradient(circle at 50% 0%, #111d33 0%, #070b13 70%);
    color: var(--text-main);
    min-height: 100vh;
    overflow-x: hidden;
  }
  /* Custom Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: rgba(10, 16, 28, 0.6); }
  ::-webkit-scrollbar-thumb { background: #223555; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #324e7b; }

  /* Top Navigation & Status Bar */
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: rgba(11, 18, 32, 0.85);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #fff;
    text-shadow: 0 0 12px var(--accent-glow);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .brand-logo::before {
    content: "";
    display: inline-block;
    width: 10px;
    height: 10px;
    background: var(--accent);
    border-radius: 2px;
    box-shadow: 0 0 10px var(--accent);
  }
  .brand-sub {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    color: var(--text-muted);
    border-left: 1px solid var(--border);
    padding-left: 12px;
    text-transform: uppercase;
  }
  .header-status-strip {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(16, 27, 46, 0.8);
    border: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
    animation: pulse-dot 1.8s infinite ease-in-out;
  }
  @keyframes pulse-dot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.6; }
  }
  .status-disconnected .pulse-dot {
    background: var(--rose);
    box-shadow: 0 0 8px var(--rose);
  }
  .status-pill strong {
    color: var(--text-main);
  }
  .fsm-tag {
    color: var(--accent);
    text-transform: uppercase;
  }
  .header-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  /* Buttons */
  button, .btn {
    background: #142138;
    color: var(--text-main);
    border: 1px solid #23375c;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  button:hover, .btn:hover {
    background: #1a2d4c;
    border-color: var(--accent);
    color: #fff;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
  }
  button:active, .btn:active {
    transform: scale(0.97);
  }
  .btn-accent {
    background: #003644;
    border-color: var(--accent);
    color: #e0faff;
    font-weight: 600;
  }
  .btn-accent:hover {
    background: #004d61;
    color: #fff;
    box-shadow: 0 0 14px var(--accent-glow);
  }
  .btn-danger {
    background: #3b111a;
    border-color: var(--rose);
    color: #ffe4e8;
    font-weight: 600;
  }
  .btn-danger:hover {
    background: #521623;
    box-shadow: 0 0 14px var(--rose-glow);
  }
  .btn-danger.active {
    background: var(--rose);
    color: #fff;
    box-shadow: 0 0 20px var(--rose);
    animation: estop-pulse 1s infinite alternate;
  }
  @keyframes estop-pulse {
    from { opacity: 0.85; }
    to { opacity: 1; transform: scale(1.02); }
  }
  .btn-sm {
    padding: 4px 8px;
    font-size: 11px;
    border-radius: 6px;
  }

  /* Grid Layout */
  .cockpit-grid {
    display: grid;
    grid-template-columns: 360px 1.25fr 1.15fr;
    gap: 16px;
    padding: 16px;
    max-width: 1920px;
    margin: 0 auto;
  }
  @media (max-width: 1440px) {
    .cockpit-grid { grid-template-columns: 340px 1fr 1fr; }
  }
  @media (max-width: 1120px) {
    .cockpit-grid { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 760px) {
    .cockpit-grid { grid-template-columns: 1fr; }
  }

  .cockpit-col {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Panel Cards */
  .panel-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .panel-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .panel-icon {
    width: 15px;
    height: 15px;
    color: var(--accent);
  }
  .panel-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(0, 240, 255, 0.1);
    border: 1px solid rgba(0, 240, 255, 0.25);
    color: var(--accent);
    text-transform: uppercase;
  }

  /* Section Subtitles */
  .section-subtitle {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* Canvas Containers */
  .face-canvas-container {
    width: 100%;
    background: #060910;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
  }
  canvas#faceCanvas {
    width: 100%;
    height: 210px;
    display: block;
    background: #060910;
  }

  /* Emotion Chips Grid */
  .chips-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .chip-btn {
    padding: 4px 8px;
    font-size: 11px;
    background: #0e192c;
    border: 1px solid #1c2e4c;
    border-radius: 6px;
    color: var(--text-secondary);
  }
  .chip-btn:hover {
    border-color: var(--accent);
    color: #fff;
  }
  .chip-btn.active {
    background: #003644;
    border-color: var(--accent);
    color: #d6fbff;
    font-weight: 600;
    box-shadow: 0 0 10px var(--accent-glow);
  }

  .control-row {
    display: flex;
    gap: 10px;
  }
  .select-group {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .select-group label {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  select, input {
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s;
  }
  select:focus, input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
  }

  /* Audio Waveform */
  .speech-waveform {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 28px;
    background: #080e1b;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .wave-bar {
    width: 4px;
    height: 4px;
    background: #233857;
    border-radius: 2px;
    transition: height 0.15s ease, background-color 0.15s ease;
  }
  .speech-waveform.active .wave-bar {
    background: var(--emerald);
    box-shadow: 0 0 6px var(--emerald);
    animation: wave-anim 0.8s infinite ease-in-out alternate;
  }
  .speech-waveform.active .wave-bar:nth-child(2) { animation-delay: 0.1s; }
  .speech-waveform.active .wave-bar:nth-child(3) { animation-delay: 0.2s; }
  .speech-waveform.active .wave-bar:nth-child(4) { animation-delay: 0.3s; }
  .speech-waveform.active .wave-bar:nth-child(5) { animation-delay: 0.4s; }
  .speech-waveform.active .wave-bar:nth-child(6) { animation-delay: 0.3s; }
  .speech-waveform.active .wave-bar:nth-child(7) { animation-delay: 0.2s; }
  .speech-waveform.active .wave-bar:nth-child(8) { animation-delay: 0.1s; }
  @keyframes wave-anim {
    0% { height: 6px; }
    100% { height: 22px; }
  }

  .speech-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 11px;
  }
  .speech-item {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .speech-label {
    font-size: 9px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    min-width: 90px;
  }
  .speech-val {
    color: var(--text-main);
    word-break: break-word;
    font-style: italic;
  }

  /* Services Grid */
  .services-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }
  .svc-item {
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .svc-name {
    font-size: 10px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
  }
  .svc-status {
    font-size: 11px;
    font-weight: 700;
    color: var(--emerald);
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .svc-status::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 6px var(--emerald);
  }

  /* Head Camera Viewport */
  .camera-viewport {
    position: relative;
    width: 100%;
    height: 240px;
    background: #05080f;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .camera-viewport img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    background: #05080f;
  }
  .cam-overlay-badge {
    position: absolute;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(7, 12, 22, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.3);
    font-size: 10px;
    font-weight: 600;
    color: var(--accent);
    pointer-events: none;
    letter-spacing: 0.5px;
  }
  .cam-overlay-badge.top-left { top: 8px; left: 8px; }
  .cam-overlay-badge.top-right { top: 8px; right: 8px; }

  .vision-hud {
    background: #080d19;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .vision-summary {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.4;
  }
  .vision-detections {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .vis-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 999px;
    background: #0f192b;
    border: 1px solid var(--border);
    font-size: 10px;
    font-weight: 600;
    color: var(--text-main);
  }

  /* Telemetry Grid & Gauges */
  .telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }
  .telem-item {
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .telem-large {
    grid-column: span 2;
  }
  .telem-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .telem-name {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  .telem-badge {
    font-size: 9px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid var(--emerald);
    color: var(--emerald);
  }
  .telem-badge.status-warning {
    background: rgba(245, 158, 11, 0.15);
    border-color: var(--amber);
    color: var(--amber);
  }
  .telem-badge.status-danger {
    background: rgba(244, 63, 94, 0.15);
    border-color: var(--rose);
    color: var(--rose);
  }
  .telem-main-val {
    font-family: Consolas, monospace;
    font-size: 20px;
    font-weight: 800;
    color: #fff;
    font-variant-numeric: tabular-nums;
  }
  .telem-unit {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
  }
  .telem-sub {
    font-size: 10px;
    color: var(--text-muted);
    font-family: Consolas, monospace;
  }
  .telem-sub strong { color: var(--text-secondary); }
  .telem-dual-val {
    display: flex;
    justify-content: space-between;
    font-family: Consolas, monospace;
    font-size: 13px;
    color: #fff;
  }
  .coord-label { color: var(--text-muted); font-weight: 600; margin-right: 2px; }

  /* Meter Bar */
  .meter-bar-track {
    width: 100%;
    height: 6px;
    background: #141e30;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }
  .meter-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--emerald) 0%, var(--accent) 100%);
    border-radius: 3px;
    transition: width 0.2s ease, background 0.2s ease;
  }
  .meter-bar-fill.meter-amber {
    background: linear-gradient(90deg, var(--amber) 0%, #ff8c00 100%);
  }
  .meter-bar-fill.meter-danger {
    background: linear-gradient(90deg, var(--rose) 0%, #ff0055 100%);
    box-shadow: 0 0 8px var(--rose);
  }
  .meter-scale {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: var(--text-muted);
    font-family: Consolas, monospace;
  }

  .safety-pills {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin-top: 2px;
  }
  .safety-chip {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 7px;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .chip-safe {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
    color: var(--emerald);
  }
  .chip-alert {
    background: rgba(244, 63, 94, 0.15);
    border-color: var(--rose);
    color: var(--rose);
    box-shadow: 0 0 6px var(--rose);
  }

  /* 2D Tactical Radar */
  .radar-split {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 12px;
    align-items: center;
  }
  @media (max-width: 580px) {
    .radar-split { grid-template-columns: 1fr; }
  }
  .radar-canvas-wrap {
    width: 220px;
    height: 220px;
    background: #05080f;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  canvas#radarCanvas {
    width: 220px;
    height: 220px;
    display: block;
  }
  .radar-legend-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    height: 220px;
  }
  .legend-header {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  .obs-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 11px;
  }
  .obs-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 6px;
    border-radius: 4px;
    background: #0d1526;
    border-left: 2px solid var(--accent);
  }
  .obs-item strong { color: #fff; }
  .obs-meta { font-size: 10px; color: var(--text-muted); }
  .empty-hint { color: var(--text-muted); font-style: italic; font-size: 11px; padding: 6px; }

  /* Autonomy Bar */
  .autonomy-bar {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }
  .autonomy-status-line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
  }
  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
  }
  .btn-group-row {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
  }

  /* Chat Box */
  .chat-container {
    height: 210px;
    overflow-y: auto;
    background: #080d18;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 12px;
  }
  .chat-bubble {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .chat-u {
    align-self: flex-end;
    background: #142844;
    border: 1px solid #1f3d6b;
    border-radius: 10px 10px 2px 10px;
    padding: 6px 10px;
    max-width: 85%;
    color: #e2f1ff;
  }
  .chat-r {
    align-self: flex-start;
    background: #111a2d;
    border: 1px solid #1c2c4a;
    border-radius: 10px 10px 10px 2px;
    padding: 6px 10px;
    max-width: 85%;
    color: var(--text-main);
  }
  .chat-meta {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  .chat-form {
    display: flex;
    gap: 6px;
  }
  .chat-form input { flex: 1; }

  .quick-prompts {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .quick-chip {
    padding: 3px 8px;
    font-size: 11px;
    background: #0e192c;
    border: 1px solid #1c2e4c;
    border-radius: 6px;
    color: var(--text-secondary);
  }
  .quick-chip:hover {
    border-color: var(--accent);
    color: #fff;
  }

  /* Motion & Actuator Deck */
  .motion-controls-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    align-items: stretch;
  }
  @media (max-width: 580px) {
    .motion-controls-grid { grid-template-columns: 1fr; }
  }
  .dpad-cluster {
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: center;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
  }
  .dpad-row {
    display: flex;
    gap: 6px;
  }
  .dpad-btn {
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    min-width: 80px;
  }
  .dpad-stop {
    background: #45121b;
    border-color: var(--rose);
    color: #ffccd5;
    font-weight: 800;
  }
  .dpad-stop:hover {
    background: #661826;
    box-shadow: 0 0 12px var(--rose-glow);
  }
  .actuator-cluster {
    display: flex;
    flex-direction: column;
    gap: 8px;
    justify-content: space-between;
  }
  .actuator-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
  }
  .actuator-group label {
    font-size: 9px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.6px;
  }
  .btn-group-col {
    display: flex;
    gap: 4px;
  }
  .btn-group-col button { flex: 1; font-size: 11px; padding: 4px 6px; }

  /* Tabs for Memory & Events */
  .tabs-header {
    display: flex;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
    gap: 6px;
  }
  .tab-btn {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .tab-btn:hover { color: var(--text-main); border-color: transparent; }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    box-shadow: none;
  }
  .tab-action-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 11px;
  }
  .sub-info { color: var(--text-muted); font-size: 11px; }

  .item-list-box {
    max-height: 160px;
    overflow-y: auto;
    background: #080d18;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 11px;
  }
  .item-list-box.monospace {
    font-family: Consolas, monospace;
    font-size: 10px;
  }
  .list-row {
    padding: 5px 6px;
    border-radius: 4px;
    background: #0d1526;
    border: 1px solid #16243d;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .list-row-main {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* "Why" row: spatial map, thinking stream, drives */
  .insight-grid {
    display: grid;
    grid-template-columns: minmax(300px, 1fr) minmax(0, 1.55fr) 340px;
    gap: 16px;
    padding: 0 16px 16px;
    max-width: 1920px;
    margin: 0 auto;
    align-items: start;
  }
  .insight-grid > .panel-card { min-width: 0; }
  @media (max-width: 1280px) {
    .insight-grid { grid-template-columns: 1fr 1fr; }
    .insight-grid > #thinkPanel { grid-column: 1 / -1; order: -1; }
  }
  @media (max-width: 760px) {
    .insight-grid { grid-template-columns: 1fr; padding: 0 12px 12px; }
  }
  .toggle-label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
  }
  .toggle-label input { margin: 0; padding: 0; accent-color: var(--amber); }
  .toggle-row { display: flex; gap: 12px; align-items: center; }
  .map-canvas-wrap {
    width: 100%;
    max-width: 520px;
    margin: 0 auto;
    aspect-ratio: 1 / 1;
    background: #05080f;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  canvas#mapCanvas { width: 100%; height: 100%; display: block; }
  .map-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    font-size: 10px;
    color: var(--text-muted);
  }
  .legend-swatch {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: -1px;
  }
  .stat-line { font-size: 11px; color: var(--text-secondary); font-family: Consolas, monospace; }
  .stat-line strong { color: var(--text-main); }
  .prompt-details summary {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    text-transform: uppercase;
    cursor: pointer;
  }
  .prompt-details pre {
    margin: 6px 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 10px;
    color: var(--text-secondary);
    background: #080d18;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 8px;
    max-height: 160px;
    overflow-y: auto;
  }
  .goal-line {
    font-size: 11px;
    color: var(--text-secondary);
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 10px;
    word-break: break-word;
  }
  .goal-line strong { color: var(--text-main); }
  .think-split {
    display: grid;
    grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
    gap: 12px;
    align-items: start;
  }
  @media (max-width: 980px) {
    .think-split { grid-template-columns: 1fr; }
  }
  .frame-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    background: #05080f;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  canvas#viewCanvas { width: 100%; height: 100%; display: block; }
  .frame-empty {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 12px;
    font-size: 11px;
    font-style: italic;
    color: var(--text-muted);
  }
  .frame-caption {
    margin-top: 6px;
    font-size: 10px;
    line-height: 1.5;
    color: var(--text-muted);
    font-family: Consolas, monospace;
    word-break: break-word;
  }
  .frame-caption strong { color: var(--text-secondary); }
  .think-log {
    height: 420px;
    overflow-y: auto;
    background: #080d18;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 11px;
  }
  .turn-card {
    border: 1px solid #1c2c4a;
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    background: #0b1222;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .turn-card.origin-autonomous { border-left-color: var(--violet); }
  .turn-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; color: var(--text-main); }
  .turn-badge {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.6px;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(0, 240, 255, 0.12);
    color: var(--accent);
  }
  .origin-autonomous .turn-badge { background: rgba(139, 92, 246, 0.18); color: #c4b5fd; }
  .turn-trigger { font-size: 10px; color: var(--amber); font-family: Consolas, monospace; }
  .turn-time { margin-left: auto; font-size: 10px; color: var(--text-muted); }
  .turn-objective { color: var(--text-secondary); font-style: italic; word-break: break-word; }
  .step {
    padding: 3px 6px;
    border-radius: 4px;
    background: #0d1526;
    word-break: break-word;
    color: var(--text-secondary);
  }
  .step-k {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-right: 4px;
    color: var(--accent);
  }
  .step-muted { color: var(--text-muted); font-size: 10px; }
  .step code {
    font-family: Consolas, monospace;
    font-size: 10px;
    color: #cbd5e1;
  }
  .step-skill.ok .step-k { color: var(--emerald); }
  .step-skill.fail .step-k { color: var(--rose); }
  .step-skill.running .step-k { color: var(--amber); }
  .step-view .step-k { color: #38bdf8; }
  .step-mem .step-k { color: #c4b5fd; }
  .step-warn .step-k { color: var(--rose); }
  .step-say { color: var(--text-main); font-style: italic; }
  .step-link {
    color: var(--accent);
    cursor: pointer;
    text-decoration: underline;
    font-size: 10px;
    margin-left: 4px;
  }
  .turn-final {
    padding: 4px 6px;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: var(--text-main);
    word-break: break-word;
  }
  .turn-final.status-bad {
    background: rgba(244, 63, 94, 0.08);
    border-color: rgba(244, 63, 94, 0.3);
  }
  .turn-final .step-k { color: var(--emerald); }
  .turn-final.status-bad .step-k { color: var(--rose); }
  .pill-row { display: flex; flex-wrap: wrap; gap: 5px; }
  .mode-pill {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: #0d1526;
  }
  .mode-pill.on { color: var(--emerald); border-color: rgba(16, 185, 129, 0.45); }
  .mode-pill.warn { color: var(--amber); border-color: rgba(245, 158, 11, 0.45); }
  .mode-pill.busy { color: #c4b5fd; border-color: rgba(139, 92, 246, 0.5); }
  .kv-list { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; font-size: 11px; }
  .kv-list dt { color: var(--text-muted); font-size: 10px; font-weight: 700; text-transform: uppercase; }
  .kv-list dd { margin: 0; color: var(--text-main); word-break: break-word; }
  .drive-row { display: flex; flex-direction: column; gap: 3px; }
  .drive-head {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    text-transform: uppercase;
  }
  .drive-head .num { font-family: Consolas, monospace; color: var(--text-main); text-transform: none; }
  .drive-track {
    position: relative;
    height: 8px;
    background: #141e30;
    border-radius: 4px;
    overflow: hidden;
  }
  .drive-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--emerald) 0%, var(--accent) 100%);
    border-radius: 4px;
  }
  .drive-fill.urgent { background: linear-gradient(90deg, var(--amber) 0%, #ff8c00 100%); }
  .drive-fill.low { background: linear-gradient(90deg, var(--rose) 0%, #ff0055 100%); }
  .drive-thresh {
    position: absolute;
    top: -1px;
    bottom: -1px;
    width: 2px;
    background: #f1f5f9;
    opacity: 0.75;
  }
</style>
</head>
<body>

<header class="app-header">
  <div class="header-brand">
    <div class="brand-logo">SPROCKET</div>
    <div class="brand-sub">SPROCKET // EMBODIED AI HARNESS DASHBOARD</div>
  </div>
  <div class="header-status-strip">
    <div class="status-pill" id="connPill"><span class="pulse-dot"></span><span id="connLabel">LIVE (0ms)</span></div>
    <div class="status-pill" id="fsmPill">FSM: <strong class="fsm-tag" id="fsmBadge">IDLE</strong></div>
    <div class="status-pill">AUDIO: <strong id="audioBadge">IDLE</strong></div>
    <div class="status-pill">EVENTS: <strong id="busBadge">0</strong></div>
  </div>
  <div class="header-actions">
    <button class="btn btn-accent btn-sm" onclick="launchNativeMujoco()" title="Launch Native Desktop 60 FPS MuJoCo Window">⚡ MuJoCo Sim</button>
    <button class="btn btn-sm" id="muteBtn" onclick="toggleMute()">MUTE: OFF</button>
    <button class="btn btn-sm" onclick="bargeIn()">BARGE-IN</button>
    <button class="btn btn-danger btn-sm" id="estopBtn" onclick="toggleEstop()">E-STOP: OFF</button>
  </div>
</header>

<main class="cockpit-grid">

  <!-- Left Column: Face Expression & Audio Telemetry & Services -->
  <section class="cockpit-col col-left">
    <!-- Live Vector Face Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
          ROBOT FACE (LIVE 60 FPS)
        </div>
        <div class="panel-tag" id="faceThemeLabel">CYAN | IDLE | NEUTRAL</div>
      </div>
      <div class="face-canvas-container">
        <canvas id="faceCanvas" width="800" height="480"></canvas>
      </div>
      <div class="section-subtitle">EMOTION PALETTE</div>
      <div class="chips-grid" id="emotionChips">
        <button class="chip-btn" data-emo="neutral" onclick="setEmotion('neutral')">Neutral</button>
        <button class="chip-btn" data-emo="happy" onclick="setEmotion('happy')">Happy</button>
        <button class="chip-btn" data-emo="excited" onclick="setEmotion('excited')">Excited</button>
        <button class="chip-btn" data-emo="curious" onclick="setEmotion('curious')">Curious</button>
        <button class="chip-btn" data-emo="proud" onclick="setEmotion('proud')">Proud</button>
        <button class="chip-btn" data-emo="shy" onclick="setEmotion('shy')">Shy</button>
        <button class="chip-btn" data-emo="love" onclick="setEmotion('love')">Love</button>
        <button class="chip-btn" data-emo="sad" onclick="setEmotion('sad')">Sad</button>
        <button class="chip-btn" data-emo="angry" onclick="setEmotion('angry')">Angry</button>
        <button class="chip-btn" data-emo="grumpy" onclick="setEmotion('grumpy')">Grumpy</button>
        <button class="chip-btn" data-emo="surprised" onclick="setEmotion('surprised')">Surprised</button>
        <button class="chip-btn" data-emo="confused" onclick="setEmotion('confused')">Confused</button>
        <button class="chip-btn" data-emo="sleepy" onclick="setEmotion('sleepy')">Sleepy</button>
        <button class="chip-btn" data-emo="scared" onclick="setEmotion('scared')">Scared</button>
      </div>
      <div class="control-row">
        <div class="select-group">
          <label>Gaze Direction</label>
          <select id="gazeSelect" onchange="setGaze(this.value)">
            <option value="center">Center</option>
            <option value="speaker">Speaker</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="up">Up</option>
            <option value="down">Down</option>
          </select>
        </div>
        <div class="select-group">
          <label>Eyes Theme</label>
          <select id="themeSelect" onchange="setTheme(this.value)">
            <option value="cyan">Cyan</option>
            <option value="green">Green</option>
            <option value="amber">Amber</option>
            <option value="blue">Blue</option>
            <option value="violet">Violet</option>
            <option value="coral">Coral</option>
            <option value="red">Red</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Speech & Audio Telemetry Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          SPEECH &amp; AUDIO TELEMETRY
        </div>
        <div class="panel-tag" id="audioStateTag">AUDIO IDLE</div>
      </div>
      <div class="speech-waveform" id="speechWaveform">
        <span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span>
        <span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span>
        <span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span>
      </div>
      <div class="speech-box">
        <div class="speech-item">
          <span class="speech-label">LAST HEARD:</span>
          <span class="speech-val" id="lastAsrText">—</span>
        </div>
        <div class="speech-item">
          <span class="speech-label">LAST SPOKEN:</span>
          <span class="speech-val" id="lastSpokenText">—</span>
        </div>
      </div>
      <div class="section-subtitle">SOUND EFFECTS LAUNCHER</div>
      <div class="chips-grid">
        <button class="chip-btn" onclick="playSfx('ack_chirp')">Chirp</button>
        <button class="chip-btn" onclick="playSfx('happy_trill')">Trill</button>
        <button class="chip-btn" onclick="playSfx('curious_chirp')">Curious?</button>
        <button class="chip-btn" onclick="playSfx('giggle')">Giggle</button>
        <button class="chip-btn" onclick="playSfx('purr')">Purr</button>
        <button class="chip-btn" onclick="playSfx('sad_whimper')">Whimper</button>
        <button class="chip-btn" onclick="playSfx('angry_grumble')">Grumble</button>
      </div>
    </div>

    <!-- System Services Health -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">SYSTEM SERVICES STATUS</div>
        <div class="panel-tag">SUPERVISOR</div>
      </div>
      <div class="services-grid" id="servicesGrid">
        <div class="svc-item"><span class="svc-name">Body HAL</span><span class="svc-status">ONLINE</span></div>
        <div class="svc-item"><span class="svc-name">Executor</span><span class="svc-status">ONLINE</span></div>
        <div class="svc-item"><span class="svc-name">Face Bridge</span><span class="svc-status">ONLINE</span></div>
        <div class="svc-item"><span class="svc-name">Audio</span><span class="svc-status" id="svcAudioStatus">ONLINE</span></div>
        <div class="svc-item"><span class="svc-name">Autonomy</span><span class="svc-status" id="svcAutoStatus">ONLINE</span></div>
        <div class="svc-item"><span class="svc-name">Dashboard</span><span class="svc-status">ONLINE</span></div>
      </div>
    </div>
  </section>

  <!-- Center Column: 1st-Person Head Camera & Telemetry Cockpit & 2D Tactical Radar -->
  <section class="cockpit-col col-center">
    <!-- 1st-Person Head Camera Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
          1ST-PERSON HEAD CAMERA VISION
        </div>
        <div class="panel-tag">LFM3-VL-2B EMBODIED VISION</div>
      </div>
      <div class="camera-viewport">
        <img id="headCamImg" alt="1st-person head camera view" />
        <div class="cam-overlay-badge top-left">HEAD CAMERA (Z=286mm)</div>
        <div class="cam-overlay-badge top-right" id="camTiltBadge">TILT: 0°</div>
      </div>
      <div class="vision-hud">
        <div class="vision-summary" id="camHud">Camera: Scanning scene...</div>
        <div class="vision-detections" id="camDetections"></div>
      </div>
    </div>

    <!-- Kinematics & Sensors Telemetry Cockpit -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          REALTIME KINEMATICS &amp; SENSORS
        </div>
        <div class="panel-tag" id="lastActionTag">ACTION: IDLE</div>
      </div>

      <div class="telemetry-grid">
        <!-- Gauge 1: Front Proximity (ToF) -->
        <div class="telem-item telem-large">
          <div class="telem-header">
            <span class="telem-name">FRONT PROXIMITY (ToF)</span>
            <span class="telem-badge" id="tofStatusBadge">SAFE</span>
          </div>
          <div class="telem-main-val"><span id="sensorTofVal">1500</span> <span class="telem-unit">mm</span></div>
          <div class="meter-bar-track">
            <div class="meter-bar-fill" id="tofMeterFill" style="width: 75%"></div>
          </div>
          <div class="meter-scale"><span>0mm (Stop: 120mm)</span><span>1000mm</span><span>2000mm</span></div>
        </div>

        <!-- Gauge 2: Heading & Compass -->
        <div class="telem-item">
          <div class="telem-header">
            <span class="telem-name">HEADING</span>
            <span class="telem-badge" id="cardinalHeading">N</span>
          </div>
          <div class="telem-main-val"><span id="poseHeadingVal">0.0</span><span class="telem-unit">°</span></div>
          <div class="telem-sub">IMU Yaw: <strong id="imuYawVal">0.0°</strong></div>
        </div>

        <!-- Gauge 3: Head Tilt Pitch -->
        <div class="telem-item">
          <div class="telem-header">
            <span class="telem-name">HEAD TILT PITCH</span>
            <span class="telem-badge">SERVO</span>
          </div>
          <div class="telem-main-val"><span id="poseTiltVal">0.0</span><span class="telem-unit">°</span></div>
          <div class="meter-bar-track">
            <div class="meter-bar-fill meter-amber" id="tiltMeterFill" style="width: 41%"></div>
          </div>
          <div class="meter-scale"><span>-25° (Down)</span><span>0°</span><span>+35° (Up)</span></div>
        </div>

        <!-- Gauge 4: Odometry Pose Coordinates -->
        <div class="telem-item">
          <div class="telem-header">
            <span class="telem-name">ODOMETRY POSITION</span>
            <span class="telem-badge">ODOM</span>
          </div>
          <div class="telem-dual-val">
            <div><span class="coord-label">X:</span> <strong id="poseXVal">0.0</strong> <span class="telem-unit">mm</span></div>
            <div><span class="coord-label">Y:</span> <strong id="poseYVal">0.0</strong> <span class="telem-unit">mm</span></div>
          </div>
          <div class="telem-sub">Ticks L/R: <strong id="sensorTicksVal">0 / 0</strong></div>
        </div>

        <!-- Gauge 5: Velocities -->
        <div class="telem-item">
          <div class="telem-header">
            <span class="telem-name">VELOCITIES</span>
            <span class="telem-badge">HAL</span>
          </div>
          <div class="telem-dual-val">
            <div><span class="coord-label">LIN:</span> <strong id="linearVelVal">0.0</strong> <span class="telem-unit">mm/s</span></div>
            <div><span class="coord-label">ANG:</span> <strong id="angularVelVal">0.0</strong> <span class="telem-unit">°/s</span></div>
          </div>
          <div class="telem-sub">IMU Pitch: <strong id="imuPitchVal">0.0°</strong></div>
        </div>

        <!-- Gauge 6: Safety Sensors Status -->
        <div class="telem-item telem-large">
          <div class="telem-header">
            <span class="telem-name">HARDWARE SAFETY SENSORS</span>
            <span class="telem-badge">REFLEX</span>
          </div>
          <div class="safety-pills">
            <span class="safety-chip chip-safe" id="cliffChip">CLIFF: SAFE</span>
            <span class="safety-chip chip-safe" id="estopChip">E-STOP: CLEAR</span>
            <span class="safety-chip chip-safe" id="pickupChip">GROUNDED</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 2D Tactical Ego-Centric Radar Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
          2D TACTICAL SPATIAL RADAR
        </div>
        <div class="panel-tag"><span id="obsCountTag">0</span> TARGETS</div>
      </div>
      <div class="radar-split">
        <div class="radar-canvas-wrap">
          <canvas id="radarCanvas" width="440" height="440"></canvas>
        </div>
        <div class="radar-legend-list">
          <div class="legend-header">ACTIVE SPATIAL OBSERVATIONS</div>
          <div class="obs-list" id="obsList">
            <div class="empty-hint">Scanning surroundings...</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Right Column: Conversation & Autonomy & Motion Deck & Memory -->
  <section class="cockpit-col col-right">
    <!-- Conversation & Embodied Autonomy Panel -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          CONVERSATION &amp; AUTONOMY
        </div>
        <div class="panel-tag" id="latencyTag">LLM: 0 ms</div>
      </div>

      <div class="autonomy-bar">
        <div class="autonomy-status-line">
          <span class="status-indicator" id="autoIndicator"></span>
          <span class="autonomy-text" id="initiativeState">Autonomy active · Observing room</span>
        </div>
        <div class="btn-group-row">
          <button class="btn btn-sm" id="btnAutoOn" onclick="setAutonomy({enabled:true,paused:false})">Autonomy On</button>
          <button class="btn btn-sm" id="btnStay" onclick="setAutonomy({paused:true})">Stay Here</button>
          <button class="btn btn-sm" id="btnQuiet" onclick="setAutonomy({quiet:true})">Quiet</button>
          <button class="btn btn-sm" id="btnShare" onclick="setAutonomy({quiet:false})">Share</button>
        </div>
      </div>

      <div class="chat-container" id="chatBox"></div>

      <form class="chat-form" onsubmit="sendChat(event)">
        <input id="chatInput" type="text" placeholder="Command Sprocket (e.g. 'What do you see?', 'Dance!')..." autocomplete="off" />
        <button type="submit" class="btn btn-accent btn-sm">Send</button>
      </form>

      <div class="quick-prompts">
        <button class="quick-chip" onclick="quickMsg('What do you see in front of you?')">"What do you see?"</button>
        <button class="quick-chip" onclick="quickMsg('Go to the red ball')">"Go to red ball"</button>
        <button class="quick-chip" onclick="quickMsg('Look at the blue cube')">"Look at blue cube"</button>
        <button class="quick-chip" onclick="quickMsg('Do a happy dance!')">"Do a dance!"</button>
        <button class="quick-chip" onclick="quickMsg('What was behind you?')">"What was behind you?"</button>
      </div>
    </div>

    <!-- Motion & Actuation Controls Deck -->
    <div class="panel-card">
      <div class="panel-header">
        <div class="panel-title">
          <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 19 21 12 17 5 21 12 2"/></svg>
          MOTION &amp; ACTUATION CONTROLS
        </div>
        <div class="panel-tag">MANUAL OVERRIDE</div>
      </div>

      <div class="motion-controls-grid">
        <div class="dpad-cluster">
          <div class="dpad-row">
            <button class="dpad-btn" onclick="runAction('drive',{direction:'forward',distance_mm:250})">▲ Fwd 250mm</button>
          </div>
          <div class="dpad-row">
            <button class="dpad-btn" onclick="runAction('spin',{angle_deg:30})">◀ Turn 30°</button>
            <button class="dpad-btn dpad-stop" onclick="runAction('stop',{})">STOP</button>
            <button class="dpad-btn" onclick="runAction('spin',{angle_deg:-30})">Turn 30° ▶</button>
          </div>
          <div class="dpad-row">
            <button class="dpad-btn" onclick="runAction('drive',{direction:'backward',distance_mm:200})">▼ Back 200mm</button>
          </div>
        </div>

        <div class="actuator-cluster">
          <div class="actuator-group">
            <label>HEAD TILT SERVO</label>
            <div class="btn-group-col">
              <button class="btn btn-sm" onclick="runAction('tilt_head',{angle_deg:15})">▲ +15°</button>
              <button class="btn btn-sm" onclick="runAction('tilt_head',{angle_deg:0})">━ Level</button>
              <button class="btn btn-sm" onclick="runAction('tilt_head',{angle_deg:-15})">▼ -15°</button>
            </div>
          </div>
          <div class="actuator-group">
            <label>EXPRESSIVE TRICKS</label>
            <div class="btn-group-col">
              <button class="btn btn-sm" onclick="runAction('perform_trick',{trick:'dance'})">💃 Dance</button>
              <button class="btn btn-sm" onclick="runAction('perform_trick',{trick:'nod'})">👍 Nod</button>
              <button class="btn btn-sm" onclick="runAction('perform_trick',{trick:'wiggle'})">✨ Wiggle</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Persistent Memory & Event Stream Panel -->
    <div class="panel-card">
      <div class="tabs-header">
        <button class="tab-btn active" id="tabMemBtn" onclick="switchTab('mem')">Long-Term Memory (<span id="pendingCount">0</span>)</button>
        <button class="tab-btn" id="tabEvtBtn" onclick="switchTab('evt')">Event Stream</button>
      </div>

      <div class="tab-content" id="tabMemContent">
        <div class="tab-action-bar">
          <span class="sub-info" id="tombstoneInfo">Tombstones: none</span>
          <button class="btn btn-sm btn-accent" onclick="runConsolidate()">Consolidate</button>
        </div>
        <div class="item-list-box" id="memList"></div>
      </div>

      <div class="tab-content" id="tabEvtContent" style="display:none;">
        <div class="tab-action-bar">
          <span class="sub-info" id="dropInfo">Dropped: 0</span>
        </div>
        <div class="item-list-box monospace" id="eventList"></div>
      </div>
    </div>
  </section>

</main>

<!-- "Why did it do that?" row: the robot's own map, its reasoning stream, and its drives -->
<section class="insight-grid">
  <!-- Spatial map (robot's own, odometry frame) -->
  <div class="panel-card" id="mapPanel">
    <div class="panel-header">
      <div class="panel-title">
        <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
        SPATIAL MAP (ROBOT'S OWN)
      </div>
      <div class="toggle-row">
        <label class="toggle-label" title="Shade floor the camera has looked at"><input type="checkbox" id="coverageToggle" checked /> COVERAGE</label>
        <label class="toggle-label" title="Overlay simulator ground truth (not known to the robot)"><input type="checkbox" id="truthToggle" /> SIM TRUTH</label>
      </div>
    </div>
    <div class="map-canvas-wrap">
      <canvas id="mapCanvas" width="520" height="520"></canvas>
    </div>
    <div class="map-legend">
      <span><span class="legend-swatch" style="background:#38bdf8"></span>tracked object (fades with age, dashed = from earlier session)</span>
      <span><span class="legend-swatch" style="background:#f43f5e"></span>ToF occupied</span>
      <span><span class="legend-swatch" style="background:rgba(0,240,255,0.25)"></span>camera coverage</span>
      <span><span class="legend-swatch" style="background:#f59e0b"></span>place</span>
      <span><span class="legend-swatch" style="border:1px dashed #94a3b8"></span>sim truth</span>
    </div>
    <div class="stat-line" id="mapStats">Loading map...</div>
    <div class="item-list-box" id="mapObjList"><div class="empty-hint">No tracked objects yet</div></div>
    <details class="prompt-details">
      <summary>What the model is told about the world</summary>
      <pre id="mapPromptText"></pre>
    </details>
  </div>

  <!-- Thinking stream -->
  <div class="panel-card" id="thinkPanel">
    <div class="panel-header">
      <div class="panel-title">
        <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.74V17h8v-2.26A7 7 0 0 0 12 2z"/></svg>
        THINKING (WHY IT ACTED)
      </div>
      <div class="panel-tag" id="thinkStatusTag">IDLE</div>
    </div>
    <div class="goal-line" id="thinkGoal">No goal yet.</div>
    <div class="think-split">
      <div>
        <div class="frame-wrap">
          <canvas id="viewCanvas" width="512" height="384"></canvas>
          <div class="frame-empty" id="viewEmpty">No camera view yet: the robot has not looked (observe / look_around).</div>
        </div>
        <div class="frame-caption" id="viewCaption"></div>
      </div>
      <div class="think-log" id="thinkLog"><div class="empty-hint">Waiting for the first turn...</div></div>
    </div>
  </div>

  <!-- Drives & autonomy -->
  <div class="panel-card" id="drivesPanel">
    <div class="panel-header">
      <div class="panel-title">
        <svg class="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"/></svg>
        DRIVES &amp; AUTONOMY
      </div>
      <div class="panel-tag" id="autoModeTag">—</div>
    </div>
    <div class="pill-row" id="autoPills"></div>
    <div class="btn-group-row">
      <button class="btn btn-sm" onclick="setAutonomy({enabled:true,paused:false})">On / resume</button>
      <button class="btn btn-sm" onclick="setAutonomy({enabled:false})">Off</button>
      <button class="btn btn-sm" onclick="setAutonomy({paused:true})">Stay here</button>
      <button class="btn btn-sm" onclick="setAutonomy({quiet:true})">Quiet</button>
      <button class="btn btn-sm" onclick="setAutonomy({quiet:false})">Share</button>
    </div>
    <dl class="kv-list" id="autoDetails"></dl>
    <div class="section-subtitle">DRIVES (bar = level, white line = threshold)</div>
    <div id="driveBars" style="display:flex;flex-direction:column;gap:8px;"></div>
    <div class="section-subtitle">BATTERY</div>
    <div id="batteryBox"></div>
  </div>
</section>

<script>
let targetFace = null;
let animFace = null;
let latestWorld = null;
let latestObservations = [];
let estopActive = false;
let audioMuted = false;
let pendingUserMsg = null;
let isRefreshing = false;
let lastCamUrl = '';
const htmlCache = {};
// Panels poll / redraw only while on screen (IntersectionObserver), to keep a Pi's CPU free.
const panelVisible = {face: true, map: true, think: true, drives: true};

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function setHtmlIfChanged(id, html, scrollBottom = false) {
  if (htmlCache[id] === html) return;
  htmlCache[id] = html;
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = html;
  if (scrollBottom) el.scrollTop = el.scrollHeight;
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  return await r.json();
}

async function submitUserText(txt) {
  if (!txt) return;
  pendingUserMsg = txt;
  if (targetFace) {
    targetFace.activity = 'thinking';
    targetFace.gaze_target = 'speaker';
  }
  renderChatBox(window._lastTurns || []);
  try {
    await apiPost('/api/message', {text: txt});
  } finally {
    pendingUserMsg = null;
  }
  await refreshState();
}

async function sendChat(e) {
  e.preventDefault();
  const inp = document.getElementById('chatInput');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  await submitUserText(txt);
}

async function quickMsg(txt) {
  await submitUserText(txt);
}

async function setAutonomy(mode) {
  await apiPost('/api/autonomy', mode);
  await refreshState();
}

async function setEmotion(emotion) {
  await apiPost('/api/expression', {emotion, intensity: 0.9});
  await refreshState();
}

async function setGaze(gaze_target) {
  await apiPost('/api/expression', {gaze_target});
  await refreshState();
}

async function setTheme(theme) {
  await apiPost('/api/expression', {theme});
  await refreshState();
}

async function runAction(tool, args) {
  await apiPost('/api/action', {tool, args});
  await refreshState();
}

async function toggleEstop() {
  estopActive = !estopActive;
  await apiPost('/api/estop', {active: estopActive});
  await refreshState();
}

async function launchNativeMujoco() {
  await apiPost('/api/sandbox', {op: 'launch_viewer'});
}

async function forgetMemory(topic) {
  await apiPost('/api/memory/forget', {topic});
  await refreshState();
}

async function runConsolidate() {
  await apiPost('/api/memory/consolidate', {});
  await refreshState();
}

async function playSfx(sound) {
  await apiPost('/api/audio/sfx', {sound});
  await refreshState();
}

async function toggleMute() {
  audioMuted = !audioMuted;
  await apiPost('/api/audio/mute', {muted: audioMuted});
  await refreshState();
}

async function bargeIn() {
  await apiPost('/api/audio/barge_in', {});
  await refreshState();
}

function switchTab(tab) {
  const isMem = tab === 'mem';
  document.getElementById('tabMemBtn').className = isMem ? 'tab-btn active' : 'tab-btn';
  document.getElementById('tabEvtBtn').className = isMem ? 'tab-btn' : 'tab-btn active';
  document.getElementById('tabMemContent').style.display = isMem ? 'block' : 'none';
  document.getElementById('tabEvtContent').style.display = isMem ? 'none' : 'block';
}

function getCardinal(deg) {
  const norm = ((deg % 360) + 360) % 360;
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const idx = Math.round(norm / 45) % 8;
  return dirs[idx];
}

// 60 FPS Face Rendering with Lerp
function drawFace(f) {
  if (!f) return;
  const c = document.getElementById('faceCanvas');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#060910';
  ctx.fillRect(0, 0, W, H);

  const rgb = f.color_rgb || [0, 228, 255];
  const color = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  const hlColor = `rgb(${Math.min(255,rgb[0]+95)},${Math.min(255,rgb[1]+95)},${Math.min(255,rgb[2]+95)})`;

  const gx = (f.gaze_x || 0) * 42;
  const gy = (f.gaze_y || 0) * 28;
  const cy = H * 0.44 + gy;
  const lcx = W * 0.31 + gx;
  const rcx = W * 0.69 + gx;
  const ew = 180 * (f.pupil_scale || 1.0);

  function drawEye(cx, openness, browDeg, isLeft) {
    const eh = Math.max(10, 190 * openness);
    const left = cx - ew / 2, top = cy - eh / 2;

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 24;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(left, top, ew, eh, Math.min(ew, eh) * 0.36);
    ctx.fill();
    ctx.restore();

    if (eh > 36) {
      ctx.fillStyle = hlColor;
      ctx.beginPath();
      ctx.arc(cx - ew / 5, cy - eh / 5, Math.max(6, Math.min(16, eh / 9)), 0, Math.PI * 2);
      ctx.fill();
    }

    if ((f.lower_lid_squint || 0) > 0.02 && eh > 24) {
      const sh = eh * Math.min(0.65, f.lower_lid_squint);
      ctx.fillStyle = '#060910';
      ctx.beginPath();
      ctx.ellipse(cx, (cy + eh / 2) - sh / 2 + 6, ew / 2 + 12, sh / 2 + 8, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    if (Math.abs(browDeg) > 1.0 && eh > 20) {
      const clampedBrow = Math.max(-30, Math.min(30, browDeg));
      const slope = Math.tan(clampedBrow * Math.PI / 180) * (ew * 0.5);
      const innerDrop = slope;
      const outerDrop = -slope;
      const leftDrop = isLeft ? outerDrop : innerDrop;
      const rightDrop = isLeft ? innerDrop : outerDrop;
      ctx.fillStyle = '#060910';
      ctx.beginPath();
      ctx.moveTo(left - 8, top - 14);
      ctx.lineTo(cx + ew / 2 + 8, top - 14);
      ctx.lineTo(cx + ew / 2 + 8, top + Math.max(0, rightDrop + 18));
      ctx.lineTo(left - 8, top + Math.max(0, leftDrop + 18));
      ctx.closePath();
      ctx.fill();
    }
  }

  drawEye(lcx, f.left_openness ?? 1.0, f.left_brow_deg || 0, true);
  drawEye(rcx, f.right_openness ?? 1.0, f.right_brow_deg || 0, false);

  const mx = W / 2 + gx * 0.4, my = H * 0.77;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 6;
  if ((f.mouth_open || 0) > 0.18) {
    const mh = Math.max(8, 68 * f.mouth_open);
    ctx.beginPath();
    ctx.roundRect(mx - 55, my - mh / 2, 110, mh, mh / 2);
    ctx.fill();
  } else {
    const curve = (f.mouth_curve || 0.2) * 28;
    ctx.beginPath();
    ctx.moveTo(mx - 55, my);
    ctx.quadraticCurveTo(mx, my + curve, mx + 55, my);
    ctx.stroke();
  }
}

// 2D Tactical Ego-Centric Radar Display
function drawRadar(world, obsList) {
  const c = document.getElementById('radarCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const cx = W / 2, cy = H / 2;
  const maxRangeMm = 1600.0;
  const scale = (W / 2 - 25) / maxRangeMm;

  ctx.fillStyle = '#05080f';
  ctx.fillRect(0, 0, W, H);

  // Concentric distance rings
  const rings = [400, 800, 1200, 1600];
  ctx.lineWidth = 1;
  rings.forEach((r, idx) => {
    ctx.strokeStyle = idx === rings.length - 1 ? 'rgba(0, 240, 255, 0.22)' : 'rgba(0, 240, 255, 0.08)';
    ctx.beginPath();
    ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(148, 163, 184, 0.45)';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${r}mm`, cx + 4, cy - r * scale + 11);
  });

  // Crosshairs
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
  ctx.beginPath();
  ctx.moveTo(cx, 12); ctx.lineTo(cx, H - 12);
  ctx.moveTo(12, cy); ctx.lineTo(W - 12, cy);
  ctx.stroke();

  // Cardinal direction labels (Ego-frame: Robot faces UP)
  ctx.fillStyle = 'rgba(0, 240, 255, 0.7)';
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('FRONT', cx, 18);
  ctx.fillText('BEHIND', cx, H - 8);
  ctx.fillText('LEFT', 22, cy + 4);
  ctx.fillText('RIGHT', W - 22, cy + 4);

  // Front ToF distance beam cone
  const tofDist = Math.min(1600, (world && world.sensors && world.sensors.front_tof_mm) || 1500);
  const tofRadius = tofDist * scale;
  const beamAngle = 18 * Math.PI / 180; // +/- 18 deg forward cone
  const grad = ctx.createRadialGradient(cx, cy, 6, cx, cy, tofRadius);
  if (tofDist <= 125) {
    grad.addColorStop(0, 'rgba(244, 63, 94, 0.45)');
    grad.addColorStop(1, 'rgba(244, 63, 94, 0.05)');
  } else if (tofDist <= 500) {
    grad.addColorStop(0, 'rgba(245, 158, 11, 0.35)');
    grad.addColorStop(1, 'rgba(245, 158, 11, 0.05)');
  } else {
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.28)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.03)');
  }
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, tofRadius, -Math.PI / 2 - beamAngle, -Math.PI / 2 + beamAngle);
  ctx.closePath();
  ctx.fill();

  // Draw active spatial observations
  (obsList || []).forEach(obs => {
    const angleRad = (obs.relative_bearing_deg - 90) * Math.PI / 180;
    const distPx = Math.min(W / 2 - 20, obs.distance_mm * scale);
    const ox = cx + Math.cos(angleRad) * distPx;
    const oy = cy + Math.sin(angleRad) * distPx;

    const ageAlpha = Math.max(0.35, 1.0 - ((obs.age_s || 0) / 30.0));
    ctx.fillStyle = `rgba(0, 240, 255, ${ageAlpha})`;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(ox, oy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = `rgba(241, 245, 249, ${ageAlpha})`;
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${obs.label} (${Math.round(obs.distance_mm)}mm)`, ox + 8, oy + 3);
  });

  // Draw Robot Chassis at Center
  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(0, -14); // Nose forward
  ctx.lineTo(9, 9);
  ctx.lineTo(-9, 9);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function renderChatBox(turns) {
  window._lastTurns = turns;
  let html = (turns || []).map(t => {
    const userLine = t.user ? `<div class="chat-bubble chat-u"><strong>You:</strong> ${esc(t.user)}</div>` : '';
    const label = t.origin === 'autonomous' ? 'Sprocket (Autonomous)' : 'Sprocket';
    const tools = (t.actions || []).map(a => `${a.tool}${a.ok === false ? ' (failed)' : ''}`).join(', ') || 'none';
    return `${userLine}<div class="chat-bubble chat-r">` +
      `<div><strong>${label}:</strong> ${esc(t.say)}</div>` +
      `<div class="chat-meta">[${esc(t.status || 'completed')} · ${esc(t.steps || 0)} step(s) | tools: ${esc(tools)} | emotion:${esc(t.emotion)} | sound:${esc(t.sound)} | ${esc(t.backend)} ${esc(t.latency_ms)}ms]</div></div>`;
  }).join('');
  if (pendingUserMsg) {
    html += `<div class="chat-bubble chat-u"><strong>You:</strong> ${esc(pendingUserMsg)}</div>` +
            `<div class="chat-bubble chat-r" style="color:var(--accent)"><strong>Sprocket:</strong> <em>*thinking...*</em></div>`;
  }
  setHtmlIfChanged('chatBox', html, true);
}

function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

function stepClientAnimation() {
  if (targetFace) {
    if (!animFace) {
      animFace = {...targetFace};
    } else {
      const alpha = 0.24;
      animFace.theme = targetFace.theme;
      animFace.color_rgb = targetFace.color_rgb;
      animFace.activity = targetFace.activity;
      animFace.emotion = targetFace.emotion;
      animFace.gaze_x = lerp(animFace.gaze_x || 0, targetFace.gaze_x || 0, alpha);
      animFace.gaze_y = lerp(animFace.gaze_y || 0, targetFace.gaze_y || 0, alpha);
      animFace.left_openness = lerp(animFace.left_openness ?? 1, targetFace.left_openness ?? 1, alpha);
      animFace.right_openness = lerp(animFace.right_openness ?? 1, targetFace.right_openness ?? 1, alpha);
      animFace.left_brow_deg = lerp(animFace.left_brow_deg || 0, targetFace.left_brow_deg || 0, alpha);
      animFace.right_brow_deg = lerp(animFace.right_brow_deg || 0, targetFace.right_brow_deg || 0, alpha);
      animFace.lower_lid_squint = lerp(animFace.lower_lid_squint || 0, targetFace.lower_lid_squint || 0, alpha);
      animFace.pupil_scale = lerp(animFace.pupil_scale || 1, targetFace.pupil_scale || 1, alpha);
      animFace.mouth_open = lerp(animFace.mouth_open || 0.04, targetFace.mouth_open || 0.04, 0.35);
      animFace.mouth_curve = lerp(animFace.mouth_curve || 0.2, targetFace.mouth_curve || 0.2, alpha);
    }
    if (panelVisible.face) drawFace(animFace);  // no per-frame canvas work while scrolled away
  }
  requestAnimationFrame(stepClientAnimation);
}

async function refreshState() {
  if (isRefreshing) return;
  isRefreshing = true;
  const t0 = performance.now();
  try {
    const stResp = await fetch('/api/state');
    const dt = Math.round(performance.now() - t0);
    const st = await stResp.json();

    // Connection latency
    document.getElementById('connLabel').textContent = `LIVE (${dt}ms)`;
    document.getElementById('connPill').className = 'status-pill status-live';

    targetFace = st.face;
    latestWorld = st.world;
    const p = st.world.pose;
    const s = st.world.sensors;

    // FSM State
    const fsmState = (st.fsm_state || 'idle').toUpperCase();
    document.getElementById('fsmBadge').textContent = `${fsmState} (${st.idle_seconds}s)`;
    const fsmPill = document.getElementById('fsmPill');
    if (fsmPill) {
      if (fsmState === 'THINKING') fsmPill.style.color = 'var(--violet)';
      else if (fsmState === 'SPEAKING') fsmPill.style.color = 'var(--emerald)';
      else if (fsmState === 'ACTING') fsmPill.style.color = 'var(--amber)';
      else fsmPill.style.color = 'var(--accent)';
    }

    // Face labels & Highlight active emotion button
    document.getElementById('faceThemeLabel').textContent = `${st.face.theme} | ${st.face.activity} | ${st.face.emotion}`;
    document.querySelectorAll('#emotionChips .chip-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.emo === st.face.emotion);
    });

    // Audio & Speech Telemetry
    if (st.audio) {
      audioMuted = !!st.audio.muted;
      document.getElementById('muteBtn').textContent = `MUTE: ${audioMuted ? 'ON' : 'OFF'}`;
      document.getElementById('audioBadge').textContent = `${st.audio.is_speaking ? 'SPEAKING' : 'IDLE'} (SFX:${st.audio.sfx_played_count} TTS:${st.audio.tts_spoken_count})`;
      document.getElementById('audioStateTag').textContent = st.audio.is_speaking ? 'SPEAKING (TTS ACTIVE)' : (audioMuted ? 'MUTED' : 'AUDIO READY');
      const waveEl = document.getElementById('speechWaveform');
      if (waveEl) waveEl.classList.toggle('active', !!st.audio.is_speaking);
      document.getElementById('lastAsrText').textContent = st.audio.last_asr_text ? `"${st.audio.last_asr_text}"` : '—';
      document.getElementById('lastSpokenText').textContent = st.audio.last_spoken_text ? `"${st.audio.last_spoken_text}"` : '—';
    }

    // Bus stats
    document.getElementById('busBadge').textContent = st.bus_stats ? st.bus_stats.published_count : 0;
    document.getElementById('dropInfo').textContent = `Dropped: ${st.bus_stats ? st.bus_stats.dropped_count : 0}`;

    // Head Camera & Visual Grounding
    if (st.sandbox && st.sandbox.camera) {
      const cam = st.sandbox.camera;
      if (cam.data_url && cam.data_url !== lastCamUrl) {
        lastCamUrl = cam.data_url;
        const camEl = document.getElementById('headCamImg');
        if (camEl) camEl.src = cam.data_url;
      }
      document.getElementById('camTiltBadge').textContent = `TILT: ${Math.round(p.head_tilt_deg)}°`;
      document.getElementById('camHud').textContent = cam.summary || 'Camera: Scanning visual scene...';
      const detsHtml = (cam.visible_objects || []).map(d =>
        `<span class="vis-pill" style="border-color:${esc(d.color)}"><strong>${esc(d.label)}</strong> (${esc(d.point_norm[0])},${esc(d.point_norm[1])}) ${esc(d.distance_mm)}mm ${esc(d.direction)}</span>`
      ).join('');
      setHtmlIfChanged('camDetections', detsHtml);
    }

    // Kinematics & Gauges
    const tofVal = Math.round(s.front_tof_mm);
    document.getElementById('sensorTofVal').textContent = tofVal;
    const tofPct = Math.min(100, Math.max(0, (tofVal / 2000) * 100));
    const tofFill = document.getElementById('tofMeterFill');
    const tofBadge = document.getElementById('tofStatusBadge');
    if (tofFill && tofBadge) {
      tofFill.style.width = `${tofPct}%`;
      if (tofVal <= 125) {
        tofFill.className = 'meter-bar-fill meter-danger';
        tofBadge.textContent = 'OBSTACLE STOP';
        tofBadge.className = 'telem-badge status-danger';
      } else if (tofVal <= 500) {
        tofFill.className = 'meter-bar-fill meter-amber';
        tofBadge.textContent = 'CAUTION';
        tofBadge.className = 'telem-badge status-warning';
      } else {
        tofFill.className = 'meter-bar-fill';
        tofBadge.textContent = 'SAFE';
        tofBadge.className = 'telem-badge';
      }
    }

    // Heading & Compass
    const headingDeg = Math.round(p.heading_deg * 10) / 10;
    document.getElementById('poseHeadingVal').textContent = headingDeg.toFixed(1);
    document.getElementById('cardinalHeading').textContent = getCardinal(headingDeg);
    document.getElementById('imuYawVal').textContent = `${(s.imu_yaw_deg || 0).toFixed(1)}°`;

    // Head Tilt Pitch
    const tiltDeg = Math.round(p.head_tilt_deg * 10) / 10;
    document.getElementById('poseTiltVal').textContent = tiltDeg.toFixed(1);
    const tiltPct = Math.min(100, Math.max(0, ((tiltDeg - (-25)) / 60) * 100));
    const tiltFill = document.getElementById('tiltMeterFill');
    if (tiltFill) tiltFill.style.width = `${tiltPct}%`;

    // Odometry Pose & Ticks
    document.getElementById('poseXVal').textContent = Math.round(p.x_mm);
    document.getElementById('poseYVal').textContent = Math.round(p.y_mm);
    document.getElementById('sensorTicksVal').textContent = `${s.wheel_ticks_left || 0} / ${s.wheel_ticks_right || 0}`;

    // Velocities & IMU
    document.getElementById('linearVelVal').textContent = Math.round(p.linear_vel_mm_s || 0);
    document.getElementById('angularVelVal').textContent = Math.round(p.angular_vel_deg_s || 0);
    document.getElementById('imuPitchVal').textContent = `${(s.imu_pitch_deg || 0).toFixed(1)}°`;

    // Safety Interlocks
    estopActive = !!s.estop_active;
    const estopBtn = document.getElementById('estopBtn');
    if (estopBtn) {
      estopBtn.textContent = `E-STOP: ${estopActive ? 'ON' : 'OFF'}`;
      estopBtn.classList.toggle('active', estopActive);
    }
    const estopChip = document.getElementById('estopChip');
    if (estopChip) {
      estopChip.textContent = estopActive ? 'E-STOP: ACTIVE' : 'E-STOP: CLEAR';
      estopChip.className = estopActive ? 'safety-chip chip-alert' : 'safety-chip chip-safe';
    }
    const cliffChip = document.getElementById('cliffChip');
    if (cliffChip) {
      cliffChip.textContent = s.cliff_triggered ? 'CLIFF: TRIGGERED' : 'CLIFF: SAFE';
      cliffChip.className = s.cliff_triggered ? 'safety-chip chip-alert' : 'safety-chip chip-safe';
    }
    const pickupChip = document.getElementById('pickupChip');
    if (pickupChip) {
      pickupChip.textContent = s.picked_up ? 'PICKED UP' : 'GROUNDED';
      pickupChip.className = s.picked_up ? 'safety-chip chip-alert' : 'safety-chip chip-safe';
    }

    // Last Action
    if (st.world.last_action_summary) {
      document.getElementById('lastActionTag').textContent = `ACT: ${st.world.last_action_summary}`;
    }

    // 2D Tactical Radar
    latestObservations = st.world.observations || [];
    document.getElementById('obsCountTag').textContent = latestObservations.length;
    drawRadar(st.world, latestObservations);

    const obsHtml = latestObservations.map(o =>
      `<div class="obs-item"><div><strong>${esc(o.label)}</strong> <span style="color:var(--accent)">${esc(o.direction)}</span></div>` +
      `<div class="obs-meta">~${esc(o.distance_mm)}mm · ${esc(o.age_s)}s ago</div></div>`
    ).join('') || '<div class="empty-hint">Scanning surroundings...</div>';
    setHtmlIfChanged('obsList', obsHtml);

    // Autonomy Controller
    const a = st.autonomy || {}, g = st.goal || {};
    document.getElementById('initiativeState').textContent =
      `${a.paused ? 'Staying here' : a.enabled ? 'Autonomy on' : 'Autonomy off'} · ${a.reason || ''}` +
      (g.status === 'running' ? ` · working on "${(g.objective || '').slice(0, 60)}" (step ${g.step || 0})` : '');
    const autoInd = document.getElementById('autoIndicator');
    if (autoInd) {
      autoInd.style.background = a.paused ? 'var(--amber)' : (a.enabled ? 'var(--accent)' : 'var(--text-muted)');
      autoInd.style.boxShadow = `0 0 8px ${autoInd.style.background}`;
    }
    if (panelVisible.drives) renderDrives(a, (st.telemetry && st.telemetry.sensors) || {}, g);

    // Chat History
    const turns = st.recent_conversations || [];
    if (turns.length) {
      const last = turns[turns.length - 1];
      document.getElementById('latencyTag').textContent = `${last.backend} (${last.latency_ms}ms)`;
    }
    renderChatBox(turns);

    // Memory Store
    document.getElementById('pendingCount').textContent = st.memory ? st.memory.consolidation_pending : 0;
    if (st.memory) {
      const memHtml = (st.memory.active_memories || []).map(m =>
        `<div class="list-row"><div class="list-row-main"><strong>(${esc(m.subject)})</strong> ${esc(m.content)} <em style="color:var(--text-muted)">[${esc(m.provenance)}]</em></div>` +
        `<button class="btn btn-sm" data-forget-topic="${esc(m.subject)}">Forget</button></div>`
      ).join('') || '<div class="empty-hint">No active long-term memories</div>';
      setHtmlIfChanged('memList', memHtml);
      document.getElementById('memList').onclick = event => {
        const button = event.target.closest('button[data-forget-topic]');
        if (button) forgetMemory(button.dataset.forgetTopic);
      };

      const tombs = (st.memory.tombstones || []).map(t => t.topic).join(', ');
      document.getElementById('tombstoneInfo').textContent = `Tombstones: ${tombs || 'none'}`;
    }

    // Event Stream
    const evList = st.recent_events || [];
    const evHtml = evList.slice().reverse().map(e =>
      `<div class="list-row"><div class="list-row-main"><strong style="color:var(--accent)">${esc(e.type)}</strong> <em style="color:var(--text-muted)">(${esc(e.source)})</em> ` +
      `<span style="color:var(--text-secondary)">${esc(JSON.stringify(e.payload))}</span></div></div>`
    ).join('');
    setHtmlIfChanged('eventList', evHtml);

  } catch (err) {
    document.getElementById('connLabel').textContent = 'DISCONNECTED';
    document.getElementById('connPill').className = 'status-pill status-disconnected';
    console.error(err);
  } finally {
    isRefreshing = false;
  }
}

async function pollLoop() {
  await refreshState();
  setTimeout(pollLoop, document.hidden ? 1000 : 200);
}

// ---------------------------------------------------------------------------------------------
// "Why" panels: the robot's own spatial map, its thinking stream, and its drives.
// ---------------------------------------------------------------------------------------------
function observeVisibility(id, key) {
  const el = document.getElementById(id);
  if (!el || !('IntersectionObserver' in window)) return;
  new IntersectionObserver(entries => {
    panelVisible[key] = entries.some(e => e.isIntersecting);
  }, {rootMargin: '120px'}).observe(el);
}

function storageGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch (e) { return fallback; }
}

function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable: keep the default */ }
}

const COLOR_WORDS = {
  red: [244, 63, 94], blue: [59, 130, 246], green: [16, 185, 129], yellow: [234, 179, 8],
  orange: [249, 115, 22], brown: [217, 119, 6], cardboard: [217, 119, 6], tan: [210, 180, 140],
  beige: [214, 196, 160], purple: [139, 92, 246], violet: [139, 92, 246], lavender: [196, 181, 253],
  pink: [236, 72, 153], white: [226, 232, 240], black: [100, 116, 139], gray: [148, 163, 184],
  grey: [148, 163, 184], teal: [20, 184, 166], cyan: [0, 240, 255], gold: [234, 179, 8],
  silver: [203, 213, 225], magenta: [217, 70, 239],
};
const FALLBACK_COLORS = [[56, 189, 248], [52, 211, 153], [251, 191, 36], [244, 114, 182], [167, 139, 250], [45, 212, 191]];

function labelRgb(label) {
  const words = String(label || '').toLowerCase().match(/[a-z]+/g) || [];
  for (const w of words) if (COLOR_WORDS[w]) return COLOR_WORDS[w];
  let h = 0;
  for (const ch of String(label || '')) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return FALLBACK_COLORS[h % FALLBACK_COLORS.length];
}

function rgba(rgb, a) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`; }

function fmtAge(s) {
  if (s == null || !isFinite(s)) return '?';
  if (s < 1) return 'just now';
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}

function fmtClock(ts) {
  if (ts == null) return '';
  if (ts > 1e9) return new Date(ts * 1000).toLocaleTimeString();
  return `t=${Number(ts).toFixed(1)}s`;
}

function fmtArgs(args) {
  if (args == null) return '';
  if (typeof args !== 'object') return String(args);
  return Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ');
}

// ---- Spatial map ------------------------------------------------------------------------------
let mapText = '';
let latestMap = null;
let showTruth = storageGet('dash.simTruth', '0') === '1';
let showCoverage = storageGet('dash.coverage', '1') === '1';

function mapView(m, W) {
  const pts = [[m.robot.x_mm, m.robot.y_mm]];
  (m.objects || []).forEach(o => pts.push([o.x_mm, o.y_mm]));
  Object.values(m.places || {}).forEach(p => pts.push(p));
  (m.occupied_cells || []).forEach(p => pts.push(p));
  if (m.sim_truth) {
    const h = m.sim_truth.room_half_extent_mm;
    pts.push([-h, -h], [h, h]);
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  pts.forEach(([x, y]) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  });
  // Fit everything, in coarse steps so the view does not jitter while the robot moves.
  const half = Math.ceil(Math.max(1000, Math.max(maxX - minX, maxY - minY) / 2 + 400) / 250) * 250;
  return {vx: Math.round((minX + maxX) / 200) * 100, vy: Math.round((minY + maxY) / 200) * 100,
          half, scale: (W / 2 - 8) / half};
}

function renderMap(m) {
  const c = document.getElementById('mapCanvas');
  if (!c || !m || !m.robot) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const v = mapView(m, W);
  // Odometry frame drawn with +x (the start heading) up and +y (left) to the left.
  const S = (x, y) => [W / 2 - (y - v.vy) * v.scale, H / 2 - (x - v.vx) * v.scale];
  ctx.fillStyle = '#05080f';
  ctx.fillRect(0, 0, W, H);
  ctx.textAlign = 'left';

  // 0.5 m grid; the axes through the start point are brighter.
  ctx.lineWidth = 1;
  const step = 500;
  for (let x = Math.floor((v.vx - v.half) / step) * step; x <= v.vx + v.half; x += step) {
    const sy = S(x, 0)[1];
    ctx.strokeStyle = x === 0 ? 'rgba(0,240,255,0.2)' : 'rgba(148,163,184,0.08)';
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W, sy); ctx.stroke();
  }
  for (let y = Math.floor((v.vy - v.half) / step) * step; y <= v.vy + v.half; y += step) {
    const sx = S(0, y)[0];
    ctx.strokeStyle = y === 0 ? 'rgba(0,240,255,0.2)' : 'rgba(148,163,184,0.08)';
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke();
  }

  // Camera coverage: runs of looked-at cells per row, drawn as single rectangles.
  const cg = m.coverage_grid;
  if (showCoverage && cg && cg.bits) {
    ctx.fillStyle = 'rgba(0,240,255,0.07)';
    const px = cg.cell_mm * v.scale;
    for (let i = 0; i < cg.n; i++) {
      const row = i * cg.n;
      const x = cg.origin_mm + (i + 0.5) * cg.cell_mm;
      let j = 0;
      while (j < cg.n) {
        if (cg.bits.charCodeAt(row + j) !== 49) { j++; continue; }
        const start = j;
        while (j < cg.n && cg.bits.charCodeAt(row + j) === 49) j++;
        const [sxA, sy] = S(x, cg.origin_mm + start * cg.cell_mm);
        const sxB = S(x, cg.origin_mm + j * cg.cell_mm)[0];
        ctx.fillRect(sxB, sy - px / 2, sxA - sxB, px);
      }
    }
  }

  // Range-sensor occupancy.
  ctx.fillStyle = 'rgba(244,63,94,0.75)';
  const cpx = Math.max(2, m.cell_mm * v.scale);
  (m.occupied_cells || []).forEach(([x, y]) => {
    const [sx, sy] = S(x, y);
    ctx.fillRect(sx - cpx / 2, sy - cpx / 2, cpx, cpx);
  });

  // Simulator ground truth (never shown to the robot).
  if (m.sim_truth) {
    const t = m.sim_truth, h = t.room_half_extent_mm;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(148,163,184,0.5)';
    const [ax, ay] = S(h, h), [bx, by] = S(-h, -h);
    ctx.strokeRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
    ctx.font = '10px Consolas, monospace';
    (t.objects || []).forEach(o => {
      const [sx, sy] = S(o.x_mm, o.y_mm);
      ctx.strokeStyle = 'rgba(203,213,225,0.55)';
      if (o.shape === 'box') {
        const w = o.depth_mm * v.scale, hh = o.width_mm * v.scale;
        ctx.strokeRect(sx - w / 2, sy - hh / 2, w, hh);
      } else {
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(3, 0.5 * Math.max(o.width_mm, o.depth_mm) * v.scale), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(203,213,225,0.55)';
      ctx.fillText(`sim: ${o.label}`, sx - 20, sy + 18);
    });
    if (t.robot) {
      const [tx, ty] = S(t.robot.x_mm, t.robot.y_mm);
      ctx.strokeStyle = 'rgba(245,158,11,0.7)';
      ctx.beginPath(); ctx.arc(tx, ty, 11, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
    ctx.fillStyle = 'rgba(245,158,11,0.95)';
    ctx.font = 'bold 11px -apple-system, sans-serif';
    ctx.fillText('SIM TRUTH overlay (dashed): not known to the robot', 8, H - 10);
  }

  // Camera field of view and coverage range from the current pose.
  const r = m.robot;
  const [rx, ry] = S(r.x_mm, r.y_mm);
  const hd = r.heading_deg * Math.PI / 180;
  const fov = ((m.camera && m.camera.hfov_deg) || 65) * Math.PI / 360;
  const range = ((m.camera && m.camera.coverage_range_mm) || 2200) * v.scale;
  const phi = -Math.PI / 2 - hd;  // canvas angle of the heading
  const grad = ctx.createRadialGradient(rx, ry, 4, rx, ry, range);
  grad.addColorStop(0, 'rgba(0,240,255,0.18)');
  grad.addColorStop(1, 'rgba(0,240,255,0)');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.arc(rx, ry, range, phi - fov, phi + fov); ctx.closePath(); ctx.fill();

  // Named places.
  Object.entries(m.places || {}).forEach(([name, p]) => {
    const [sx, sy] = S(p[0], p[1]);
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(sx, sy - 6); ctx.lineTo(sx + 6, sy); ctx.lineTo(sx, sy + 6); ctx.lineTo(sx - 6, sy);
    ctx.closePath(); ctx.fill();
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillText(name, sx + 8, sy - 6);
  });

  // Tracked objects: fade with time since last seen; dashed = restored from an earlier session.
  (m.objects || []).forEach(o => {
    const [sx, sy] = S(o.x_mm, o.y_mm);
    const age = o.age_s || 0;
    const alpha = Math.max(0.25, Math.min(1, 1 - age / 300)) * (o.stale ? 0.7 : 1);
    const rgb = labelRgb(o.label);
    const rad = Math.max(5, o.radius_mm * v.scale);
    ctx.save();
    if (o.stale) ctx.setLineDash([4, 3]);
    ctx.lineWidth = 2;
    ctx.fillStyle = rgba(rgb, 0.28 * alpha);
    ctx.strokeStyle = rgba(rgb, alpha);
    ctx.beginPath(); ctx.arc(sx, sy, rad, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = `rgba(241,245,249,${alpha})`;
    ctx.font = 'bold 11px -apple-system, sans-serif';
    ctx.fillText(o.id, sx + rad + 4, sy - 2);
    ctx.fillStyle = `rgba(148,163,184,${alpha})`;
    ctx.font = '10px -apple-system, sans-serif';
    ctx.fillText(`${o.label} · ${Math.round((o.confidence || 0) * 100)}% · ${fmtAge(age)}`, sx + rad + 4, sy + 10);
  });

  // Robot pose estimate and heading.
  ctx.save();
  ctx.translate(rx, ry);
  ctx.rotate(-hd);
  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 10;
  ctx.beginPath(); ctx.moveTo(0, -13); ctx.lineTo(9, 9); ctx.lineTo(-9, 9); ctx.closePath(); ctx.fill();
  ctx.restore();

  ctx.fillStyle = 'rgba(148,163,184,0.7)';
  ctx.font = '10px Consolas, monospace';
  ctx.fillText(`up = +x (start heading) · grid 0.5 m · view ±${(v.half / 1000).toFixed(2)} m`, 8, 14);

  const objs = (m.objects || []).slice().sort((a, b) => a.distance_mm - b.distance_mm);
  setHtmlIfChanged('mapStats',
    `objects <strong>${objs.length}</strong> · occupied cells <strong>${(m.occupied_cells || []).length}</strong>` +
    ` · coverage <strong>${(100 * (m.coverage || 0)).toFixed(1)}%</strong> of map` +
    ` · pose (<strong>${Math.round(r.x_mm)}, ${Math.round(r.y_mm)}</strong>) ${Math.round(r.heading_deg)}°`);
  setHtmlIfChanged('mapObjList', objs.map(o =>
    `<div class="list-row"><div class="list-row-main"><strong style="color:${rgba(labelRgb(o.label), 1)}">${esc(o.id)}</strong> ${esc(o.label)}` +
    `${o.stale ? ' <em style="color:var(--text-muted)">(earlier session)</em>' : ''}</div>` +
    `<span class="obs-meta">${(o.distance_mm / 1000).toFixed(1)} m · ${Math.round(o.bearing_deg)}° · ${esc(o.seen_count)}x · conf ${Math.round(100 * o.confidence)}% · ${fmtAge(o.age_s)}</span></div>`
  ).join('') || '<div class="empty-hint">No tracked objects yet (the robot has not looked around)</div>');
  const prompt = document.getElementById('mapPromptText');
  if (prompt && prompt.textContent !== (m.prompt_view || '')) prompt.textContent = m.prompt_view || '';
}

async function fetchMap() {
  const r = await fetch('/api/map' + (showTruth ? '?truth=1' : ''));
  const txt = await r.text();
  if (r.ok && txt !== mapText) {
    mapText = txt;
    latestMap = JSON.parse(txt);
    renderMap(latestMap);
  }
}

async function pollMap() {
  let delay = 1000;
  try {
    if (!document.hidden && panelVisible.map) await fetchMap();
    else delay = 1500;
  } catch (err) {
    console.error(err);
    delay = 3000;
  }
  setTimeout(pollMap, delay);
}

// ---- Thinking stream --------------------------------------------------------------------------
let thinkSeq = 0;
let thinkEntries = [];
let viewVersion = -1;
let viewPinned = false;  // the user opened an older frame from the log
let serverView = null;

function drawView(img, view, note) {
  const c = document.getElementById('viewCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#05080f';
  ctx.fillRect(0, 0, W, H);
  if (img) ctx.drawImage(img, 0, 0, W, H);
  const empty = document.getElementById('viewEmpty');
  empty.style.display = img ? 'none' : 'flex';
  if (!img) empty.textContent = note || 'No image was kept for this view.';
  ctx.textAlign = 'left';
  (view.detections || []).forEach(d => {
    const [x1, y1, x2, y2] = d.bbox;
    const rgb = labelRgb(d.label);
    ctx.strokeStyle = rgba(rgb, 1);
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
    ctx.font = 'bold 12px -apple-system, sans-serif';
    const tw = ctx.measureText(d.label).width + 8;
    const ty = y1 * H >= 16 ? y1 * H - 16 : y1 * H;
    ctx.fillStyle = rgba(rgb, 0.85);
    ctx.fillRect(x1 * W, ty, tw, 16);
    ctx.fillStyle = '#05080f';
    ctx.fillText(d.label, x1 * W + 4, ty + 12);
  });
  const p = view.pose;
  const dets = view.detections || [];
  const parts = [`<strong>view #${esc(view.version)}</strong> ${esc(fmtClock(view.ts))}`];
  if (p) parts.push(`pose (${Math.round(p.x_mm)}, ${Math.round(p.y_mm)}) ${Math.round(p.heading_deg)}° tilt ${Math.round(p.head_tilt_deg)}°`);
  parts.push(view.query ? `looked for "${esc(view.query)}"` : 'looked for everything');
  parts.push(`${dets.length} detection(s): ${esc(dets.map(d => d.label).join(', ') || 'none')}`);
  if ((view.seen || []).length) parts.push(`map ids: ${esc(view.seen.join(', '))}`);
  if ((view.lost || []).length) parts.push(`<span style="color:var(--rose)">lost: ${esc(view.lost.join(', '))}</span>`);
  if (viewPinned) parts.push('<span class="step-link" data-view-latest="1">back to latest view</span>');
  setHtmlIfChanged('viewCaption', parts.join(' · '));
}

function showView(view) {
  viewVersion = view.version;
  if (!view.has_image) { drawView(null, view); return; }
  const img = new Image();
  img.onload = () => { if (viewVersion === view.version) drawView(img, view); };
  img.onerror = () => {
    if (viewVersion === view.version) drawView(null, view, 'This frame is no longer kept on the server.');
  };
  img.src = `/api/thinking/frame?v=${encodeURIComponent(view.version)}`;
}

function renderStep(e) {
  switch (e.type) {
    case 'cognition.model_call': {
      const tok = e.prompt_tokens ? ` · ${e.prompt_tokens} tok in` + (e.completion_tokens ? ` / ${e.completion_tokens} out` : '') : '';
      const calls = (e.tool_calls || []).map(c => `<code>${esc(c.name)}(${esc(fmtArgs(c.args))})</code>`).join(' ');
      let body = '<div class="step-muted">(empty reply)</div>';
      if (calls) body = `<div>calls ${calls}</div>` + (e.content ? `<div class="step-muted">${esc(e.content)}</div>` : '');
      else if (e.content) body = `<div class="step-say">"${esc(e.content)}"</div>`;
      return `<div class="step step-model"><span class="step-k">MODEL #${esc(e.step)}</span>` +
        `<span class="step-muted">${esc(Math.round(e.latency_ms || 0))} ms${esc(tok)}</span>${body}</div>`;
    }
    case 'skill.completed': {
      let data = '';
      if (typeof e.data === 'string') data = e.data;
      else if (e.data && Object.keys(e.data).length) data = JSON.stringify(e.data);
      return `<div class="step step-skill ${e.ok ? 'ok' : 'fail'}"><span class="step-k">${e.ok ? '✓' : '✗'} ${esc(e.tool)}</span>` +
        `<code>${esc(fmtArgs(e.args))}</code><div>${esc(e.message)}` +
        `${!e.ok && e.status ? ` <span class="step-muted">[${esc(e.status)}]</span>` : ''}</div>` +
        (data ? `<div class="step-muted"><code>${esc(data)}</code></div>` : '') + '</div>';
    }
    case 'perception.view': {
      const labels = (e.detections || []).map(d => d.label).join(', ') || 'nothing';
      const extra = [e.query ? `looked for "${e.query}"` : '',
        (e.seen || []).length ? `ids ${e.seen.join(', ')}` : '',
        (e.lost || []).length ? `lost ${e.lost.join(', ')}` : ''].filter(Boolean).join(' · ');
      return `<div class="step step-view"><span class="step-k">SAW</span>${esc(labels)}` +
        (extra ? ` <span class="step-muted">${esc(extra)}</span>` : '') +
        (e.has_image ? `<span class="step-link" data-view="${esc(e.version)}">frame #${esc(e.version)}</span>` : '') + '</div>';
    }
    case 'memory.retrieved':
      return `<div class="step step-mem"><span class="step-k">MEMORY</span>${esc(e.count)} recalled` +
        `${(e.memories || []).length ? ': ' + esc(e.memories.join(' | ')) : ''}</div>`;
    case 'cognition.recovered_calls':
      return `<div class="step step-warn"><span class="step-k">RECOVERED</span>tool call(s) written as text: ${esc((e.calls || []).join(', '))}</div>`;
    case 'cognition.unavailable':
      return `<div class="step step-warn"><span class="step-k">MODEL OFFLINE</span>${esc(e.error)}</div>`;
    case 'turn.cancelled':
      return '<div class="step step-warn"><span class="step-k">CANCELLED</span>the turn was interrupted</div>';
    case 'reflex.triggered':
      return `<div class="step step-warn"><span class="step-k">REFLEX</span><code>${esc(fmtArgs(e.detail))}</code></div>`;
    default:
      return `<div class="step"><span class="step-k">${esc(e.type)}</span><code>${esc(fmtArgs(e.detail))}</code></div>`;
  }
}

function renderTurn(g) {
  const h = g.head || {};
  const origin = h.origin || (g.final && g.final.origin) || (g.key.startsWith('sys-') ? 'system' : 'user');
  let html = `<div class="turn-head"><span class="turn-badge">${esc(origin.toUpperCase())}</span>` +
    (h.trigger ? `<span class="turn-trigger">${esc(h.trigger)}</span>` : '') +
    `<span class="turn-time">${esc(fmtClock(g.ts))}</span></div>` +
    (h.objective ? `<div class="turn-objective">"${esc(h.objective)}"</div>` : '');
  // A started skill is shown only until its completion arrives.
  const completions = {};
  g.items.forEach(e => { if (e.type === 'skill.completed') completions[e.tool] = (completions[e.tool] || 0) + 1; });
  g.items.forEach(e => {
    if (e.type !== 'skill.started') { html += renderStep(e); return; }
    if (completions[e.tool] > 0) { completions[e.tool] -= 1; return; }
    html += `<div class="step step-skill running"><span class="step-k">RUNNING ${esc(e.tool)}</span><code>${esc(fmtArgs(e.args))}</code></div>`;
  });
  if (g.final) {
    const f = g.final;
    const bad = f.status !== 'completed';
    const meta = [`${f.steps} step(s)`, `${Math.round(f.latency_ms || 0)} ms`, f.backend,
      f.emotion ? `feels ${f.emotion}` : '', f.next_wake_s != null ? `wake again in ${f.next_wake_s}s` : '']
      .filter(Boolean).join(' · ');
    html += `<div class="turn-final${bad ? ' status-bad' : ''}"><span class="step-k">${esc(String(f.status || '').toUpperCase())}</span>` +
      `<span class="step-muted">${esc(meta)}</span>` +
      (f.say ? `<div class="step-say">"${esc(f.say)}"</div>` : '<div class="step-muted">(said nothing)</div>') + '</div>';
    if (g.lateCancel) html += '<div class="step-muted">cancel notice arrived after the reply was sent (no effect)</div>';
  }
  return `<div class="turn-card origin-${esc(origin)}">${html}</div>`;
}

function renderThinking() {
  const groups = [];
  const byTurn = {};
  let current = null;
  thinkEntries.forEach(e => {
    // Events without a turn id (reflexes) belong to the turn that is still running, if any.
    let g = e.turn ? byTurn[e.turn] : (current && !current.final ? current : null);
    if (!g) {
      g = {key: e.turn || `sys-${e.seq}`, head: null, items: [], final: null, ts: e.ts};
      groups.push(g);
      byTurn[g.key] = g;
    }
    if (e.type === 'turn.started') g.head = e;
    else if (e.type === 'robot.response') g.final = e;
    else if (e.type === 'turn.cancelled' && g.final) g.lateCancel = true;  // after the reply: no effect
    else g.items.push(e);
    current = g;
  });
  const box = document.getElementById('thinkLog');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  setHtmlIfChanged('thinkLog',
    groups.slice(-12).map(renderTurn).join('') || '<div class="empty-hint">Waiting for the first turn...</div>');
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function renderThinkStatus(d) {
  document.getElementById('thinkStatusTag').textContent =
    d.busy ? `THINKING · ${String(d.fsm_state || '').toUpperCase()}` : `IDLE · ${d.backend || ''}`;
  const g = d.goal;
  setHtmlIfChanged('thinkGoal', g
    ? `GOAL <strong>"${esc(g.objective)}"</strong> · ${esc(g.status)} · step ${esc(g.step)} · ${esc(g.origin)}`
    : 'No goal yet.');
}

async function pollThinking() {
  let delay = 1500;
  try {
    if (!document.hidden && panelVisible.think) {
      const r = await fetch(`/api/thinking?since=${thinkSeq}`);
      const d = await r.json();
      if (d.reset) thinkEntries = [];
      if (d.reset || (d.entries && d.entries.length)) {
        thinkEntries = thinkEntries.concat(d.entries || []).slice(-300);
        renderThinking();
      }
      thinkSeq = d.latest_seq;
      renderThinkStatus(d);
      serverView = d.view;
      if (d.view && !viewPinned && d.view.version !== viewVersion) showView(d.view);
      delay = d.busy ? 500 : 1200;
    }
  } catch (err) {
    console.error(err);
    delay = 3000;
  }
  setTimeout(pollThinking, delay);
}

// ---- Drives & autonomy (fed by /api/state, no extra request) ---------------------------------
function renderDrives(a, sensors, goal) {
  document.getElementById('autoModeTag').textContent =
    !a.enabled ? 'OFF' : a.paused ? 'PAUSED' : a.active ? 'DECIDING' : 'ON';
  const pill = (label, cls) => `<span class="mode-pill ${cls}">${esc(label)}</span>`;
  setHtmlIfChanged('autoPills', [
    pill(a.enabled ? 'ENABLED' : 'DISABLED', a.enabled ? 'on' : ''),
    pill(a.paused ? 'PAUSED (staying here)' : 'NOT PAUSED', a.paused ? 'warn' : 'on'),
    pill(a.quiet ? 'QUIET' : 'MAY TALK', a.quiet ? 'warn' : 'on'),
    a.active ? pill('BACKGROUND TURN RUNNING', 'busy') : '',
    goal && goal.status === 'running' ? pill(`${goal.origin || 'user'} goal running`, 'busy') : '',
  ].join(''));
  const kv = [
    ['Reason', a.reason || '-'],
    ['Last trigger', a.last_trigger || 'none yet'],
    ['Requested wake', a.next_decision_in_s != null ? `in ${a.next_decision_in_s}s` : 'none'],
    ['Since human', `${a.minutes_since_human != null ? a.minutes_since_human : 0} min`],
    ['Unexplored', a.unexplored_fraction != null ? `${Math.round(a.unexplored_fraction * 100)}% within 1.5 m` : '?'],
  ];
  setHtmlIfChanged('autoDetails', kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join(''));
  const bars = Object.entries(a.drives || {}).map(([name, d]) => {
    const lvl = Math.max(0, Math.min(1, d.level || 0)), th = Math.max(0, Math.min(1, d.threshold || 0));
    const urgent = th > 0 && lvl >= th;
    return `<div class="drive-row"><div class="drive-head"><span>${esc(name)}${urgent ? ' · <span style="color:var(--amber)">URGE</span>' : ''}</span>` +
      `<span class="num">${lvl.toFixed(2)} / ${th.toFixed(2)}${d.rate_per_min ? ` · +${esc(d.rate_per_min)}/min` : ''}</span></div>` +
      `<div class="drive-track"><div class="drive-fill${urgent ? ' urgent' : ''}" style="width:${(lvl * 100).toFixed(1)}%"></div>` +
      `<div class="drive-thresh" style="left:calc(${(th * 100).toFixed(1)}% - 1px)"></div></div></div>`;
  }).join('') || '<div class="empty-hint">No drive data</div>';
  setHtmlIfChanged('driveBars', bars);
  // Live body reading first; the drive system only samples it on autonomy ticks.
  const battery = sensors.battery_pct != null ? sensors.battery_pct : a.battery_pct;
  const charging = sensors.battery_pct != null ? sensors.charging : a.charging;
  if (battery == null) {
    setHtmlIfChanged('batteryBox', '<div class="empty-hint">No battery reading</div>');
  } else {
    const pct = Math.max(0, Math.min(100, battery));
    const cls = pct < 20 ? ' low' : pct < 35 ? ' urgent' : '';
    setHtmlIfChanged('batteryBox',
      `<div class="drive-row"><div class="drive-head"><span>${charging ? 'CHARGING on dock' : 'ON BATTERY'}</span>` +
      `<span class="num">${pct.toFixed(1)}%</span></div>` +
      `<div class="drive-track"><div class="drive-fill${cls}" style="width:${pct.toFixed(1)}%"></div></div></div>`);
  }
}

(function initInsightPanels() {
  observeVisibility('faceCanvas', 'face');
  observeVisibility('mapPanel', 'map');
  observeVisibility('thinkPanel', 'think');
  observeVisibility('drivesPanel', 'drives');
  const truth = document.getElementById('truthToggle');
  const coverage = document.getElementById('coverageToggle');
  truth.checked = showTruth;
  coverage.checked = showCoverage;
  truth.addEventListener('change', () => {
    showTruth = truth.checked;
    storageSet('dash.simTruth', showTruth ? '1' : '0');
    fetchMap().catch(err => console.error(err));
  });
  coverage.addEventListener('change', () => {
    showCoverage = coverage.checked;
    storageSet('dash.coverage', showCoverage ? '1' : '0');
    if (latestMap) renderMap(latestMap);
  });
  document.getElementById('thinkLog').addEventListener('click', ev => {
    const link = ev.target.closest('[data-view]');
    if (!link) return;
    const entry = thinkEntries.find(e => e.type === 'perception.view' && String(e.version) === link.dataset.view);
    if (!entry) return;
    viewPinned = true;
    showView(entry);
  });
  document.getElementById('viewCaption').addEventListener('click', ev => {
    if (!ev.target.closest('[data-view-latest]')) return;
    viewPinned = false;
    if (serverView) showView(serverView);
  });
})();

requestAnimationFrame(stepClientAnimation);
pollLoop();
pollMap();
pollThinking();
</script>
</body>
</html>
"""


class DashboardServer:
    """Lightweight asyncio HTTP server providing live diagnostics and interactive controls."""

    name = "dashboard"

    def __init__(
        self,
        harness: RobotHarness,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        max_events: int = 120,
    ) -> None:
        self.harness = harness
        self.host = host
        self.port = port
        self.bound_port: int = port
        self._server: asyncio.AbstractServer | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._last_chat_correlation: str | None = None
        self._conversations: deque[dict[str, Any]] = deque(maxlen=30)
        self._sub_ids: list[str] = []
        self._last_user_text: str = ""
        self._capabilities_dict: dict[str, Any] = self.harness.registry.to_dict()
        self._memory_cache: dict[str, Any] | None = None
        self._memory_revision = -1
        self._clients: set[asyncio.Task] = set()
        # "Thinking" stream: why the robot did what it did.
        self._thinking: deque[dict[str, Any]] = deque(maxlen=MAX_THINKING_ENTRIES)
        self._thinking_seq = 0
        self._thinking_turn: str | None = None
        self._view: dict[str, Any] | None = None  # latest camera view the robot itself looked at
        self._view_frames: deque[tuple[int, bytes]] = deque(maxlen=KEPT_VIEW_FRAMES)
        self._view_version = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.bound_port}"

    async def start(self) -> None:
        if self._server is not None:
            return
        self._sub_ids.append(
            self.harness.bus.subscribe(
                "*", self._on_any_event, name="dashboard:tap", max_queue_size=64, lossy=True
            )
        )
        for pattern in THINKING_PATTERNS:
            self._sub_ids.append(
                self.harness.bus.subscribe(
                    pattern, self._on_thinking_event, name=f"dashboard:think:{pattern}", immediate=True
                )
            )
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
            limit=8192,
        )
        sockets = self._server.sockets or []
        if sockets:
            self.bound_port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        for sid in self._sub_ids:
            self.harness.bus.unsubscribe(sid)
        self._sub_ids.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        clients = tuple(self._clients)
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)

    async def health(self) -> ServiceHealth:
        running = self._server is not None and self._server.is_serving()
        return ServiceHealth(
            name=self.name,
            healthy=running,
            status="ok" if running else "stopped",
            details={"url": self.url},
        )

    def build_state_payload(self) -> dict[str, Any]:
        """Assemble full diagnostic state JSON for `/api/state`."""
        face_params = self.harness.face_bridge.current_frame.to_dict()
        self.harness.sandbox.update_face_state(face_params)
        if self._memory_cache is None or self._memory_revision != self.harness.memory.revision:
            self._memory_cache = self.harness.memory.dashboard_snapshot()
            self._memory_revision = self.harness.memory.revision
        mem_export = self._memory_cache

        # Retrieve freshest live pose & sensors from body HAL
        live_pose = self.harness.body.pose().to_dict()
        live_sensors = self.harness.body.sensors().to_dict()
        world_snapshot = self.harness.world.snapshot()
        world_snapshot["pose"].update(live_pose)
        world_snapshot["sensors"].update(live_sensors)

        services_health = {
            "body": "online",
            "executor": "online",
            "face_bridge": "online",
            "audio": "online" if (self.harness.audio.enable_speaker or self.harness.audio.enable_mic) else "standby",
            "dashboard": "online",
            "autonomy": "active" if self.harness.autonomy.enabled else "idle",
        }

        return {
            "robot_name": self.harness.config.name,
            "fsm_state": str(self.harness.fsm.state),
            "in_follow_up": self.harness.fsm.state == "follow_up"
            and self.harness.clock.monotonic() <= self.harness.fsm.follow_up_until_ts,
            "idle_seconds": round(self.harness.fsm.idle_seconds(), 1),
            "face": face_params,
            "audio": {
                "is_speaking": self.harness.audio.is_speaking,
                "muted": self.harness.audio.muted,
                "enable_speaker": self.harness.audio.enable_speaker,
                "enable_mic": self.harness.audio.enable_mic,
                "sfx_played_count": self.harness.audio.sfx_played_count,
                "tts_spoken_count": self.harness.audio.tts_spoken_count,
                "barge_in_count": self.harness.audio.barge_in_count,
                "last_spoken_text": self.harness.audio.last_spoken_text,
                "last_asr_text": self.harness.audio.last_asr_text,
            },
            "world": world_snapshot,
            "telemetry": {
                "pose": live_pose,
                "sensors": live_sensors,
            },
            "services": services_health,
            "autonomy": self.harness.autonomy.snapshot(),
            "goal": self.harness.arbiter.active_goal,
            "sandbox": self.harness.sandbox.snapshot(self.harness.body.pose()),
            "capabilities": self._capabilities_dict,
            "memory": {
                "active_memories": mem_export["active_memories"],
                "tombstones": mem_export["tombstones"],
                "recent_episodes": mem_export["recent_episodes"],
                "consolidation_pending": self.harness.consolidator.pending_count,
                "last_consolidation": self.harness.consolidator.last_report.to_dict(),
            },
            "bus_stats": {
                "published_count": self.harness.bus.published_count,
                "dropped_count": self.harness.bus.dropped_count,
            },
            "recording": None
            if self.harness.recorder is None
            else {
                "recorded": self.harness.recorder.recorded_count,
                "dropped": self.harness.recorder.dropped_count,
                "failed": self.harness.recorder.failed,
            },
            "recent_conversations": list(self._conversations),
            "recent_events": list(self._events)[-45:],
        }

    def _on_any_event(self, event: Event) -> None:
        self._events.append(event.to_dict())
        if event.type.startswith("memory."):
            self._memory_cache = None
        if event.type == "attention.engaged":
            self._last_user_text = str(event.payload.get("text", ""))
        elif event.type == "robot.response":
            self._memory_cache = None
            p = event.payload
            if not p.get("say"):
                return
            origin = p.get("origin", "user")
            user_text = self._last_user_text if origin == "user" and event.correlation_id != self._last_chat_correlation else ""
            self._last_chat_correlation = event.correlation_id
            self._conversations.append(
                {
                    "user": user_text,
                    "origin": origin,
                    "say": p.get("say", ""),
                    "emotion": (p.get("emotion") or {}).get("name", "neutral"),
                    "sound": p.get("sound", "none"),
                    "actions": p.get("actions", []),
                    "status": p.get("status", "completed"),
                    "steps": p.get("steps", 0),
                    "goal_state": p.get("goal_state"),
                    "backend": p.get("backend", "unknown"),
                    "latency_ms": p.get("latency_ms", 0.0),
                }
            )

    # ---- "why" views: spatial map, thinking stream, drives ------------------------------------------
    def build_map_payload(self, *, truth: bool = False) -> dict[str, Any]:
        """The robot's own spatial map (odometry frame) for `/api/map`; `truth` adds simulator ground truth."""
        harness = self.harness
        pose = harness.body.pose()
        smap = harness.spatial_map
        data = smap.snapshot(pose)
        for obj in data["objects"]:
            for key in ("x_mm", "y_mm", "radius_mm", "first_seen", "last_seen"):
                obj[key] = round(float(obj[key]), 1)
            obj["confidence"] = round(float(obj["confidence"]), 2)
        seen = np.asarray(smap.seen, dtype=bool)
        k = COVERAGE_DOWNSAMPLE
        m = seen.shape[0] // k
        coarse = seen[: m * k, : m * k].reshape(m, k, m, k).any(axis=(1, 3))
        model = getattr(harness.camera, "model", None)
        data.update(
            {
                "robot": {
                    "x_mm": round(pose.x_mm, 1),
                    "y_mm": round(pose.y_mm, 1),
                    "heading_deg": round(pose.heading_deg, 1),
                    "head_tilt_deg": round(pose.head_tilt_deg, 1),
                },
                "half_extent_mm": smap.half,
                "cell_mm": CELL_MM,
                "coverage_grid": {
                    "n": m,
                    "cell_mm": CELL_MM * k,
                    "origin_mm": -smap.half,
                    # Row i is x, column j is y: bits[i * n + j].
                    "bits": "".join("1" if v else "0" for v in coarse.ravel().tolist()),
                },
                "camera": {
                    "hfov_deg": round(float(model.hfov_deg), 1) if model is not None else 65.0,
                    "coverage_range_mm": 2200.0,
                },
                "prompt_view": smap.render_for_prompt(pose),
            }
        )
        if truth:
            sandbox = harness.sandbox
            true_pose_fn = getattr(harness.body, "true_pose", None)
            true_pose = true_pose_fn() if callable(true_pose_fn) else pose
            data["sim_truth"] = {
                "room_half_extent_mm": sandbox.room_half_extent_mm,
                "robot": {
                    "x_mm": round(true_pose.x_mm, 1),
                    "y_mm": round(true_pose.y_mm, 1),
                    "heading_deg": round(true_pose.heading_deg, 1),
                },
                "objects": [
                    {
                        key: obj[key]
                        for key in ("id", "label", "shape", "x_mm", "y_mm", "width_mm", "depth_mm", "color")
                    }
                    for obj in sandbox.list_objects()
                ],
            }
        return data

    def build_thinking_payload(self, since: int = 0) -> dict[str, Any]:
        """Thinking entries newer than `since`, plus the goal and the latest view the robot looked at."""
        since = max(0, since)
        reset = since > self._thinking_seq  # the server restarted under a long-lived page
        entries = [e for e in self._thinking if reset or e["seq"] > since]
        arbiter = self.harness.arbiter
        return {
            "now": self.harness.clock.now(),
            "latest_seq": self._thinking_seq,
            "reset": reset,
            "entries": entries,
            "view": self._view,
            "kept_frames": [version for version, _ in self._view_frames],
            "goal": dict(arbiter.active_goal) if arbiter.active_goal else None,
            "busy": arbiter.busy,
            "fsm_state": str(self.harness.fsm.state),
            "backend": getattr(self.harness.backend, "name", type(self.harness.backend).__name__),
        }

    def build_autonomy_payload(self) -> dict[str, Any]:
        """Autonomy switches, drive levels and the body's battery for `/api/autonomy`."""
        sensors = self.harness.body.sensors()
        arbiter = self.harness.arbiter
        return {
            **self.harness.autonomy.snapshot(),
            "body": {
                "battery_pct": round(float(getattr(sensors, "battery_pct", 100.0)), 1),
                "charging": bool(getattr(sensors, "charging", False)),
            },
            "goal": dict(arbiter.active_goal) if arbiter.active_goal else None,
            "busy": arbiter.busy,
        }

    def view_frame(self, version: int | None = None) -> bytes | None:
        """PNG of a recent view the robot looked at (the latest when `version` is None)."""
        if not self._view_frames:
            return None
        if version is None:
            return self._view_frames[-1][1]
        return next((png for v, png in self._view_frames if v == version), None)

    def _on_thinking_event(self, event: Event) -> None:
        try:
            self._record_thinking(event)
        except Exception:  # an observer must never disturb the robot
            logger.debug("Dashboard could not record %s", event.type, exc_info=True)

    def _append_thinking(self, kind: str, event: Event, data: dict[str, Any]) -> None:
        self._thinking_seq += 1
        self._thinking.append(
            {"seq": self._thinking_seq, "ts": event.ts, "type": kind, "turn": event.correlation_id, **data}
        )

    def _turn_header(self, turn: str, payload: dict[str, Any]) -> dict[str, Any]:
        goal = self.harness.arbiter.active_goal or {}
        if goal.get("correlation_id") == turn:
            origin = str(goal.get("origin", "user"))
            objective = str(goal.get("objective", ""))
        else:
            origin = str(payload.get("origin", "user"))
            objective = self._last_user_text if origin == "user" else ""
        trigger = self.harness.autonomy.last_trigger if origin == "autonomous" else None
        return {"origin": origin, "objective": _clip_text(objective, 300), "trigger": trigger}

    def _record_thinking(self, event: Event) -> None:
        kind, p, turn = event.type, event.payload, event.correlation_id
        if turn and turn != self._thinking_turn and kind not in ("turn.cancelled", "reflex.triggered"):
            self._thinking_turn = turn
            self._append_thinking("turn.started", event, self._turn_header(turn, p))
        if kind == "cognition.model_call":
            calls = [c for c in (p.get("tool_calls") or []) if isinstance(c, dict)][:8]
            data = {
                "step": p.get("step"),
                "latency_ms": p.get("latency_ms"),
                "prompt_tokens": p.get("prompt_tokens"),
                "completion_tokens": p.get("completion_tokens"),
                "content": _clip_text(p.get("content"), 400),
                "tool_calls": [
                    {"name": _clip_text(c.get("name"), 40), "args": _clip_json(c.get("args"), 300)} for c in calls
                ],
            }
        elif kind == "skill.started":
            data = {"tool": _clip_text(p.get("tool"), 40), "args": _clip_json(p.get("args"), 300)}
        elif kind == "skill.completed":
            data = {
                "tool": _clip_text(p.get("tool"), 40),
                "args": _clip_json(p.get("args"), 300),
                "ok": bool(p.get("ok")),
                "status": _clip_text(p.get("status"), 24),
                "message": _clip_text(p.get("message"), 400),
                "data": _clip_json(p.get("data"), 600),
            }
        elif kind == "perception.view":
            data = self._record_view(event)
        elif kind == "memory.retrieved":
            memories = [m for m in (p.get("memories") or []) if isinstance(m, dict)][:4]
            data = {
                "count": int(p.get("count", len(memories)) or 0),
                "memories": [_clip_text(m.get("content", ""), 140) for m in memories],
            }
        elif kind == "robot.response":
            emotion = p.get("emotion") if isinstance(p.get("emotion"), dict) else {}
            data = {
                "say": _clip_text(p.get("say"), 400),
                "status": _clip_text(p.get("status", "completed"), 24),
                "steps": p.get("steps", 0),
                "backend": _clip_text(p.get("backend", ""), 40),
                "latency_ms": p.get("latency_ms", 0.0),
                "origin": _clip_text(p.get("origin", "user"), 16),
                "next_wake_s": p.get("next_wake_s"),
                "emotion": _clip_text(emotion.get("name", ""), 24),
                "tools": [_clip_text(t.get("tool"), 40) for t in (p.get("actions") or []) if isinstance(t, dict)],
            }
        elif kind == "cognition.recovered_calls":
            data = {"calls": [_clip_text(c, 40) for c in (p.get("calls") or [])][:8]}
        elif kind == "cognition.unavailable":
            data = {"error": _clip_text(p.get("error"), 200)}
        else:  # turn.cancelled, reflex.triggered, any other cognition.* event
            data = {"detail": _clip_json(p, 300)}
        self._append_thinking(kind, event, data)

    def _record_view(self, event: Event) -> dict[str, Any]:
        """Pair a perception.view with the frame it was detected on.

        `perceive()` stores the captured frame on the skill context right before emitting the view,
        and this handler runs synchronously inside that emit, so the pairing is exact.
        """
        p = event.payload
        detections = []
        for det in (p.get("detections") or [])[:16]:
            if not isinstance(det, dict):
                continue
            try:
                bbox = [round(float(v), 3) for v in det.get("bbox", [])][:4]
            except (TypeError, ValueError):
                continue
            if len(bbox) == 4:
                detections.append({"label": _clip_text(det.get("label"), 48), "bbox": bbox})
        frame = getattr(getattr(self.harness, "skill_context", None), "attached_image", None)
        png = getattr(frame, "png_bytes", None)
        has_image = isinstance(png, bytes | bytearray) and len(png) > 0
        self._view_version += 1
        if has_image:
            self._view_frames.append((self._view_version, bytes(png)))
        pose = getattr(frame, "pose", None)
        view = {
            "version": self._view_version,
            "ts": event.ts,
            "turn": event.correlation_id,
            "has_image": has_image,
            "pose": None
            if pose is None
            else {
                "x_mm": round(pose.x_mm, 1),
                "y_mm": round(pose.y_mm, 1),
                "heading_deg": round(pose.heading_deg, 1),
                "head_tilt_deg": round(pose.head_tilt_deg, 1),
            },
            "query": _clip_text(p.get("query"), 60) if p.get("query") else None,
            "detections": detections,
            "seen": [_clip_text(i, 32) for i in (p.get("seen") or [])][:16],
            "lost": [_clip_text(i, 32) for i in (p.get("lost") or [])][:16],
        }
        self._view = view
        return {k: v for k, v in view.items() if k not in ("ts", "turn")}

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        admitted = len(self._clients) < 8
        if admitted:
            self._clients.add(task)
        status, content_type, data = (
            "503 Service Unavailable",
            "application/json",
            b'{"error":"robot is busy"}',
        )
        try:
            if admitted:
                async with asyncio.timeout(5.0):
                    line = await reader.readline()
                    if len(line) > 2048:
                        raise ValueError("Request target is too long")
                    parts = line.decode("ascii").strip().split()
                    if len(parts) != 3:
                        raise ValueError("Malformed request")
                    method, target, _version = parts
                    headers = {}
                    size = 0
                    for _ in range(32):
                        line = await reader.readline()
                        size += len(line)
                        if size > 8192:
                            raise ValueError("Headers are too large")
                        if line == b"\r\n":
                            break
                        if not line or b":" not in line:
                            raise ValueError("Malformed headers")
                        key, value = line.decode("ascii").split(":", 1)
                        key = key.lower().strip()
                        if key in headers:
                            raise ValueError("Duplicate header")
                        headers[key] = value.strip()
                    else:
                        raise ValueError("Too many headers")
                    allowed_hosts = {f"127.0.0.1:{self.bound_port}", f"localhost:{self.bound_port}"}
                    if headers.get("host") not in allowed_hosts:
                        raise PermissionError("Invalid Host")
                    if headers.get("origin") and headers["origin"] not in {
                        "http://" + h for h in allowed_hosts
                    }:
                        raise PermissionError("Cross-origin control is forbidden")
                    length = int(headers.get("content-length", "0"))
                    if not 0 <= length <= 16384 or "transfer-encoding" in headers:
                        raise ValueError("Invalid or oversized body")
                    if (
                        method == "POST"
                        and headers.get("content-type", "").split(";", 1)[0].strip() != "application/json"
                    ):
                        raise ValueError("Expected application/json")
                    raw = await reader.readexactly(length) if length else b"{}"
                    body = json.loads(
                        raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON"))
                    )
                    if not isinstance(body, dict):
                        raise ValueError("JSON body must be an object")
                parsed = urlparse(target)
                status, content_type, data = await self._route_request(
                    method.upper(), parsed.path, parse_qs(parsed.query), body
                )
        except PermissionError as exc:
            status, data = "403 Forbidden", json.dumps({"error": str(exc)}).encode()
        except (ValueError, TypeError, KeyError, OverflowError, asyncio.IncompleteReadError) as exc:
            status, data = "400 Bad Request", json.dumps({"error": str(exc)}).encode()
        except TimeoutError:
            status, data = "408 Request Timeout", b'{"error":"request timed out"}'
        except RuntimeError as exc:
            status, data = "503 Service Unavailable", json.dumps({"error": str(exc)}).encode()
        except Exception:
            logger.exception("Dashboard request failed")
            status, data = "500 Internal Server Error", b'{"error":"request failed"}'
        finally:
            try:
                header = (
                    f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(data)}\r\n"
                    "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
                    "Content-Security-Policy: frame-ancestors 'none'\r\nConnection: close\r\n\r\n"
                ).encode()
                writer.write(header + data)
                await asyncio.wait_for(writer.drain(), timeout=1.0)
            except (ConnectionError, TimeoutError):
                pass
            writer.close()
            self._clients.discard(task)

    async def _route_request(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[str, str, bytes]:
        if method == "GET" and path in ("/", "/index.html"):
            return "200 OK", "text/html; charset=utf-8", DASHBOARD_HTML.encode("utf-8")

        if method == "GET" and path == "/api/state":
            data = self.build_state_payload()
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/events":
            data = {"events": list(self._events)}
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/map":
            data = self.build_map_payload(truth=_query_flag(query, "truth"))
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/thinking":
            since = int((query.get("since") or ["0"])[0] or 0)
            data = self.build_thinking_payload(since)
            return "200 OK", "application/json", json.dumps(data).encode("utf-8")

        if method == "GET" and path == "/api/thinking/frame":
            raw_version = (query.get("v") or [""])[0]
            png = self.view_frame(int(raw_version) if raw_version else None)
            if png is None:
                return "404 Not Found", "application/json", b'{"error":"no camera view kept for that version"}'
            return "200 OK", "image/png", png

        if method == "GET" and path == "/api/autonomy":
            return "200 OK", "application/json", json.dumps(self.build_autonomy_payload()).encode("utf-8")

        if method == "POST" and path == "/api/message":
            text = str(body.get("text", "")).strip()
            if not text:
                return "400 Bad Request", "application/json", b'{"error":"empty text"}'
            self._last_user_text = text
            resp = await self.harness.submit_message(text)
            return "200 OK", "application/json", json.dumps(resp).encode("utf-8")

        if method == "POST" and path == "/api/autonomy":
            if not body or any(k not in {"enabled", "paused", "quiet"} or not isinstance(v, bool) for k, v in body.items()):
                return "400 Bad Request", "application/json", b'{"error":"expected autonomy boolean switches"}'
            await self.harness.autonomy.set_mode(**body)
            return "200 OK", "application/json", json.dumps(self.harness.autonomy.snapshot()).encode("utf-8")

        if method == "POST" and path == "/api/expression":
            if "theme" in body:
                await self.harness.bus.emit(
                    "face.theme",
                    source="dashboard",
                    payload={"theme": str(body["theme"])},
                    priority=EventPriority.HIGH,
                )
            if any(k in body for k in ("emotion", "intensity", "gaze_target")):
                emo = str(body.get("emotion", self.harness.emotion.snapshot().emotion))
                inten = float(body.get("intensity", self.harness.emotion.snapshot().intensity))
                gaze = body.get("gaze_target")
                await self.harness.bus.emit(
                    "expression.manual",
                    source="dashboard",
                    payload={"emotion": emo, "intensity": inten, "gaze_target": gaze},
                    priority=EventPriority.HIGH,
                )
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(self.harness.face_bridge.current_frame.to_dict()).encode("utf-8"),
            )

        if method == "POST" and path == "/api/action":
            tool = str(body.get("tool", ""))
            args = dict(body.get("args") or {})
            from robot.actions.tools import ToolCallRequest

            await self.harness.bus.emit(
                "action.cancel", "dashboard", {"reason": "user_stop_command" if tool == "stop" else "manual_control"},
                priority=EventPriority.CRITICAL,
            )
            result = await self.harness.executor.execute_call(ToolCallRequest(tool, args))
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(
                    {"ok": result.ok, "status": result.status, "message": result.message, "pose": result.pose}
                ).encode(),
            )

        if method == "POST" and path == "/api/sandbox":
            op = str(body.get("op", "snapshot")).strip().lower()
            if op == "move":
                self.harness.sandbox.move_object(
                    str(body.get("id", "")),
                    float(body.get("x_mm", 0.0)),
                    float(body.get("y_mm", 0.0)),
                )
            elif op == "ray_move":
                self.harness.sandbox.move_object_by_overview_ray(
                    str(body.get("id", "")),
                    float(body.get("u_norm", 0.5)),
                    float(body.get("v_norm", 0.5)),
                )
            elif op == "orbit":
                self.harness.sandbox.set_orbit_camera(
                    azimuth_deg=float(body["azimuth_deg"]) if "azimuth_deg" in body else None,
                    elevation_deg=float(body["elevation_deg"]) if "elevation_deg" in body else None,
                    distance_m=float(body["distance_m"]) if "distance_m" in body else None,
                    lookat_m=list(body["lookat_m"]) if "lookat_m" in body else None,
                )
            elif op == "launch_viewer":
                self.harness.sandbox.launch_native_viewer(self.url)
            elif op == "add":
                self.harness.sandbox.add_object(
                    label=str(body.get("label", "Orange Cone")),
                    shape=str(body.get("shape", "cylinder")),
                    x_mm=float(body.get("x_mm", 550.0)),
                    y_mm=float(body.get("y_mm", 0.0)),
                    size_mm=float(body.get("size_mm", 100.0)),
                    color=str(body.get("color", "#f97316")),
                )
            elif op == "remove":
                self.harness.sandbox.remove_object(str(body.get("id", "")))
            elif op == "cliff_zone":
                self.harness.sandbox.set_cliff_zone(bool(body.get("enabled", False)))
            elif op == "reset":
                self.harness.sandbox.reset_default_scene()
                self.harness.body._pose = BodyPose()
                self.harness.body._manual_tof_override = False
                self.harness.body._manual_cliff_override = False
                self.harness.sandbox._sync_sensors_and_observations(
                    self.harness.body.pose(), emit_bus=False
                )
                await self.harness.body._emit_pose()
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(self.harness.sandbox.snapshot(self.harness.body.pose())).encode("utf-8"),
            )

        if method == "POST" and path == "/api/estop":
            active = bool(body.get("active", False))
            self.harness.body.set_estop(active)
            if active:
                await self.harness.bus.emit(
                    "action.cancel", "dashboard", {"reason": "estop"}, priority=EventPriority.CRITICAL
                )
            await self.harness.body._emit_pose()
            return "200 OK", "application/json", json.dumps({"estop_active": active}).encode("utf-8")

        if method == "POST" and path == "/api/obstacle":
            tof = body.get("front_tof_mm")
            cliff = body.get("cliff")
            self.harness.body.set_simulated_obstacle(
                front_tof_mm=float(tof) if tof is not None else None,
                cliff=bool(cliff) if cliff is not None else None,
            )
            await self.harness.body._emit_pose()
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps({"sensors": self.harness.body.sensors().to_dict()}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/observe":
            await self.harness.bus.emit(
                "perception.observation",
                source="dashboard",
                payload={
                    "label": str(body.get("label", "object")),
                    "distance_mm": float(body.get("distance_mm", 800.0)),
                    "relative_bearing_deg": float(body.get("relative_bearing_deg", 0.0)),
                },
                priority=EventPriority.NORMAL,
            )
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps({"observations": self.harness.world.active_observations()}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/memory/forget":
            topic = str(body.get("topic", "")).strip()
            await self.harness.bus.emit(
                "action.cancel", "dashboard", {"reason": "forget"}, priority=EventPriority.CRITICAL
            )
            count, tomb = self.harness.memory.forget(topic)
            self.harness.prompts.clear()
            self.harness.consolidator.clear_pending()
            self._memory_cache = None
            return (
                "200 OK",
                "application/json",
                json.dumps({"forgotten_count": count, "tombstone": tomb.__dict__}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/memory/consolidate":
            rep = await self.harness.consolidator.consolidate(emit_when_empty=True)
            await self.harness.bus.drain()
            return "200 OK", "application/json", json.dumps(rep.to_dict()).encode("utf-8")

        if method == "POST" and path == "/api/audio/sfx":
            snd = str(body.get("sound", "ack_chirp")).strip().lower()
            samples = await self.harness.audio.play_sfx(snd)
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps({"sound": snd, "samples": int(len(samples))}).encode("utf-8"),
            )

        if method == "POST" and path == "/api/audio/speak":
            txt = str(body.get("text", "")).strip()
            speech = await self.harness.audio.speak(txt)
            await self.harness.bus.drain()
            return (
                "200 OK",
                "application/json",
                json.dumps(
                    {
                        "text": speech.text,
                        "duration_s": round(speech.duration_s, 3),
                        "backend": speech.backend_name,
                    }
                ).encode("utf-8"),
            )

        if method == "POST" and path == "/api/audio/mute":
            muted = bool(body.get("muted", False))
            self.harness.audio.muted = muted
            return "200 OK", "application/json", json.dumps({"muted": muted}).encode("utf-8")

        if method == "POST" and path == "/api/audio/barge_in":
            await self.harness.bus.emit(
                "audio.barge_in",
                source="dashboard",
                payload={"rms": 0.25},
                priority=EventPriority.CRITICAL,
            )
            await self.harness.bus.drain()
            return "200 OK", "application/json", json.dumps({"barge_in": True}).encode("utf-8")

        return "404 Not Found", "application/json", b'{"error":"not found"}'
