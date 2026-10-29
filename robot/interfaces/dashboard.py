"""Zero-dependency async HTTP diagnostic dashboard and REST API server."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from robot.body.hal import BodyPose
from robot.core.events import Event, EventPriority
from robot.core.service import ServiceHealth

if TYPE_CHECKING:
    from robot.main import RobotHarness

logger = logging.getLogger(__name__)


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
            <button class="dpad-btn" onclick="runAction('turn',{angle_deg:30})">◀ Turn 30°</button>
            <button class="dpad-btn dpad-stop" onclick="runAction('stop',{})">STOP</button>
            <button class="dpad-btn" onclick="runAction('turn',{angle_deg:-30})">Turn 30° ▶</button>
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
    const camVer = t.camera_verification ? ` | cam_check: ${esc(t.camera_verification)}` : '';
    const userLine = t.user ? `<div class="chat-bubble chat-u"><strong>You:</strong> ${esc(t.user)}</div>` : '';
    const label = t.origin === 'autonomous' ? 'Sprocket (Autonomous)' : 'Sprocket';
    return `${userLine}<div class="chat-bubble chat-r">` +
      `<div><strong>${label}:</strong> ${esc(t.say)}</div>` +
      `<div class="chat-meta">[emotion:${esc(t.emotion)} | sound:${esc(t.sound)} | actions:${esc(JSON.stringify(t.actions))}${camVer} | ${esc(t.backend)} ${esc(t.latency_ms)}ms]</div></div>`;
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
    drawFace(animFace);
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
      (g.status === 'continue' ? ` · ${g.phase || 'working'} ${g.target || ''}` : '');
    const autoInd = document.getElementById('autoIndicator');
    if (autoInd) {
      autoInd.style.background = a.paused ? 'var(--amber)' : (a.enabled ? 'var(--accent)' : 'var(--text-muted)');
      autoInd.style.boxShadow = `0 0 8px ${autoInd.style.background}`;
    }

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

requestAnimationFrame(stepClientAnimation);
pollLoop();
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
                    "camera_verification": p.get("camera_verification"),
                    "status": p.get("status", "completed"),
                    "goal_state": p.get("goal_state"),
                    "backend": p.get("backend", "unknown"),
                    "latency_ms": p.get("latency_ms", 0.0),
                }
            )

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
        _query: dict[str, list[str]],
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
            self.harness.context_builder.clear_turns()
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
