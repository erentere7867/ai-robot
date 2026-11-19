"""Exercise the dashboard's microphone lifecycle in Chrome with a real fake-device stream."""

import os
import shutil
from pathlib import Path

import pytest

from robot.interfaces.live import Live


@pytest.fixture
def mic_page():
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chrome is required for the microphone browser check")
    live = Live()
    live.update(voice=True)
    server = live.serve(port=0)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome,
            args=["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        page = browser.new_page(viewport={"width": 1100, "height": 850})
        page.add_init_script("""
            const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            window.micRequested = false;
            navigator.mediaDevices.getUserMedia = constraints => new Promise((resolve, reject) => {
                window.grantMic = async () => {
                    window.micStream = await getUserMedia(constraints);
                    resolve(window.micStream);
                };
                window.denyMic = () => reject(new Error('permission denied'));
                window.micRequested = true;
            });
        """)
        uploads = []
        page.on("request", lambda request: uploads.append(request) if request.url.endswith("/voice") else None)
        page.goto(f"http://127.0.0.1:{server.server_port}")
        page.locator("#talk").wait_for(state="visible")
        yield page, uploads
        browser.close()
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("event", ["pointerup", "pointerleave", "pointercancel"])
def test_mic_granted_after_release_is_stopped(mic_page, event):
    page, uploads = mic_page
    page.locator("#talk").dispatch_event("pointerdown")
    page.wait_for_function("window.micRequested")
    page.locator("#talk").dispatch_event(event)
    page.evaluate("window.grantMic()")
    page.wait_for_function("window.micStream !== undefined")
    page.evaluate("new Promise(resolve => setTimeout(resolve, 100))")
    states = page.evaluate("window.micStream.getTracks().map(track => track.readyState)")
    label = page.locator("#talk").inner_text()
    if evidence := os.environ.get("PTT_EVIDENCE"):
        page.evaluate("""states => {
            const overlay = document.createElement('div');
            overlay.textContent = 'Delayed mic grant after release | tracks: ' + states.join(', ');
            overlay.style = 'position:fixed;top:0;left:0;right:0;background:#fff;color:#111;padding:16px;z-index:9999';
            document.body.append(overlay);
        }""", states)
        page.screenshot(path=str(Path(evidence) / f"{event}.png"))
    assert states and all(state == "ended" for state in states), (states, label)
    assert label == "Hold to talk"
    assert uploads == []


def test_mic_held_until_grant_uploads_on_release(mic_page):
    page, uploads = mic_page
    page.locator("#talk").dispatch_event("pointerdown")
    page.evaluate("window.grantMic()")
    page.wait_for_function("document.getElementById('talk').textContent.includes('Listening')")
    page.locator("#talk").dispatch_event("pointerup")
    page.wait_for_function("window.micStream.getTracks().every(track => track.readyState === 'ended')")
    assert page.locator("#talk").inner_text() == "Hold to talk"
    assert len(uploads) == 1


def test_mic_permission_denied_stays_idle(mic_page):
    page, uploads = mic_page
    page.locator("#talk").dispatch_event("pointerdown")
    page.evaluate("window.denyMic()")
    page.wait_for_function("document.getElementById('heard').textContent.includes('permission denied')")
    assert page.locator("#talk").inner_text() == "Hold to talk"
    assert uploads == []
