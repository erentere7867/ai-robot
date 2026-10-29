# Runtime hardening and harness resource use

The changes in this pass target the Python harness. Existing model weights, quantization,
GPU settings, inference context size and generation limits were not retuned. The Pi profile
contains only harness limits. Measurements below were made on Windows/Python 3.15, not a Pi.

## Cancellation and lifecycle

- All console, dashboard and microphone messages use the same attention path.
- Safety observers run before normal event queues and diagnostic recording. A broken observer
  cannot prevent the other safety observers from receiving a critical event.
- Stop invalidates the active turn, delayed inference, pending speech production, remaining
  plan steps and queued commands. New input replaces the previous conversational turn.
- Requests have individual correlation IDs and a maximum of eight outstanding HTTP/console
  waiters. One request cannot consume another request's response.
- The simulator integrates motion in steps, continuously checks stop/cliff/obstacle/pickup
  conditions, reports partial movement and uses the same checked primitives for tricks.
- Invalid or non-finite motor inputs are rejected. Joint limits are applied to trick segments.
- Emotion-only default gestures use the head; they do not automatically drive or spin the base.
- Real playback completion opens the follow-up window. A guessed speech-duration task is no
  longer created for every turn.
- Idle/consolidation processing runs without a dashboard. Startup failure unwinds partially
  started services, and shutdown cancels owned work before closing the database.
- Headless mode runs without input(), console event subscribers or a terminal. SIGTERM is
  handled on supported platforms.

## Resource bounds

| Resource | Bound/default |
| --- | --- |
| Subscriber event queues | 256; Pi profile 64 |
| Event-type subscriber matching cache | 64 event types; invalidated on subscription changes |
| Dashboard observer queue | 64, disposable diagnostics |
| HTTP clients | 8 |
| HTTP body / headers | 16 KiB / 8 KiB; bounded read time |
| Per-action execution deadline | at most 30 seconds; cancellation remains immediate |
| Inference response body / JSON nesting | 64 KiB / 16 levels in validated model output |
| Current user message | 4,096 characters, additionally checked against context budget |
| Spoken response / memory fact | 400 / 2,048 characters |
| Recent conversation messages | 8; Pi profile 4; each message is size bounded |
| Pending consolidation turns | 32; assistant text is not duplicated |
| Spatial observations | 128; label 160 characters; TTL at most 120 seconds |
| Dashboard memories/tombstones/episodes | 100 / 100 / 10 |
| SQLite page cache | approximately 2 MiB; mmap disabled |
| Retained episodic log rows | 1,000; semantic facts and their versions are preserved |
| Speech synthesis | one native job, one producer, at most 16 ordered utterances; cancelled on interruption |
| Transcription | one active utterance plus one latest pending utterance |
| Microphone frame queue | 16 frames; oldest frames dropped on overload |
| Recording queue | 256 events, independent writer thread |
| Recording files per session | four files of approximately 8 MiB each |

Normal reliable event overload is rejected before dispatch instead of waiting on a subscriber's
own full queue. Telemetry eviction removes only disposable telemetry; it cannot evict a
cancellation. Critical safety handling does not depend on diagnostic queue capacity.

The dashboard refreshes twice per second while visible and less often while hidden. It caches
capabilities and bounded memory snapshots, and invalidates memory snapshots using store revisions.
It never exports all historical facts during a background refresh. Consolidation uses indexed
subject lookups instead of loading all active facts for each candidate.

The optional face renderer is imported only in the standalone face process. Core installation
requires NumPy, httpx, PyYAML and MuJoCo; pygame and sounddevice are optional extras. Automatic ASR never
starts a second large language model process. A whisper.cpp adapter is available when explicitly
configured/installed; no speech model was downloaded during this work.

## Memory and recordings

Broad forget requests tombstone the matching canonical subjects as well as their broad topic.
Implicit consolidation cannot resurrect them. An explicit remember request can restore a fact.
Forgetting clears recent prompt history and pending consolidation to prevent immediate recall
from stale working context. Replacing a fact and retiring its prior version is one transaction.
Unrelated retrieval results are no longer padded with arbitrary recent facts.

Forget is logical removal from active semantic recall. Existing historical database rows and
diagnostic recordings may still contain prior text. It is not a claim of forensic erasure.
Pi-profile diagnostic recording is off by default. Enabled recording is best effort: drops,
writer failures and the per-session size cap are observable in the dashboard payload. Partial
recordings cannot provide a complete replay. Previously saved sessions are not deleted.

## Validation and remaining boundaries

Regression tests exercise late inference after stop, multi-action cancellation, rapid replacement
of requests, critical delivery under overload, partial movement, custom joint limits, encoder
precision, startup failure, audio task ownership, nonblocking VAD, actual speech completion,
memory resurrection, transaction rollback, bounded context and observations, HTTP validation,
replay isolation and recording rotation.

The 1,000-turn benchmark exercises the dashboard, conversation history, observations, episodes
and event delivery without loading a model. Run it again on the actual Raspberry Pi to measure
its allocator, OS libraries, microphone, screen and hardware driver costs. It does not establish
physical emergency-stop latency or prove that the full robot is flawless.

Local validation: **57 tests passed** with resource/unraisable warnings treated as errors;
lint, formatting, Python 3.11 syntax checks, muted CLI smoke test, wheel import/bundled configuration,
and dashboard JavaScript/escaping checks passed.

| Local measurement | Before | After |
| --- | ---: | ---: |
| CPU time for 100 offline turns | 0.531 s | 0.141 s |
| Dashboard snapshot with 2,000 historical versions | 16.414 ms | 0.193 ms |
| RSS after that snapshot | 61.05 MiB | 54.68 MiB |

These are individual Windows runs using the same fixture, not statistically aggregated Pi results.
The separate 1,000-turn run ended at 57.91 MiB RSS, with 230 KiB of retained Python allocation
growth, 128 observations, four recent messages, 1,000 episodes, zero pending requests/commands,
empty subscriber queues and zero subscriber errors. See the JSON files under `artifacts/`.

The current body is simulated. A real motor controller needs its own watchdog and physical
emergency-stop enforcement. Energy-threshold VAD also does not provide acoustic echo cancellation;
speaker/microphone feedback must be tested on the assembled robot.
