# ORBITA presentation package

**Current submission edition:** `submission/ORBITA-demo-ava.mp4`, with Andrew, Ryan and Guy alternatives in the same folder. All four films are exactly 30 seconds, use fixed upright product framing, explicitly describe a predictive digital twin with chatbot assistance, and end with a still ORBITA closing image plus a quiet voice-aligned sonic signature. They include the visible SI-generated-video/synthetic-voice label and descriptive MP4 metadata required by the supplied submission form. Ready-to-copy English fields are in `submission/submission-fields-en.md`. Public hosting links are still required; no external project was published or submitted by this local preparation.

The main dashboard is an English presentation of one crew-health research concept. It deliberately does not combine unrelated participants into one fictional astronaut.

Run `python scripts/dashboard.py` from the repository and open http://127.0.0.1:8765/. The previous research interface remains available at `/research`. Select **Play the 30-second story** for the guided sequence, or open `/?tour`.

## Ready-to-use assets

- `orbita-30s.mp4`: actual dashboard capture, 30 seconds, 1920 × 1080, 20 fps, H.264 with English AAC narration and visible captions.
- `architecture.drawio`: editable diagrams.net architecture.
- `architecture.svg`: animated, source-separated architecture used in the dashboard and video.
- `storyboard-30s.md`, `narration.txt`, `narration-30s.wav`: English script, storyboard and narration.
- `video-manifest.json`: capture settings and model artifact directory.

## What changed in the forecasts

The continuous heads now issue **96 genuinely trained queries**, every 15 minutes through +24 hours. They are not four sparse forecasts interpolated into a smooth curve. The 1/4/8/24-hour controls change the displayed extent of that trajectory.

The display defaults to **2 hours of history and 4 hours ahead**. History controls offer 1, 2 or 24 hours, independently of the 1/4/8/24-hour look-ahead controls. The model still receives its complete 24-hour input context; zooming the displayed history does not retrain or truncate the model. Historical error and coverage are recomputed for the displayed observations.

The historical display shows forecasts recomputed at earlier issue times: each point is a one-hour-ahead forecast using only the history available at that issue time. This is a retrospective replay with a fixed checkpoint, not a stored log of live predictions. The source-specific quantile band and median join genuine outputs in target-time order. They are never forced to the last observation to hide an estimation error at NOW.

Each current forecast uses a 24-hour context at 15-minute resolution with missing-data masks. Observations after NOW are shown only for comparison and are excluded from its inputs. Missing actual measurements are not connected across long gaps. Glucose is the initial example because it has 96 observed historical comparisons in this fixed held-out case; the same case's pulse has only eight.

Cardiovascular heads retain real visit-based histories and next-visit forecasts. Sparse NASA measurements are not made to look hourly. The **Explore measurements** dialog exposes all 32 study-specific outputs with search and category filters. All seven current source heads and their variables also remain available under **View all research outputs**, with a quantile table and measured historical error/coverage. Nominal q10–q90 coverage is 80%; measured coverage can differ substantially. There is no clinical validation.

## Data attribution

| Evidence | Actual use | Important distinction |
|---|---|---|
| NASA Inspiration4, OSD-575 and OSD-569 | Earlier exploratory biomarker model | Four crew members, discrete pre-/post-flight visits; not the current dense hourly head |
| NASA LSDA UTMB / CFT70 | Current cardiovascular heads | Earth-based bed-rest analogues; visit-based observations |
| NASA OSDR RadLab, ISS DosTel1 | Current environmental radiation head | Station instrument measurements, not personal dosimetry |
| BIG IDEAs wearable research | Current continuous glucose/pulse demonstration | Earth-based participants, not astronauts |

The primary goal is to collect more participants and more frequent **paired mission measurements** to test and improve prediction quality. More data does not guarantee better performance. The full-height guide panel opens along the right edge and keeps the desktop chart visible. It includes the selected measurement, its forecast range, contextual explanations and quick questions; on mobile it occupies the full screen with a pinned composer. The guide explains evidence and uncertainty; it does not invent causal meal/exercise simulations or provide treatment advice.

## Reproduce the dense campaign

```powershell
python scripts/train_source_campaign_v3.py --output-dir outputs/presentation_campaign --dense-quarter-hour --warm-start outputs/source_campaign_v3/source_twin.pt --epochs 24
```

This creates a separate campaign and preserves V3 artifacts and the existing train/development/calibration/test divisions. Development data select the checkpoint and ensemble; calibration data adjust intervals. The fixed presentation cases are not selected for low test error. Sparse cardiac targets remain visit-based.

## Capture and validate

With the local dashboard running and Chrome exposing a local debugging port at 9224:

```powershell
node scripts/check_presentation.cjs
node scripts/render_presentation_video.cjs
python -m pytest -q --basetemp=outputs/test_tmp_presentation_review --tb=short
```

The renderer uses local FFmpeg; set `ORBITA_FFMPEG` to override its default path. It captures 600 real UI frames against `ORBITA.setTime(seconds)` and combines them with the narration. No external rendering service is required. `build_architecture.py` regenerates the editable XML and SVG.

Validation completed: 165 Python tests passed; browser checks passed for all primary signals and horizons, theme switching, story pages, visit outputs and a 390-pixel mobile viewport. The exported MP4 was decoded successfully and frames from each scene were visually reviewed.

The updated video also demonstrates the measurement explorer, a NASA visit-based cardiac output, the 1/2-hour history views and the full-height ORBITA panel. Browser checks exercise all 32 outputs, history-dependent metrics, search/category filters, desktop side-by-side geometry, mobile drawer geometry, shortcuts and both themes.

## Updated neural narration

`orbita-30s-v3-neural.mp4` replaces the original Windows speech with the synthetic English voice `en-US-AvaMultilingualNeural`. The visual video packets are byte-identical to V2. Speech is aligned separately to the five scenes, with normalized loudness and a 30-second PCM master in `narration-neural-30s.wav`. The canonical `orbita-30s.mp4` and `narration-30s.wav` now use this narration. Earlier V2 and Windows audio are preserved for comparison.

Rebuild with `python scripts/build_neural_narration.py`. Cached source audio is under `voice_candidates/`; missing sources are generated using [edge-tts](https://github.com/rany2/edge-tts) installed locally under `outputs/voice_runtime`. Only the public presentation script is sent to the speech service. Scene timings, synthesis settings and audio hash are in `narration-neural-manifest.json`. Validation confirmed 30 seconds, speech in every scene, zero clipped PCM samples and successful video/audio decoding.

## Edited montage workflow

The film is composed offline by `scripts/render_montage.py`; see `montage/README.md` for the editable timeline, source data, source product image and validation. Use `--stills` for layout review, then render the full movie. The earlier V2 and V3 web-tour recordings are preserved as previous versions.
