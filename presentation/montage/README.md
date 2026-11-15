# ORBITA — editorial research film

Final film: `../orbita-30s-v5-nozoom.mp4`, also copied to `../orbita-30s.mp4`.

This is an offline video composition, not a screen recording of the website's play mode. It has nine individually designed shots, edited transitions, fixed product framing, animated charts, a rotating conceptual neural network, moving signal tokens, staggered NASA cards and kinetic closing typography. English neural narration is synchronized to five narrative beats and explicitly introduces a predictive digital twin with chatbot assistance. Four voice editions are described in `../voice-editions/README.md`. Original, low-volume procedural swells accompany the transitions.

| Start | Editorial shot |
|---|---|
| 0.00 s | ORBITA opening, orbital motion and a fixed upright product image |
| 1.80 s | Fixed product overview, cropped to signals and forecasts |
| 5.05 s | Real observations versus historical/future model forecasts; curve reveal |
| 8.10 s | Forecast range and a research-chatbot explanation using actual +4 h q10/q50/q90 |
| 11.25 s | Independent sources, encoders, shared temporal memory and separate heads |
| 14.00 s | Rotating conceptual latent/network illustration |
| 16.00 s | Separate source forecast cards using actual model outputs |
| 18.10 s | NASA Inspiration4, UTMB/CFT70 and RadLab evidence cards |
| 24.00 s | More participants and more frequent paired measurements; closing title |

## Sources and boundaries

- `forecast-snapshot.json` freezes the actual local `/api/presentation` response. The chart and range values come directly from these outputs. No good-looking replacement curves are invented.
- `product-screen-source.png` is a frozen image of the real product. The film crops away navigation and tour controls.
- The shared neural network is explicitly a conceptual illustration, not a plot of real latent patient embeddings.
- Independent sources are never presented as one astronaut: wearable data are terrestrial, NASA cardiovascular data are visit-based ground analogues, and RadLab data describe the ISS instrument environment.
- Inspiration4 blood panels are identified as the earlier exploratory biomarker prototype, rather than as continuous hourly mission telemetry.
- The final denser timeline is a research goal, not a demonstrated accuracy improvement.

## Reproduce

From the repository root, with Pillow, NumPy, Segoe UI fonts and FFmpeg available:

```powershell
python scripts/render_montage.py --stills
python scripts/render_montage.py
```

Set `ORBITA_FFMPEG` to override the local FFmpeg executable path. No browser or dashboard server is required to render the frozen presentation assets. Neural narration is read from `../voice-editions/ava-30s.wav`; its source script and voice settings are in the parent presentation folder.

`montage-manifest.json` records the timeline and forecast snapshot hash. `captions-en.srt` provides the narration as optional English subtitles. `encoder.log` contains the export log.

## Validation

The exported MP4 contains 900 frames at 30 fps, 1920 × 1080, exactly 30 seconds. Audio decodes successfully, remains below clipping, and lasts 30 seconds. Encoded frames from every shot were visually inspected in `encoded-contact-sheet.png`. Pixel-change checks confirm movement in the neural network, forecast reveal and closing data-density animation. Details are in `validation.json`.
