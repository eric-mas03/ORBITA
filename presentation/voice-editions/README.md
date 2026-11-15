# ORBITA — voice editions, revision 5

All four complete films run for exactly 30 seconds and share identical 1080p/30 fps video packets. The dashboard images have fixed framing: no zoom, pan or tilt. Animated graphics and edited transitions remain. The introduction explicitly says **“a predictive digital twin with chatbot assistance”**, and the forecast-range shot demonstrates a research-chatbot explanation.

| Edition | Neural voice | Accent | Complete video in the parent folder |
|---|---|---|---|
| Ava | en-US-AvaMultilingualNeural | American English, female | orbita-30s-v5-ava.mp4 |
| Andrew | en-US-AndrewMultilingualNeural | American English, male | orbita-30s-v5-andrew.mp4 |
| Ryan | en-GB-RyanNeural | British English, male | orbita-30s-v5-ryan.mp4 |
| Guy | en-US-GuyNeural | American English, male | orbita-30s-v5-guy.mp4 |

`../orbita-voice-comparison.mp4` is a 24-second comparison: the same introduction, six seconds per voice, in the order Ava, Andrew, Ryan and Guy. The existing female voice remains the default until a different edition is chosen.

`script.txt` holds the shared English script. Each edition has cached source audio and speech boundaries, a 30-second aligned PCM master, a mixed soundtrack, an English SRT file and a manifest. The script was shortened to preserve natural pacing. Only Andrew's introduction requires a small additional tempo adjustment (3.1%); pitch is preserved. Voice generation uses the same +3% synthesis rate across editions.

## Reproduce

```powershell
python scripts/prepare_voice_editions.py --prepare
python scripts/render_montage.py
python scripts/prepare_voice_editions.py
```

The first command generates any missing public-script speech through the workspace-local edge-tts runtime and aligns the five scenes. The second renders the revised visuals once. The third copies that video stream into four voice editions and creates the comparison. Cached synthesis avoids repeat network requests.

Validation in `validation.json` confirms 900 frames at 30 fps per full edition, a 24-second comparison, audio in every scene, no clipped audio, and identical encoded video packets across voices. `../montage/nozoom-validation.json` records pixel-identical dashboard crops across time in both product shots.
