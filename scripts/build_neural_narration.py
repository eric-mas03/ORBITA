"""Build scene-aligned neural narration, then replace audio without reencoding video.

Run with the workspace-local edge-tts installation in outputs/voice_runtime.
Cached source audio avoids a network request on subsequent runs. Only the public
presentation script is sent to the speech service when sources are missing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / 'presentation'
CANDIDATES = PRESENTATION / 'voice_candidates'
VOICE = 'en-US-AvaMultilingualNeural'
FFMPEG = os.environ.get('ORBITA_FFMPEG') or shutil.which('ffmpeg')
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        FFMPEG = 'ffmpeg'
ORIGINAL = ('Meet Orbita: a research prototype for future crew health support. '
            'It compares earlier forecasts with real observations, then shows what may happen next, including uncertainty. '
            'Different signals learn patterns through a shared memory, with a forecast for each measurement. '
            'NASA data support the research: astronaut blood panels, ground-based cardiovascular studies, and space-station radiation. '
            'To improve prediction quality, the next step is more participants and more frequent measurements.')
COMPARISON = 'Compare earlier forecasts with real observations. Then look ahead, with uncertainty.'
NASA = 'NASA contributes astronaut blood panels, ground studies, and space station radiation.'


def run(*args):
    result = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])


async def ensure_sources():
    sources = [('ava', ORIGINAL), ('ava-comparison', COMPARISON), ('ava-nasa', NASA)]
    missing = [(name, text) for name, text in sources if not (CANDIDATES / f'{name}.mp3').exists() or not (CANDIDATES / f'{name}-timings.json').exists()]
    if not missing:
        return
    sys.path.insert(0, str(ROOT / 'outputs/voice_runtime'))
    import edge_tts

    async def generate(name, text):
        timings = []
        speech = edge_tts.Communicate(text, VOICE, rate='-3%', pitch='+0Hz', connect_timeout=15, receive_timeout=30)
        with (CANDIDATES / f'{name}.mp3').open('wb') as audio:
            async for item in speech.stream():
                if item['type'] == 'audio':
                    audio.write(item['data'])
                elif item['type'] == 'SentenceBoundary':
                    timings.append(item)
        (CANDIDATES / f'{name}-timings.json').write_text(json.dumps(timings, indent=2), encoding='utf8')

    await asyncio.gather(*(generate(name, text) for name, text in missing))


def span(name, sentence=None):
    rows = json.loads((CANDIDATES / f'{name}-timings.json').read_text(encoding='utf8'))
    if sentence is not None:
        rows = [rows[sentence]]
    return rows[0]['offset'] / 1e7, (rows[-1]['offset'] + rows[-1]['duration']) / 1e7


def main():
    CANDIDATES.mkdir(exist_ok=True)
    asyncio.run(ensure_sources())
    # Preserve the first Windows narration as an explicit comparison asset.
    old = PRESENTATION / 'narration-windows-30s.wav'
    if not old.exists():
        shutil.copyfile(PRESENTATION / 'narration-30s.wav', old)
    clips = [
        ('ava', 0, .25, 4.60, 'Introduction'),
        ('ava-comparison', None, 5.05, 5.80, 'Forecast comparison'),
        ('ava', 2, 11.25, 6.30, 'Architecture'),
        ('ava-nasa', None, 18.10, 5.70, 'NASA evidence'),
        ('ava', 4, 24.10, 5.70, 'More frequent measurements'),
    ]
    inputs = ['ava', 'ava-comparison', 'ava-nasa']
    args = []
    for name in inputs:
        args.extend(['-i', CANDIDATES / f'{name}.mp3'])
    filters, manifest = [], []
    for index, (name, sentence, start, slot, label) in enumerate(clips):
        begin, end = span(name, sentence)
        speed = max(1., (end - begin) / slot)
        if speed > 1.12:
            raise ValueError(f'{label}: revise wording rather than speeding speech excessively ({speed:.3f})')
        duration = (end - begin) / speed
        filters.append(f'[{inputs.index(name)}:a]atrim=start={begin}:end={end},asetpts=PTS-STARTPTS,atempo={speed:.8f},afade=t=in:d=0.012,afade=t=out:st={max(0,duration-.020):.8f}:d=0.020,adelay={round(start*1000)}:all=1[s{index}]')
        manifest.append({'scene': label, 'start_seconds': start, 'end_seconds': start+duration, 'tempo_factor': speed, 'source': f'{name}.mp3'})
    filters.append(''.join(f'[s{i}]' for i in range(5)) + 'amix=inputs=5:duration=longest:dropout_transition=0:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,apad,atrim=duration=30[out]')
    narration = PRESENTATION / 'narration-neural-30s.wav'
    run(*args, '-filter_complex', ';'.join(filters), '-map', '[out]', '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', narration)
    with wave.open(str(narration), 'rb') as audio:
        params, frames = audio.getparams(), audio.readframes(audio.getnframes())
    # Resampling can leave a few milliseconds of filter delay. Pad only silence
    # at the tail, where the last spoken scene has already finished by 29.8 s.
    desired = 30 * params.framerate * params.nchannels * params.sampwidth
    if abs(len(frames) - desired) > .05 * params.framerate * params.nchannels * params.sampwidth:
        raise ValueError('Unexpected narration duration after resampling')
    with wave.open(str(narration), 'wb') as audio:
        audio.setparams(params)
        audio.writeframes(frames[:desired].ljust(desired, b'\0'))
    video = PRESENTATION / 'orbita-30s-v3-neural.mp4'
    run('-i', PRESENTATION / 'orbita-30s-v2.mp4', '-i', narration, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-t', '30', '-movflags', '+faststart', video)
    # Compare compressed video packets: the visual presentation is unchanged.
    def video_hash(path):
        result = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-map', '0:v:0', '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    assert video_hash(video) == video_hash(PRESENTATION / 'orbita-30s-v2.mp4')
    shutil.copyfile(narration, PRESENTATION / 'narration-30s.wav')
    shutil.copyfile(video, PRESENTATION / 'orbita-30s.mp4')
    script = '\n\n'.join([ORIGINAL.split('. ')[0]+'.', COMPARISON, ORIGINAL.split('. ')[2]+'.', NASA, ORIGINAL.split('. ')[4]]) + '\n'
    (PRESENTATION / 'narration-neural.txt').write_text(script, encoding='utf8')
    (PRESENTATION / 'narration.txt').write_text(script, encoding='utf8')
    metadata = {'engine': 'Microsoft Edge neural speech via edge-tts', 'voice': VOICE, 'language': 'en-US', 'synthetic_voice': True, 'duration_seconds': 30, 'sample_rate': 48000, 'loudness_target_lufs': -16, 'true_peak_target_dbfs': -1.5, 'scenes': manifest, 'video_packets_unchanged': True, 'video_file': video.name, 'narration_sha256': hashlib.sha256(narration.read_bytes()).hexdigest()}
    (PRESENTATION / 'narration-neural-manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    video_manifest = PRESENTATION / 'video-manifest.json'
    current = json.loads(video_manifest.read_text(encoding='utf8'))
    current.update(versioned_copy=video.name, narration='narration-neural-30s.wav', narration_voice=VOICE, narration_revision='scene-aligned-neural')
    video_manifest.write_text(json.dumps(current, indent=2), encoding='utf8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
