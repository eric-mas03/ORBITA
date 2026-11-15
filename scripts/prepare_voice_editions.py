"""Four scene-aligned neural voice editions of the fixed 30-second film."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'presentation/voice-editions'
FFMPEG = os.environ.get('ORBITA_FFMPEG') or shutil.which('ffmpeg')
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        FFMPEG = 'ffmpeg'
PARAGRAPHS=[
    'Meet Orbita: a predictive digital twin with chatbot assistance.',
    'Compare forecasts with observations, and ask the chatbot about uncertainty.',
    'Different signals learn patterns through a shared memory, with a forecast for each measurement.',
    'NASA contributes astronaut blood panels, ground studies, and station radiation.',
    'More astronaut data, more often, to improve and validate predictions.',
]
VOICES={'ava':('en-US-AvaMultilingualNeural','American English','Female'),'andrew':('en-US-AndrewMultilingualNeural','American English','Male'),'ryan':('en-GB-RyanNeural','British English','Male'),'guy':('en-US-GuyNeural','American English','Male')}
SLOTS=[(.25,4.60),(5.05,5.80),(11.25,6.30),(18.10,5.70),(24.10,5.70)]
SCRIPT=' '.join(PARAGRAPHS)


def run(*args):
    result=subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y',*map(str,args)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr[-3000:])


async def synthesize():
    sys.path.insert(0,str(ROOT/'outputs/voice_runtime'))
    import edge_tts
    semaphore=asyncio.Semaphore(1)
    async def generate(name,spec):
        mp3=OUT/f'{name}-raw.mp3';meta=OUT/f'{name}-raw.json'
        if mp3.exists() and mp3.stat().st_size>0 and meta.exists() and json.loads(meta.read_text(encoding='utf8')).get('text')==SCRIPT and json.loads(meta.read_text(encoding='utf8')).get('rate')=='+3%':return
        async with semaphore:
            for attempt in range(3):
                try:
                    rows=[]
                    talk=edge_tts.Communicate(SCRIPT,spec[0],rate='+3%',pitch='+0Hz',connect_timeout=15,receive_timeout=30)
                    with mp3.open('wb') as audio:
                        async for item in talk.stream():
                            if item['type']=='audio':audio.write(item['data'])
                            elif item['type']=='SentenceBoundary':rows.append(item)
                    if len(rows)!=5:raise ValueError(f'{name}: expected five complete sentences, got {len(rows)}')
                    meta.write_text(json.dumps({'text':SCRIPT,'voice':spec[0],'rate':'+3%','sentences':rows},indent=2),encoding='utf8')
                    print(f'Generated {name}: {mp3.stat().st_size} bytes',flush=True);return
                except Exception:
                    if attempt==2:raise
                    await asyncio.sleep(3*(attempt+1))
    await asyncio.gather(*(generate(name,spec) for name,spec in VOICES.items()))


def align(name,spec):
    source=json.loads((OUT/f'{name}-raw.json').read_text(encoding='utf8'));rows=source['sentences'];filters=[];scenes=[]
    for i,(row,(start,slot)) in enumerate(zip(rows,SLOTS)):
        begin=row['offset']/1e7;duration=row['duration']/1e7
        speed=max(1.,duration/slot)
        if speed>1.15:raise ValueError(f'{name}, scene {i}: wording must be shortened rather than rushing ({speed:.3f})')
        end=begin+duration;processed=duration/speed
        filters.append(f'[0:a]atrim=start={begin}:end={end},asetpts=PTS-STARTPTS,atempo={speed:.8f},afade=t=in:d=0.012,afade=t=out:st={processed-.02:.8f}:d=0.02,adelay={round(start*1000)}:all=1[s{i}]')
        scenes.append({'text':PARAGRAPHS[i],'start':start,'end':start+processed,'tempo_factor':speed})
    filters.append(''.join(f'[s{i}]' for i in range(5))+'amix=inputs=5:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000,apad=whole_len=1440000,atrim=end_sample=1440000[out]')
    target=OUT/f'{name}-30s.wav'
    run('-i',OUT/f'{name}-raw.mp3','-filter_complex',';'.join(filters),'-map','[out]','-ar','48000','-ac','1','-c:a','pcm_s16le',target)
    with wave.open(str(target),'rb') as f:params=f.getparams();frames=f.readframes(f.getnframes())
    expected=30*48000*2
    if abs(len(frames)-expected)>4800:raise ValueError('Unexpected audio duration')
    with wave.open(str(target),'wb') as f:f.setparams(params);f.writeframes(frames[:expected].ljust(expected,b'\0'))
    samples=np.frombuffer(target.read_bytes()[44:],dtype='<i2')
    peak=float(np.max(np.abs(samples.astype(np.int32))))/32768
    assert peak<.99
    metadata={'voice':spec[0],'accent':spec[1],'gender':spec[2],'duration_seconds':30,'synthetic':True,'scenes':scenes,'peak_dbfs':20*math.log10(peak),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (OUT/f'{name}-manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(f'Aligned {name}: max tempo {max(s["tempo_factor"] for s in scenes):.3f}',flush=True)


def video_hash(path):
    p=subprocess.run([FFMPEG,'-v','error','-i',str(path),'-map','0:v:0','-c:v','copy','-f','hash','-hash','sha256','-'],check=True,capture_output=True,text=True)
    return p.stdout.strip()


def assemble():
    submission=ROOT/'presentation/submission';submission.mkdir(exist_ok=True)
    base=ROOT/'presentation/orbita-30s-v6-submission.mp4';original_hash=video_hash(base)
    for name in VOICES:
        metadata=json.loads((OUT/f'{name}-manifest.json').read_text(encoding='utf8'))
        # A quiet sonic signature follows the final spoken phrase, rather than
        # cutting off speech or leaving an unintended silent tail.
        start=min(29.65,metadata['scenes'][-1]['end']+.12)
        duration=min(.65,29.97-start);rate=48000
        t=np.arange(round(duration*rate))/rate
        envelope=np.sin(np.pi*t/duration)**2
        signal=.022*envelope*(np.sin(2*np.pi*660*t)+.4*np.sin(2*np.pi*990*t))
        ending=np.zeros(30*rate,dtype='<i2');offset=round(start*rate)
        ending[offset:offset+len(signal)]=np.int16(signal*32767)
        chime=OUT/f'{name}-closing.wav'
        with wave.open(str(chime),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(ending.tobytes())
        mixed=OUT/f'{name}-mix.wav'
        run('-i',OUT/f'{name}-30s.wav','-i',ROOT/'presentation/montage/transition-sound-design.wav','-i',chime,'-filter_complex','[0:a][1:a][2:a]amix=inputs=3:duration=first:normalize=0,alimiter=limit=.89:level=false[out]','-map','[out]','-ar','48000','-ac','1','-c:a','pcm_s16le',mixed)
        target=submission/f'ORBITA-demo-{name}.mp4'
        disclosure=f'SI-generated presentation visuals and SI-assisted code: OpenAI Codex. SI-generated narration: Microsoft Edge neural voice {VOICES[name][0]}. Measured research source data were not synthesized.'
        run('-i',base,'-i',mixed,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-t','30','-movflags','+faststart+use_metadata_tags','-metadata',f'description={disclosure}','-metadata',f'comment={disclosure}','-metadata:s:a:0',f'title=SI-generated narration: {VOICES[name][0]}',target)
        assert video_hash(target)==original_hash
        print(f'Saved {target.name}',flush=True)
    # Keep the existing female voice as the default until the user chooses an
    # alternative. All four editions share exactly the same video packets.
    shutil.copyfile(submission/'ORBITA-demo-ava.mp4',base)
    shutil.copyfile(base,ROOT/'presentation/orbita-30s.mp4')
    shutil.copyfile(OUT/'ava-30s.wav',ROOT/'presentation/narration-30s.wav')
    (ROOT/'presentation/narration.txt').write_text('\n\n'.join(PARAGRAPHS)+'\n',encoding='utf8')
    path=ROOT/'presentation/video-manifest.json'
    current=json.loads(path.read_text(encoding='utf8'))
    current.update(versioned_copy=base.name,narration='voice-editions/ava-30s.wav',narration_voice=VOICES['ava'][0],narration_revision='submission-closing-v6',voice_variants={name:f'submission/ORBITA-demo-{name}.mp4' for name in VOICES},voice_comparison='orbita-voice-comparison.mp4',product_camera='fixed: no zoom, pan or tilt',closing_image_starts_at=28.25,si_disclosure='Visible SI-generated video / synthetic voice label and descriptive MP4 metadata')
    path.write_text(json.dumps(current,indent=2),encoding='utf8')
    shutil.copyfile(ROOT/'presentation/montage/closing-card.png',submission/'ORBITA-closing-card.png')


def comparison():
    rate=48000;block=6;names=list(VOICES);audio=np.zeros(rate*block*len(names),dtype='<i2');previews={}
    for i,name in enumerate(names):
        with wave.open(str(OUT/f'{name}-30s.wav'),'rb') as f:samples=np.frombuffer(f.readframes(5*rate),dtype='<i2').copy()
        audio[i*block*rate:i*block*rate+len(samples)]=samples;previews[name]=samples.astype(np.float32)/32768
    wav=OUT/'voice-comparison.wav'
    with wave.open(str(wav),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(audio.tobytes())
    target=ROOT/'presentation/orbita-voice-comparison.mp4';width,height,fps=1280,720,24
    command=[FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-i',str(wav),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-t','24','-movflags','+faststart',str(target)]
    with (OUT/'comparison-encoder.log').open('w') as errors:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors)
        fnt=lambda size,bold=False:ImageFont.truetype(str(Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf')),size)
        for frame in range(24*fps):
            t=frame/fps;index=min(3,int(t//6));name=names[index];local=t-index*6
            im=Image.new('RGB',(width,height),'#142d25');d=ImageDraw.Draw(im)
            d.text((65,42),'ORBITA / VOICE COMPARISON',font=fnt(16,True),fill='#8fcfb0')
            d.text((65,98),'Same script. Four voices.',font=fnt(47,True),fill='#f4f7f1')
            d.text((65,170),'Predictive digital twin with chatbot assistance',font=fnt(23),fill='#a7c3b2')
            d.text((65,250),name.title(),font=fnt(64,True),fill='#f4f7f1')
            d.text((67,335),f'{VOICES[name][1]} · {VOICES[name][2]} neural voice',font=fnt(23),fill='#a7c3b2')
            d.text((67,405),'“Meet Orbita: a predictive digital twin with chatbot assistance.”',font=fnt(24),fill='#d7e7dc')
            for j in range(78):
                sample=int((local-.25+j*.008)*rate);segment=previews[name][max(0,sample):max(0,sample)+640]
                rms=float(np.sqrt(np.mean(segment**2))) if len(segment) else 0
                bar=3+min(44,rms*180);x=66+j*14
                d.rounded_rectangle((x,510-bar,x+6,510+bar),3,fill='#72c29b')
            for i,other in enumerate(names):
                x=65+i*288;active=i==index
                d.rounded_rectangle((x,595,x+268,658),12,fill='#2b5942' if active else '#203d31',outline='#85c5a4' if active else '#3b5947')
                d.text((x+18,611),f'{i+1}. {other.title()}',font=fnt(23,True),fill='#f4f7f1' if active else '#9fbbaa')
            d.rectangle((0,714,width*t/24,720),fill='#78c9a1')
            process.stdin.write(im.tobytes())
        process.stdin.close()
        if process.wait():raise RuntimeError((OUT/'comparison-encoder.log').read_text())
    print(f'Saved {target.name}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.prepare:
        asyncio.run(synthesize())
        for name,spec in VOICES.items():align(name,spec)
        (OUT/'script.txt').write_text('\n\n'.join(PARAGRAPHS)+'\n',encoding='utf8')
    else:assemble()


if __name__=='__main__':main()
