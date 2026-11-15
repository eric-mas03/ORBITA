"""Prepare the public repository and GitHub Pages from audited local artifacts."""
from __future__ import annotations

import gzip
import hashlib
import json
import runpy
import re
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from astronaut_twin.presentation_forecasts import presentation_payload


def main():
    docs=ROOT/'docs';media=docs/'media';media.mkdir(exist_ok=True)
    (docs/'data').mkdir(exist_ok=True);(docs/'presentation').mkdir(exist_ok=True)
    backup=docs/'DEVELOPMENT_HISTORY.md'
    if not backup.exists():shutil.copyfile(ROOT/'README.md',backup)
    html=(ROOT/'dashboard/presentation.html').read_text(encoding='utf8')
    html=html.replace('href="/"','href="./"').replace('href="/presentation.css"','href="./presentation.css"').replace('src="/presentation.js"','src="./presentation.js"').replace('src="/assistant-connection.js"','src="./assistant-connection.js"').replace('="/presentation/','="./presentation/')
    html=html.replace('▷ Play the 30-second story','▷ Watch the 30-second film').replace('● Retrospective research replay','● Recorded research replay')
    html=html.replace('BIG IDEAs glucose and heart-rate data', '<a href="https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/" target="_blank" rel="noopener">BIG IDEAs v1.1.3 (ODC-By)</a> glucose and heart-rate data')
    html=html.replace('<button id="assistantLauncher"', '<footer style="max-width:1680px;margin:0 auto;padding:0 42px 30px;font-size:11px;display:flex;gap:24px;flex-wrap:wrap"><a href="https://github.com/eric-mas03/ORBITA">Source code &amp; data attribution ↗</a><a href="https://github.com/eric-mas03/ORBITA/blob/main/docs/PERSONAL_ASSISTANT.md">Connect your own assistant ↗</a><span>Research replay · Not live crew monitoring</span></footer><button id="assistantLauncher"')
    (docs/'index.html').write_text(html,encoding='utf8')
    js=(ROOT/'dashboard/presentation.js').read_text(encoding='utf8').replace("fetch('/api/presentation')","fetch('./data/presentation.json')").replace("$('#play').onclick=play;","$('#play').onclick=()=>{location.href='./demo.html'};")
    (docs/'presentation.js').write_text(js,encoding='utf8');shutil.copyfile(ROOT/'dashboard/presentation.css',docs/'presentation.css');shutil.copyfile(ROOT/'dashboard/assistant-connection.js',docs/'assistant-connection.js')
    (ROOT/'demo').mkdir(exist_ok=True)
    for voice in ['ava','andrew','ryan','guy']:shutil.copyfile(ROOT/f'presentation/submission/ORBITA-demo-{voice}.mp4',ROOT/f'demo/ORBITA-{voice}.mp4')
    shutil.copyfile(ROOT/'demo/ORBITA-ava.mp4',ROOT/'demo/ORBITA.mp4')
    (docs/'demo').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'demo/ORBITA.mp4',docs/'demo/ORBITA.mp4')
    payload=presentation_payload()
    (docs/'data/presentation.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
    (docs/'.nojekyll').write_text('',encoding='utf8')
    svg=(ROOT/'presentation/architecture.svg').read_text(encoding='utf8')
    svg=svg if 'SI-assisted diagram · OpenAI Codex' in svg else svg.replace('</svg>','<text x="1560" y="887" text-anchor="end" style="font-family:Segoe UI,Arial,sans-serif;font-size:13px;fill:#486c5b">SI-assisted diagram · OpenAI Codex</text></svg>')
    (docs/'presentation/architecture.svg').write_text(svg,encoding='utf8');(ROOT/'presentation/architecture.svg').write_text(svg,encoding='utf8')
    shutil.copyfile(ROOT/'presentation/architecture.drawio',docs/'presentation/architecture.drawio')
    for voice in ['ava','andrew','ryan','guy']:
        shutil.copyfile(ROOT/f'presentation/submission/ORBITA-demo-{voice}.mp4',media/f'ORBITA-demo-{voice}.mp4')
        shutil.copyfile(ROOT/f'presentation/voice-editions/{voice}-en.srt',media/f'ORBITA-demo-{voice}.srt')
        vtt=(media/f'ORBITA-demo-{voice}.srt').read_text(encoding='utf8')
        (media/f'ORBITA-demo-{voice}.vtt').write_text('WEBVTT\n\n'+re.sub(r'(?<=\d),(?=\d{3}\b)', '.', vtt),encoding='utf8')
    shutil.copyfile(ROOT/'presentation/submission/ORBITA-closing-card.png',media/'ORBITA-closing-card.png')
    (ROOT/'assets').mkdir(exist_ok=True)
    runpy.run_path(str(ROOT/'scripts/render_montage.py'))['render_frame'](3.5).save(ROOT/'assets/dashboard-preview.png')
    demo='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ORBITA · 30-second project demonstration</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#132b23;color:#eef5ef;font:16px/1.6 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:35px 26px}a{color:#9dddbb;text-decoration:none}.top{display:flex;justify-content:space-between;gap:20px;font-size:14px}h1{font-size:clamp(28px,4vw,48px);letter-spacing:-1.4px;margin:36px 0 6px}p{color:#b2ccbc;margin:0 0 25px}video{display:block;width:100%;aspect-ratio:16/9;background:#11261f;border:1px solid #355c45;border-radius:18px}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:22px 0}button{border:1px solid #46654e;border-radius:9px;background:#203d30;color:#e9f4ec;padding:9px 15px;font:inherit;cursor:pointer}button.active{background:#87cba6;color:#11281c}button:focus-visible,a:focus-visible{outline:3px solid #a4d4ff;outline-offset:4px}.download-button{display:inline-flex;align-items:center;justify-content:center;margin-left:auto;padding:11px 20px;border-radius:10px;background:#a7dfbe;color:#10271b;font-weight:700;box-shadow:0 5px 18px #0002}.download-button:hover{background:#c3efd3}.download-button small{margin-left:10px;font-weight:400}@media(max-width:650px){.download-button{width:100%;margin:8px 0 0}}.note{font-size:12px;color:#9ab8a5;margin-top:24px}footer{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-top:30px;font-size:14px}</style></head>
<body><main><div class="top"><a href="./">◌ ORBITA · Explore the prototype</a><a href="https://github.com/eric-mas03/ORBITA">View code ↗</a></div>
<h1>A predictive digital twin with chatbot assistance.</h1><p>30-second project demonstration · ViMas Tech · NASA Space Apps 2026</p>
<video id="film" controls playsinline preload="metadata" poster="./media/ORBITA-closing-card.png"><source src="./media/ORBITA-demo-ava.mp4" type="video/mp4"><track id="subtitles" kind="subtitles" srclang="en" label="English" src="./media/ORBITA-demo-ava.vtt">Your browser does not support this video. Use the download link below.</video>
<div class="controls"><span>English narration:</span><button class="active" data-voice="ava">Ava</button><button data-voice="andrew">Andrew</button><button data-voice="ryan">Ryan</button><button data-voice="guy">Guy</button><a class="download-button" id="download" href="./media/ORBITA-demo-ava.mp4" download="ORBITA-ava.mp4">↓ Download video <small>MP4 · 30 seconds</small></a></div>
<footer><span>Download includes the selected voice · No sign-in required</span><a href="./">Open the interactive research replay ↗</a></footer>
<p class="note">Research prototype · Not clinically validated. SI-generated presentation visuals and AI-assisted code: OpenAI Codex. Synthetic narration: Microsoft Edge neural speech through edge-tts. Data sources and original authors are credited in the repository.</p>
</main><script>const film=document.querySelector('#film');document.querySelectorAll('[data-voice]').forEach(b=>b.onclick=()=>{film.pause();film.src='./media/ORBITA-demo-'+b.dataset.voice+'.mp4';document.querySelector('#download').href=film.src;document.querySelector('#download').download='ORBITA-'+b.dataset.voice+'.mp4';document.querySelector('#subtitles').src='./media/ORBITA-demo-'+b.dataset.voice+'.vtt';document.querySelectorAll('[data-voice]').forEach(x=>x.classList.toggle('active',x===b));film.load()})</script></body></html>'''
    (docs/'demo.html').write_text(demo,encoding='utf8')
    notices=ROOT/'third_party';notices.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'data/raw/research_expansion_20261114/BIGIDEAS_1.1.3/LICENSE.txt',notices/'BIG_IDEAs-ODC-BY-1.0.txt')
    for campaign in ['presentation_campaign','source_campaign_v3']:
        source=ROOT/f'outputs/{campaign}/predictions.csv';target=source.with_suffix('.csv.gz')
        if source.exists() and (not target.exists() or target.stat().st_mtime<source.stat().st_mtime):
            with source.open('rb') as a,gzip.open(target,'wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
    resources=json.loads((ROOT/'presentation/submission/resources.json').read_text(encoding='utf8'))
    (docs/'SOURCES.md').write_text('# Data sources and resources\n\nIndependent studies are not synchronized measurements of one astronaut. NASA cardiovascular studies are ground analogues; ISS radiation is environmental instrument data; the glucose/pulse demonstration is terrestrial.\n\n'+ '\n\n'.join(f'## {r["link_text"]}\n\n{r["url"]}\n\n{r["use"]}' for r in resources),encoding='utf8')
    licensed='''# Data and third-party notices

The repository includes transformed research series and forecasts for a reproducible demonstration. Source observations were not generated by a generative model. The raw data archives, local browsing profiles, database journals and credentials are not distributed.

## BIG IDEAs wearable data

Derived wearable series in `outputs/*/series.json`, forecast artifacts and the published replay contain 15-minute aggregations of the BIG IDEAs Lab Glycemic Variability and Wearable Device Data v1.1.3. Original authors: Peter Cho, Juseong Kim, Brinnae Bent and Jessilyn Dunn. Source: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/ . Dataset DOI: https://doi.org/10.13026/aw6y-fc44 . Original publication: https://doi.org/10.1038/s41746-021-00465-w . The source database is licensed under the Open Data Commons Attribution License v1.0; its complete license is included in `third_party/BIG_IDEAs-ODC-BY-1.0.txt`. These aggregated/adapted outputs are not the original database. Changes include binning, missing-data masks, train/development/calibration/test roles, model-derived predictions and evaluation summaries. No new measurements are claimed.

## NASA research

NASA OSD-575/569 Inspiration4 panels were used in the earlier exploratory biomarker prototype. Current cardiac/vascular heads use NASA-hosted LSDA UTMB/CFT70 ground-analogue measurements; the radiation head uses ISS DosTel1 environmental instrument measurements from RadLab. Full source links and roles are in `docs/SOURCES.md`. Source material retains its original terms and attribution; no ownership or blanket relicensing of third-party data is claimed. NASA attribution does not imply endorsement, personal dosimetry or in-flight clinical validation.

## Software and media

Third-party packages retain their own licenses. Software/resource links are recorded in `docs/SOURCES.md`. System fonts are rendered in the presentation but their binaries are not redistributed. The videos carry visible SI labels and synthetic-voice metadata. No external music, astronaut likeness, NASA logo, flag or mission patch was generated for this presentation. `AI_DISCLOSURE.md` describes the actual use of AI/SI tools. No additional blanket license is assigned to independently copyrighted source resources by publication of this repository.
'''
    (ROOT/'THIRD_PARTY_NOTICES.md').write_text(licensed,encoding='utf8')
    # Hash the small frozen inference bundle and the four final labelled demos.
    paths=[]
    for campaign in ['presentation_campaign','source_campaign_v3']:
        paths.extend(p for p in (ROOT/'outputs'/campaign).iterdir() if p.is_file() and p.suffix in {'.json','.pt','.gz','.csv'} and p.name!='predictions.csv')
    paths.extend(media.glob('*.mp4'));paths.append(docs/'data/presentation.json')
    bundle={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
    (ROOT/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'repository':'https://github.com/eric-mas03/ORBITA','frozen_research_replay':True,'clinical_validation':False,'artifacts':bundle},indent=2),encoding='utf8')
    print(f'Prepared public Pages replay, four labelled videos and {len(bundle)} hashed artifacts.')


if __name__=='__main__':main()
