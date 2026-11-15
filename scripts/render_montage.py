"""Offline editorial film: actual product/data, motion design and neural narration.

This does not open the website, run its tour, or capture browser playback.
Use --stills to review the layout; omit it to encode the 30-second film.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation/montage'
W, H, FPS = 1920, 1080, 30
SHOW_SI_LABEL = True
FFMPEG = os.environ.get('ORBITA_FFMPEG') or shutil.which('ffmpeg')
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        FFMPEG = 'ffmpeg'
INK, MUTED, MINT, CORAL, BLUE = '#203d36', '#6a8076', '#6ac4a2', '#dc8271', '#87adce'
WHITE, PALE, DARK = '#f8faf6', '#e1efe7', '#132a24'
DATA = json.loads((OUT / 'forecast-snapshot.json').read_text(encoding='utf8'))
PANELS = {p['id']: p for p in DATA['panels']}
GLUCOSE = next(v for v in PANELS['wearable']['variables'] if v['id'] == 'glucose')
HEART = next(v for v in PANELS['wearable']['variables'] if v['id'] == 'heart_rate')
RADIATION = PANELS['radiation']['variables'][0]
CARDIAC = next(v for v in PANELS['echo_cft']['variables'] if v['id'] == 'LVDD')


def clamp(x):
    return max(0., min(1., x))


def ease(x):
    x = clamp(x)
    return x*x*(3-2*x)


def color(s, alpha=255):
    if isinstance(s, tuple):
        return (*s[:3], alpha)
    return (*tuple(int(s[i:i+2], 16) for i in (1, 3, 5)), alpha)


@lru_cache(maxsize=80)
def font(size, bold=False):
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf')), size)


def text(image, xy, value, size=28, fill=INK, bold=False, alpha=1, anchor=None):
    if alpha <= 0:
        return
    d = ImageDraw.Draw(image, 'RGBA')
    d.text(xy, value, font=font(size, bold), fill=color(fill, round(clamp(alpha)*255)), anchor=anchor, spacing=10)


def enter_text(image, xy, value, size, fill, t, delay=0, bold=False):
    p = ease((t-delay)/.6)
    text(image, (xy[0], xy[1]+26*(1-p)), value, size, fill, bold, p)


def rounded(image, box, fill, radius=22, outline=None, width=1):
    ImageDraw.Draw(image, 'RGBA').rounded_rectangle(tuple(map(round, box)), radius, fill=fill, outline=outline, width=width)


@lru_cache(maxsize=8)
def background(dark=False):
    yy, xx = np.mgrid[0:H, 0:W]
    if dark:
        start = np.array([16., 35., 30.])
        glow = np.exp(-((xx-1380)**2/(730**2)+(yy-430)**2/(610**2)))[..., None]
        pixels = start+glow*np.array([17., 36., 25.])
    else:
        start = np.array([246., 248., 243.])
        glow = np.exp(-((xx-1500)**2/(660**2)+(yy-300)**2/(700**2)))[..., None]
        pixels = start+glow*np.array([-12., -3., -8.])
    result = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), 'RGB')
    d = ImageDraw.Draw(result, 'RGBA')
    for x in range(40, W, 80):
        for y in range(35, H, 80):
            d.ellipse((x,y,x+2,y+2), fill=color('#bfd7c9', 14 if dark else 35))
    return result


def logo(im, x=100, y=62, dark=False, size=30):
    d = ImageDraw.Draw(im, 'RGBA')
    d.ellipse((x,y+3,x+31,y+34), outline=color(MINT), width=2)
    d.arc((x-5,y+11,x+36,y+26), 8, 185, fill=color(MINT), width=2)
    text(im, (x+49,y-1), 'ORBITA', size, WHITE if dark else INK, True)


def eyebrow(im, xy, value, dark=False):
    text(im, xy, value, 19, MINT if dark else '#367c65', True)


def pill(im, xy, value, dark=False, accent=MINT, size=20):
    f = font(size)
    width = f.getlength(value)+34
    rounded(im,(xy[0],xy[1],xy[0]+width,xy[1]+42),color(accent, 25 if dark else 35),21, color(accent,90))
    text(im,(xy[0]+17,xy[1]+6),value,size,WHITE if dark else INK)
    return width


def footer(im, value, dark=False):
    text(im,(100,1018),value,18,'#9bb6a7' if dark else MUTED)


def dash(d, points, fill, width=3, on=12, off=8):
    phase = 0.
    drawing = True
    remaining = float(on)
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if not length:
            continue
        at=0.
        while at<length:
            step=min(remaining,length-at)
            p=(a[0]+dx*at/length,a[1]+dy*at/length)
            q=(a[0]+dx*(at+step)/length,a[1]+dy*(at+step)/length)
            if drawing:
                d.line((p,q),fill=fill,width=width)
            at+=step;remaining-=step
            if remaining<=1e-5:
                drawing=not drawing;remaining=float(on if drawing else off)


@lru_cache(maxsize=4)
def product_asset():
    source = Image.open(OUT/'product-screen-source.png').convert('RGB')
    # A clean editorial crop, without browser/site navigation, play or chat controls.
    return source.crop((108,215,1800,974))


def screen(im, box):
    # Product shots use a fixed camera: no digital zoom, pan or rotation.
    x,y,w,h=map(round,box)
    crop=product_asset().resize((w,h),Image.Resampling.LANCZOS)
    mask=Image.new('L',(w,h));ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),24,fill=255)
    shadow=Image.new('RGBA',im.size);rounded(shadow,(x+5,y+18,x+w+5,y+h+20),(0,30,20,25),28)
    im.paste(Image.alpha_composite(im.convert('RGBA'),shadow.filter(ImageFilter.GaussianBlur(19))).convert('RGB'))
    im.paste(crop,(x,y),mask)


def chart(im, box, variable=GLUCOSE, horizon=4, history=2, reveal=1., compact=False, dark=False):
    x,y,w,h=box
    d=ImageDraw.Draw(im,'RGBA')
    rounded(im,(x,y,x+w,y+h),color('#203b32' if dark else '#ffffff',255),24,color('#3a5748' if dark else '#dbe6dd'),1)
    L,R,T,B=(65,35,50,68) if not compact else (25,25,22,30)
    X=lambda z:x+L+(z+history)/(history+horizon)*(w-L-R)
    rows=[r for r in variable['historical'] if r['x']>=-history]+[r for r in variable['future'] if r['x']<=horizon]
    obs=[r for r in variable['observed'] if -history<=r['x']<=horizon and r['value'] is not None]
    vals=[r['value'] for r in obs]+[r[k] for r in rows for k in ['q10','q90']]
    low,high=min(vals),max(vals);pad=(high-low)*.14 or 1;low-=pad;high+=pad
    Y=lambda z:y+h-B-(z-low)/(high-low)*(h-T-B)
    for i in range(4):
        value=low+(high-low)*i/3
        if not compact:
            d.line((x+L,Y(value),x+w-R,Y(value)),fill=color('#335344' if dark else '#e4ebe5'),width=1)
            text(im,(x+L-15,Y(value)-14),f'{value:.0f}',19,'#a7bfb0' if dark else MUTED,anchor='ra')
    visible=rows[:max(0,min(len(rows),int(len(rows)*reveal)+1))]
    if len(visible)>1:
        band=[(X(r['x']),Y(r['q90'])) for r in visible]+[(X(r['x']),Y(r['q10'])) for r in reversed(visible)]
        d.polygon(band,fill=color(MINT,48 if not dark else 45))
    points=[(X(r['x']),Y(r['value'])) for r in obs]
    if len(points)>1:
        d.line(points,fill=color(CORAL),width=4,joint='curve')
    if len(visible)>1:
        dash(d,[(X(r['x']),Y(r['q50'])) for r in visible],color('#25856c' if not dark else MINT),4)
    dash(d,[(X(0),y+T-13),(X(0),y+h-B+6)],color('#4c6858' if not dark else '#a8c3b1'),2,7,7)
    if not compact:
        text(im,(X(0),y+17),'FORECAST ISSUED',15,'#355c48' if not dark else '#c0d8c8',True,anchor='ma')
        for z,label in [(-history,f'−{history} h'),(0,'NOW'),(horizon/2,f'+{horizon/2:g} h'),(horizon,f'+{horizon} h')]:
            text(im,(X(z),y+h-35),label,19,'#a6beb0' if dark else MUTED,anchor='ma')
    return X(0)


def legend(im, x,y):
    d=ImageDraw.Draw(im,'RGBA')
    d.line((x,y+13,x+38,y+13),fill=color(CORAL),width=4);text(im,(x+50,y-3),'Observed',23,MUTED)
    dash(d,[(x+220,y+13),(x+258,y+13)],color('#25856c'),4);text(im,(x+272,y-3),'Predicted median',23,MUTED)
    rounded(im,(x+550,y+5,x+586,y+23),color(MINT,65),4);text(im,(x+600,y-3),'q10–q90 range',23,MUTED)


def opening(t):
    im=background(True).copy();d=ImageDraw.Draw(im,'RGBA');logo(im,dark=True,size=35)
    for radius in [325,440,565]:
        d.ellipse((1430-radius,490-radius,1430+radius,490+radius),outline=color(MINT,28),width=1)
    angle=t*.24
    for i,r in enumerate([325,440,565]):
        a=angle+i*1.3;px,py=1430+r*math.cos(a),490+r*math.sin(a)
        d.ellipse((px-5,py-5,px+5,py+5),fill=color(MINT,170))
    enter_text(im,(110,265),'See change.\nLook ahead.',109,WHITE,t,.12,True)
    enter_text(im,(115,570),'Predictive digital twin.\nChatbot assistance for future crew health.',30,'#b1cabb',t,.35)
    screen(im,(860,300,960,435))
    pill(im,(114,748),'RESEARCH PROTOTYPE',True)
    footer(im,'Open research. A clearer view of what may come next.',True)
    return im


def product(t):
    im=background().copy();eyebrow(im,(110,60),'MEASUREMENTS + FORECASTS')
    enter_text(im,(110,103),'One clear view of change.',67,INK,t,0,True)
    pill(im,(1450,125),'32 measurement outputs')
    screen(im,(125,239,1670,748))
    return im


def comparison(t):
    im=background().copy();eyebrow(im,(110,65),'LOOK BACK TO JUDGE THE FORECAST')
    enter_text(im,(110,108),'Compare prediction with reality.',66,INK,t,0,True)
    text(im,(112,204),'Glucose · Earth-based wearable demonstration',25,MUTED)
    legend(im,150,270)
    chart(im,(125,335,1670,585),reveal=.38+.62*ease(t/1.5))
    footer(im,'Real measurements and genuine model outputs · Retrospective replay · No future observations enter the model.')
    return im


def uncertainty(t):
    im=background().copy();eyebrow(im,(110,65),'PREDICTIVE DIGITAL TWIN + CHATBOT ASSISTANCE')
    enter_text(im,(110,110),'Forecasts, with context.',76,INK,t,0,True)
    text(im,(112,220),'See uncertainty. Ask ORBITA what it means.',28,MUTED)
    chart(im,(110,340,1120,545))
    q=next(r for r in GLUCOSE['future'] if r['x']==4)
    x=1290
    rounded(im,(x,340,x+500,885),color('#ffffff'),24,color('#d9e5dc'))
    text(im,(x+35,375),'ASK ORBITA',27,'#35735b',True)
    text(im,(x+35,421),'Research chatbot · current forecast',20,MUTED)
    p=ease((t-.25)/.45)
    rounded(im,(x+35,475+18*(1-p),x+465,547+18*(1-p)),color(MINT,42),15)
    text(im,(x+55,497+18*(1-p)),'What does this range mean?',22,INK,alpha=p)
    p=ease((t-.65)/.5)
    y=574+22*(1-p)
    rounded(im,(x+35,y,x+465,y+168),color('#f0f5ef'),15,color('#dce7df'))
    answer=f"The median is {q['q50']:.1f} mg/dL.\nThe q10–q90 range is {q['q10']:.1f}–{q['q90']:.1f}.\nIt shows model uncertainty,\nnot a guaranteed outcome."
    text(im,(x+55,y+19),answer,21,INK,alpha=p)
    text(im,(x+35,779),'GLUCOSE · +4 HOURS',18,'#35735b',True)
    text(im,(x+35,818),f"q10 {q['q10']:.1f}     q50 {q['q50']:.1f}     q90 {q['q90']:.1f}",23,INK,True)
    for i,label in enumerate(['1 h','4 h','8 h','24 h']):
        pill(im,(285+i*155,925),label,accent=MINT if i==1 else '#a9bdb0',size=20)
    footer(im,'Research chatbot: evidence and uncertainty explanations · Earth wearable example · No clinical treatment advice.')
    return im


def bezier(a,b,c,d,u):
    return ((1-u)**3*a[0]+3*(1-u)**2*u*b[0]+3*(1-u)*u*u*c[0]+u**3*d[0],(1-u)**3*a[1]+3*(1-u)**2*u*b[1]+3*(1-u)*u*u*c[1]+u**3*d[1])


def neural_points(t,center,radius):
    points=[]
    for i in range(17):
        phi=math.acos(1-2*(i+.5)/17);theta=i*math.pi*(3-math.sqrt(5))
        xx,yy,zz=math.sin(phi)*math.cos(theta),math.cos(phi),math.sin(phi)*math.sin(theta)
        a=t*.22;rx=xx*math.cos(a)+zz*math.sin(a);rz=-xx*math.sin(a)+zz*math.cos(a)
        scale=1/(1-rz*.16)
        points.append((center[0]+rx*radius*scale,center[1]+yy*radius*.86*scale,rz))
    return points


def network(im,t,center,radius):
    points=neural_points(t,center,radius);d=ImageDraw.Draw(im,'RGBA')
    palette=[MINT,BLUE,CORAL]
    for i,a in enumerate(points):
        for j,b in enumerate(points[i+1:],i+1):
            if (i*7+j*11)%5<2 and math.hypot(a[0]-b[0],a[1]-b[1])<radius*1.6:
                alpha=35+int(35*(.5+.5*math.sin(t*1.1+i)))
                d.line((a[:2],b[:2]),fill=color('#91bca8',alpha),width=2)
                if (i+j)%4==0:
                    u=(t*.4+i*.13)%1;px=a[0]+(b[0]-a[0])*u;py=a[1]+(b[1]-a[1])*u
                    d.ellipse((px-3,py-3,px+3,py+3),fill=color(palette[i%3],200))
    for i,(x,y,z) in sorted(enumerate(points),key=lambda it:it[1][2]):
        r=7+(z+1)*4+(2*math.sin(t*2+i))
        d.ellipse((x-r*2.2,y-r*2.2,x+r*2.2,y+r*2.2),fill=color(palette[i%3],17))
        d.ellipse((x-r,y-r,x+r,y+r),fill=color(palette[i%3],int(140+(z+1)*45)))


def architecture(t):
    im=background(True).copy();eyebrow(im,(110,65),'FROM OBSERVATIONS TO A SHARED MEMORY',True)
    enter_text(im,(110,109),'Past signals become patterns.',68,WHITE,t,0,True)
    stages=[(105,'OBSERVE'),(515,'ENCODE'),(900,'SHARED MEMORY'),(1505,'FORECAST')]
    for x,label in stages:text(im,(x,266),label,19,'#9eb9a8',True)
    d=ImageDraw.Draw(im,'RGBA')
    sources=[('Daily health','Earth-based wearables','Wearable encoder',MINT),('Cardiac research','NASA ground analogues','Cardiac encoder',CORAL),('Station radiation','NASA ISS instrument','Radiation encoder',BLUE)]
    for i,(name,provenance,encoder,accent) in enumerate(sources):
        y=335+i*175
        rounded(im,(100,y,425,y+118),color('#203e32'),20,color(accent,90))
        d.line((115,y+24,115,y+92),fill=color(accent),width=4)
        text(im,(135,y+22),name,29,WHITE,True);text(im,(135,y+70),provenance,20,'#b5cebc')
        rounded(im,(500,y+17,730,y+100),color(accent,22),17,color(accent,90))
        text(im,(615,y+43),encoder,21,WHITE,True,anchor='ma')
        d.line((425,y+59,500,y+59),fill=color(accent,110),width=2)
        path=[bezier((730,y+59),(800,y+59),(805,550),(860,550),u/40) for u in range(41)]
        d.line(path,fill=color(accent,75),width=2)
        output=[bezier((1240,550),(1330,550),(1380,y+59),(1475,y+59),u/40) for u in range(41)]
        d.line(output,fill=color(accent,65),width=2)
        for j in range(3):
            u=(t*.32+j*.34+i*.11)%1
            px,py=bezier((730,y+59),(800,y+59),(805,550),(860,550),u)
            d.ellipse((px-5,py-5,px+5,py+5),fill=color(accent,200))
        rounded(im,(1475,y+4,1810,y+114),color('#203e32'),18,color(accent,65))
        text(im,(1500,y+23),['Health signals','Cardiac measures','Radiation'][i],26,WHITE,True)
        text(im,(1500,y+68),'Median + forecast range',18,'#aac4b2')
    d.ellipse((839,337,1261,759),fill=color('#294c3b',90),outline=color(MINT,55),width=2)
    network(im,t,(1050,548),176)
    text(im,(1050,808),'Shared temporal model',26,'#bed8c6',anchor='ma')
    footer(im,'Independent studies · Shared model weights · Separate timelines · Conceptual architecture animation',True)
    return im


def latent(t):
    im=background(True).copy();logo(im,dark=True)
    enter_text(im,(110,287),'Learning\nwhat changes.',96,WHITE,t,0,True)
    text(im,(116,560),'Separate signals.\nA shared space for temporal patterns.',30,'#afc9b7')
    pill(im,(115,716),'SHARED TEMPORAL MEMORY',True)
    d=ImageDraw.Draw(im,'RGBA')
    radius=292+12*ease(t/2)
    for r in [radius+45,radius+95]:d.ellipse((1360-r,520-r,1360+r,520+r),outline=color(MINT,30),width=1)
    network(im,t+3.5,(1360,520),radius)
    footer(im,'Conceptual view of the neural network · The dots are illustrative, not a projection of actual patient embeddings.',True)
    return im


def mini_forecast(im,variable,box,visits=False,accent=MINT,reveal=1):
    x,y,w,h=box;d=ImageDraw.Draw(im,'RGBA')
    full=variable['future'] if visits else [r for r in variable['future'] if r['x']<=4]
    rows=full[:max(2,int(len(full)*reveal))]
    values=[r[k] for r in full for k in ['q10','q90']];lo,hi=min(values),max(values);span=hi-lo or 1
    X=lambda i:x+i/(max(1,len(full)-1))*w;Y=lambda v:y+h-(v-lo)/span*h*.85-h*.075
    poly=[(X(i),Y(r['q90'])) for i,r in enumerate(rows)]+[(X(i),Y(r['q10'])) for i,r in reversed(list(enumerate(rows)))]
    d.polygon(poly,fill=color(accent,43));dash(d,[(X(i),Y(r['q50'])) for i,r in enumerate(rows)],color(accent),3,8,6)


def heads(t):
    im=background().copy();eyebrow(im,(110,67),'SOURCE-SPECIFIC OUTPUTS')
    enter_text(im,(110,110),'Separate forecasts. Visible uncertainty.',65,INK,t,0,True)
    for i,(name,source,var,visits,accent) in enumerate([('Glucose','Earth wearable research',GLUCOSE,False,'#418f73'),('Cardiac measures','NASA ground analogue',CARDIAC,True,'#bd786c'),('Station radiation','NASA ISS instrument',RADIATION,False,'#6d93b5')]):
        dy=(1-ease((t-i*.12)/.6))*45;x=110+i*575;y=345+dy
        rounded(im,(x,y,x+550,y+490),color('#ffffff'),25,color('#d9e5dc'))
        text(im,(x+30,y+34),name,34,INK,True);text(im,(x+30,y+91),source,21,MUTED)
        mini_forecast(im,var,(x+35,y+175,480,195),visits,accent,.65+.35*ease(t))
        text(im,(x+30,y+420),'Next visits' if visits else 'Next 4 hours',22,accent,True)
    footer(im,'Actual forecast outputs · Original measurement cadence · Independent source timelines')
    return im


def source_icon(d,kind,x,y,accent):
    if kind==0:
        d.line([(x+38,y),(x+13,y+37),(x+6,y+58)],fill=color(accent),width=5)
        d.line([(x+38,y),(x+63,y+37),(x+70,y+58)],fill=color(accent),width=5)
        d.arc((x+6,y+22,x+70,y+86),0,180,fill=color(accent),width=5)
    elif kind==1:
        d.line([(x,y+40),(x+20,y+40),(x+30,y+20),(x+43,y+67),(x+57,y+5),(x+70,y+40),(x+92,y+40)],fill=color(accent),width=5)
    else:
        d.rounded_rectangle((x+33,y+20,x+68,y+62),6,outline=color(accent),width=4)
        d.rectangle((x,y+9,x+25,y+74),outline=color(accent),width=3);d.rectangle((x+77,y+9,x+102,y+74),outline=color(accent),width=3)
        d.line((x+25,y+42,x+77,y+42),fill=color(accent),width=3)


def evidence(t):
    im=background().copy();eyebrow(im,(110,65),'NASA EVIDENCE · CLEARLY IDENTIFIED')
    enter_text(im,(110,109),'Real data. Different roles.',72,INK,t,0,True)
    entries=[('Inspiration4','SPACEFLIGHT BIOMARKERS','4 crew members.\nDiscrete blood panels.','Earlier exploratory\nbiomarker prototype.','#bd786c'),('UTMB / CFT70','GROUND ANALOGUES','Bed-rest cardiovascular\nresearch.','Current cardiac and\nvascular forecast heads.','#418f73'),('RadLab · DosTel1','ISS ENVIRONMENT','Station instrument\nradiation measurements.','Environmental exposure,\nnot personal dosimetry.','#6d93b5')]
    for i,(title,tag,body,use,accent) in enumerate(entries):
        p=ease((t-i*.35)/.65);x=110+i*575;y=325+55*(1-p)
        rounded(im,(x,y,x+550,y+570),color('#ffffff'),26,color('#d9e5dc'))
        source_icon(ImageDraw.Draw(im,'RGBA'),i,x+32,y+34,accent)
        text(im,(x+32,y+150),tag,18,accent,True)
        text(im,(x+32,y+198),title,39,INK,True)
        text(im,(x+32,y+279),body,28,MUTED)
        ImageDraw.Draw(im).line((x+32,y+397,x+518,y+397),fill='#e0e9e2',width=2)
        text(im,(x+32,y+433),use,25,INK)
    footer(im,'Continuous glucose and pulse demonstration: Earth-based BIG IDEAs data · These are not one synchronized astronaut.')
    return im


def ending(t):
    im=background(True).copy();logo(im,dark=True,size=35)
    eyebrow(im,(112,235),'THE NEXT MISSION: BETTER DATA',True)
    enter_text(im,(108,292),'More people.',100,WHITE,t,0,True)
    enter_text(im,(108,413),'More often.',100,MINT,t,.55,True)
    enter_text(im,(115,572),'Collect paired astronaut measurements.\nTest. Improve. Validate.',34,'#b3cdbb',t,1.1)
    d=ImageDraw.Draw(im,'RGBA');x0,x1=1030,1790
    text(im,(1030,308),'CURRENT RESEARCH',19,'#b3cdbb',True)
    text(im,(1030,528),'GOAL: DENSER MEASUREMENTS',19,MINT,True)
    for y in [420,640]:d.line((x0,y,x1,y),fill=color(MINT,40),width=2)
    for i in range(5):
        x=x0+i*(x1-x0)/4;d.ellipse((x-7,413,x+7,427),fill=color('#b7cfc0'))
    n=int(31*ease((t-.5)/2))
    for i in range(n):
        x=x0+i*(x1-x0)/30;r=5.5+1.5*math.sin(t*2-i*.2)
        d.ellipse((x-r,640-r,x+r,640+r),fill=color(MINT))
    p=ease((t-2)/.8)
    rounded(im,(112,811,1805,929),color(MINT,round(18*p)),24,color(MINT,round(45*p)))
    text(im,(150,841),'Toward better-tested predictions for future missions.',40,WHITE,True,p)
    footer(im,'Research prototype · Not clinically validated · More data must be paired with careful evaluation.',True)
    return im


def closing(t):
    """An intentional, still closing image while the last sentence resolves."""
    im=background(True).copy();d=ImageDraw.Draw(im,'RGBA')
    for r in [170,230,295]:
        d.ellipse((960-r,330-r,960+r,330+r),outline=color(MINT,30),width=1)
    d.ellipse((890,250,1030,390),outline=color(MINT,220),width=4)
    d.arc((865,290,1055,348),8,185,fill=color(MINT,220),width=4)
    text(im,(960,432),'ORBITA',111,WHITE,True,anchor='ma')
    text(im,(960,593),'Predictive digital twin. Chatbot assistance.',39,'#c7dece',anchor='ma')
    text(im,(960,705),'More astronaut data. More often.',29,MINT,True,anchor='ma')
    text(im,(960,911),'Research prototype · Not clinically validated',21,'#aac4b2',anchor='ma')
    return im


SHOTS=[(0,opening,'Brand / orbital opening'),(1.8,product,'Product overview / fixed framing'),(5.05,comparison,'Observed versus predicted / curve reveal'),(8.1,uncertainty,'Forecast context / chatbot assistance'),(11.25,architecture,'Signals through source encoders'),(14.,latent,'Latent-space macro / rotating network'),(16.,heads,'Separate forecast outputs'),(18.1,evidence,'NASA provenance / staggered cards'),(24.,ending,'More frequent data / closing statement'),(28.25,closing,'Still branded closing image')]


def _render_frame(t):
    index=max(i for i,(start,_,_) in enumerate(SHOTS) if start<=t)
    start,renderer,_=SHOTS[index]
    # Editorial transitions between independent shots, not website navigation.
    if index and t<start+.32:
        p=ease((t-start)/.32)
        prev_start,prev,_=SHOTS[index-1]
        old=prev(t-prev_start)
        new=renderer(t-start)
        if 0<p<1:
            old=old.filter(ImageFilter.GaussianBlur(2.6*math.sin(math.pi*p)))
        return Image.blend(old,new,p)
    return renderer(t-start)


def render_frame(t):
    im=_render_frame(t)
    if not SHOW_SI_LABEL:
        return im
    label='SI-GENERATED VIDEO | SYNTHETIC VOICE'
    width=font(18,True).getlength(label)+28
    x=W-width-40
    rounded(im,(x,1041,W-40,1074),color('#152e26',235),8)
    text(im,(x+14,1045),label,18,'#e8f3e8',True)
    return im


def make_sound_design():
    rate=48000;n=30*rate;rng=np.random.default_rng(822)
    samples=np.zeros(n,dtype=np.float64)
    for at in [1.8,5.05,8.1,11.25,14.,16.,18.1,24.,28.25]:
        length=int(.42*rate)
        noise=rng.normal(0,1,length)
        noise=np.convolve(noise,np.ones(25)/25,mode='same')
        noise*=np.hanning(length)**1.5
        noise/=max(1e-8,np.max(np.abs(noise)))
        start=int((at-.12)*rate);samples[start:start+length]+=noise*.008
    path=OUT/'transition-sound-design.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(np.int16(samples*32767).tobytes())
    mix=OUT/'narration-and-transitions.wav'
    subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-i',str(ROOT/'presentation/voice-editions/ava-30s.wav'),'-i',str(path),'-filter_complex','[0:a][1:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89:level=false[out]','-map','[out]','-ar','48000','-ac','1','-c:a','pcm_s16le',str(mix)],check=True)
    return mix


def main():
    global SHOW_SI_LABEL
    parser=argparse.ArgumentParser();parser.add_argument('--stills',action='store_true');parser.add_argument('--clean',action='store_true');args=parser.parse_args()
    SHOW_SI_LABEL=not args.clean
    OUT.mkdir(parents=True,exist_ok=True)
    sample_times=[.9,3.5,6.8,9.5,12.8,15.2,17.2,21.5,27.5]
    for t in sample_times:
        render_frame(t).save(OUT/f'frame-{t:04.1f}s.png')
    render_frame(29.4).save(OUT/'closing-card.png')
    contact=Image.new('RGB',(1920,1080),'#eef2ec')
    for i,t in enumerate(sample_times):
        frame=Image.open(OUT/f'frame-{t:04.1f}s.png').resize((640,360),Image.Resampling.LANCZOS)
        contact.paste(frame,((i%3)*640,(i//3)*360))
    contact.save(OUT/'contact-sheet.png')
    if args.stills:
        print('Saved montage review frames and contact sheet',flush=True);return
    audio=make_sound_design();video=ROOT/('presentation/orbita-30s-v7-clean.mp4' if args.clean else 'presentation/orbita-30s-v6-submission.mp4')
    command=[FFMPEG,'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio),'-map','0:v','-map','1:a','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-t','30','-movflags','+faststart','-metadata','title=ORBITA | Research film',str(video)]
    log=OUT/'encoder.log'
    begin=time.time()
    with log.open('w',encoding='utf8') as errors:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors)
        try:
            for frame in range(30*FPS):
                image=render_frame(frame/FPS)
                process.stdin.write(image.tobytes())
                if frame%90==0:print(f'Edited film: {frame}/{30*FPS} frames · {time.time()-begin:.1f} s elapsed',flush=True)
            process.stdin.close();code=process.wait()
        except BaseException:
            process.kill();process.wait();raise
    if code:
        raise RuntimeError(log.read_text(encoding='utf8')[-4000:])
    metadata={'file':video.name,'duration_seconds':30,'width':W,'height':H,'fps':FPS,'shots':[{'starts_at':a,'description':c} for a,b,c in SHOTS], 'style':'Offline editorial composition, fixed product framing, blur dissolves, curve reveals, kinetic text, 3D network illustration and staggered evidence cards. No website playback.','forecast_data':'forecast-snapshot.json','forecast_snapshot_sha256':hashlib.sha256((OUT/'forecast-snapshot.json').read_bytes()).hexdigest(),'narration':'voice-editions/ava-30s.wav','sound_design':'Original procedural low-volume transition swells','network_visualization':'Conceptual illustration, not actual patient embeddings','scientific_limits':'Independent studies, visit-based NASA cardiac data, environmental ISS instrument and Earth wearable demonstration remain distinguished.'}
    (OUT/'montage-manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    shutil.copyfile(video,ROOT/'presentation/orbita-30s.mp4')
    manifest_path=ROOT/'presentation/video-manifest.json'
    current=json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {}
    current.update(file='orbita-30s.mp4',versioned_copy=video.name,duration_seconds=30,width=W,height=H,fps=FPS,frames=30*FPS,capture=metadata['style'],audio_mix='montage/narration-and-transitions.wav',montage_manifest='montage/montage-manifest.json')
    manifest_path.write_text(json.dumps(current,indent=2),encoding='utf8')
    print(f'Saved {video}',flush=True)


if __name__=='__main__':main()
