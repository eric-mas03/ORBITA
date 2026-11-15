const fs=require('fs'),assert=require('assert');
(async()=>{
 const base=(process.argv[2]||'http://127.0.0.1:8767/').replace(/\/?$/,'/');
 const tabs=await fetch('http://127.0.0.1:9224/json').then(r=>r.json());const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r));
 let id=0;const pending=new Map(),errors=[],agentRequests=[];
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){pending.get(m.id)?.(m);pending.delete(m.id)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description||m.params.exceptionDetails.text);else if(m.method==='Network.requestWillBeSent'&&m.params.request.url.startsWith('http://127.0.0.1:8766'))agentRequests.push(m.params.request.url)});
 function send(method,params={}){return new Promise(resolve=>{pending.set(++id,resolve);ws.send(JSON.stringify({id,method,params}))})}
 async function ev(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.result?.exceptionDetails)throw Error(r.result.exceptionDetails.exception?.description);return r.result?.result?.value}
 async function until(expr){for(let i=0;i<80;i++){if(await ev(expr))return;await new Promise(r=>setTimeout(r,300))}throw Error('Timeout: '+expr)}
 await send('Runtime.enable');await send('Network.enable');await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:base});await until('window.ORBITA?.ready&&window.ORBITA_CHAT');
 assert.equal(await ev('ORBITA.state.data.panels.reduce((n,p)=>n+p.variables.length,0)'),32);assert(await ev("document.querySelector('#chatInput').disabled&&!ORBITA_CHAT.status.connected"));
 await ev("document.querySelector('[data-history=\"1\"]').click();document.querySelector('[data-h=\"24\"]').click();document.querySelector('#browseMeasurements').click()");assert.equal(await ev("document.querySelectorAll('[data-measurement]').length"),32);
 await ev("document.querySelector('[data-measurement=\"echo_3d|LVDV\"]').click()");assert.equal(await ev('ORBITA.state.panel.clock'),'visits');
 await ev("document.querySelector('[data-signal=\"wearable|glucose\"]').click();document.querySelector('#assistantLauncher').click();document.querySelector('#openConnection').click()");assert.equal(await ev("document.querySelectorAll('#agentProvider option').length"),3);assert.equal(agentRequests.length,0);assert.equal(await ev("document.querySelector('#agentPairing').value"),'');
 await ev("document.querySelector('#closeAssistant').click();document.querySelector('#play').click()");await until("location.pathname.endsWith('/demo.html')&&document.querySelector('#film')?.readyState>=1");assert.equal(await ev("document.querySelector('#film').duration"),30);
 for(const name of ['andrew','ryan','guy','ava']){await ev(`document.querySelector('[data-voice="${name}"]').click()`);await until(`document.querySelector('#film').readyState>=1&&document.querySelector('#film').currentSrc.endsWith('ORBITA-demo-${name}.mp4')`);assert.equal(await ev("document.querySelector('#film').duration"),30)}
 await ev("document.querySelector('#film').muted=true;document.querySelector('#film').play()");await until("document.querySelector('#film').currentTime>0.2");await ev("document.querySelector('#film').pause()");
 assert(await ev("fetch('./media/ORBITA-demo-ava.vtt').then(r=>r.text()).then(t=>t.startsWith('WEBVTT'))"));assert.equal(errors.length,0,errors.join('\n'));
 const shot=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync('outputs/publication-demo.png',Buffer.from(shot.result.data,'base64'));
 console.log('PASS: '+base+' · 32 outputs, static replay controls, account required, no automatic agent request, four 30-second videos, playback and subtitles.');ws.close();
})().catch(e=>{console.error(e);process.exit(1)});
