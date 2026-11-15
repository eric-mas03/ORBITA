/* Personal local-agent connection. No provider API keys or shared credentials. */
(()=>{
  'use strict';
  const q=s=>document.querySelector(s);
  const endpoint='http://127.0.0.1:8766';
  const names={codex:'Codex',claude:'Claude Code',opencode:'OpenCode'};
  const logins={codex:'codex login',claude:'claude auth login',opencode:'opencode auth login'};
  let connection=null,busy=false,sequence=0;
  const history=[];
  function error(message){q('#connectionError').textContent=message;q('#connectionError').hidden=!message}
  function refresh(){
    const connected=!!connection;
    q('#connectionStatus').textContent=connected?'YOUR ACCOUNT CONNECTED':'CONNECTION REQUIRED';
    q('#connectionTitle').textContent=connected?names[connection.provider]+' · your local account':'Connect your own assistant';
    q('#connectionDescription').textContent=connected?'Replies use your local agent and your account. Your agent’s usage limits apply.':'Use your personal Codex, Claude Code or OpenCode account. Credentials stay on your computer.';
    q('#openConnection').hidden=connected;q('#disconnectAgent').hidden=!connected;
    q('#assistantConnection').classList.toggle('connected',connected);
    q('#chatInput').disabled=!connected||busy;q('#sendChat').disabled=!connected||busy;
    q('#chatInput').placeholder=connected?'Ask about this forecast…':'Connect your own account to chat…';
    document.querySelectorAll('[data-question]').forEach(b=>b.disabled=!connected||busy);
  }
  function commands(){const p=q('#agentProvider').value;q('#agentCommands').textContent=logins[p]+'\npython scripts/local_agent_bridge.py --provider '+p}
  function disconnect(){sequence++;connection=null;busy=false;history.length=0;q('#agentPairing').value='';q('#connectionSetup').hidden=true;error('');refresh()}
  async function request(path,payload,token){
    const response=await fetch(endpoint+path,{method:'POST',credentials:'omit',cache:'no-store',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify(payload)});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Your local agent could not complete the request.');return data;
  }
  q('#openConnection').onclick=()=>{q('#connectionSetup').hidden=!q('#connectionSetup').hidden;commands()};
  q('#agentProvider').onchange=commands;
  q('#disconnectAgent').onclick=disconnect;
  q('#connectAgent').onclick=async()=>{
    const provider=q('#agentProvider').value,token=q('#agentPairing').value.trim();
    if(!/^[A-Za-z0-9_-]{20,100}$/.test(token)){error('Paste the pairing code printed by your local connector. Do not paste a provider API key.');return}
    q('#connectAgent').disabled=true;error('Connecting to your own local agent…');const id=++sequence;
    try{
      const data=await request('/api/connect',{provider},token);
      if(id!==sequence)return;
      if(data.connected!==true||data.shared_account!==false||data.provider!==provider)throw Error('The connector did not confirm your personal account.');
      connection={provider,token};q('#agentPairing').value='';q('#connectionSetup').hidden=true;error('');refresh();
    }catch(e){if(id===sequence){connection=null;error(e instanceof TypeError?'Start your local connector, keep its terminal open and allow local-network access if your browser asks. Then retry.':e.message);refresh()}}
    finally{q('#connectAgent').disabled=false}
  };
  function append(value,role){const p=document.createElement('p');p.textContent=value;p.dataset.role=role;q('#messages').appendChild(p);q('#messages').scrollIntoView({block:'nearest'})}
  async function send(){
    if(!connection){q('#connectionSetup').hidden=false;commands();error('Connect your own account before sending a chat message.');return}
    if(busy)return;
    const question=q('#chatInput').value.trim();if(!question)return;
    if(question.length>3000){error('Keep your question within 3,000 characters.');return}
    const state=window.ORBITA?.state;if(!state?.variable)return;
    const v=state.variable,panel=state.panel,point=v.future.find(p=>Math.abs(p.x-state.horizon)<.001)||v.future.at(-1);
    const context={measurement:v.label,unit:v.unit,latest:v.current,source:panel.evidence_label,source_url:panel.source_url,clock:panel.clock,horizon:state.horizon,forecast:{q10:point.q10,q50:point.q50,q90:point.q90},shown_history:state.history,history_metrics:state.visibleMetrics,clinical_validation:false,notes:state.data.context};
    const saved=connection,id=++sequence;busy=true;refresh();append(question,'user');q('#chatInput').value='';error('Asking your local agent…');
    try{
      const data=await request('/api/chat',{provider:saved.provider,question,context,history:history.slice(-8)},saved.token);
      if(id!==sequence)return;
      if(typeof data.answer!=='string'||!data.answer.trim())throw Error('Your local agent returned no answer.');
      append(data.answer,'assistant');history.push({role:'user',content:question},{role:'assistant',content:data.answer});error('');
    }catch(e){
      if(id!==sequence)return;
      connection=null;history.length=0;error(e instanceof TypeError?'Your local connector is unavailable. Restart it and connect your own account again.':e.message);q('#connectionSetup').hidden=false;
    }finally{if(id===sequence){busy=false;refresh()}}
  }
  window.ORBITA_CHAT={send,disconnect,get status(){return {connected:!!connection,provider:connection?.provider||null}}};
  commands();refresh();
})();
