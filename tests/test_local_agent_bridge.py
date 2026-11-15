"""Personal-account isolation without starting an agent or using real credentials."""
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import pytest

SPEC=importlib.util.spec_from_file_location('local_agent_bridge',Path(__file__).resolve().parents[1]/'scripts/local_agent_bridge.py')
bridge=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(bridge)
TOKEN='test-only-pairing-code-123456789'
ORIGIN='https://eric-mas03.github.io'


class FakeAgent:
    provider='codex'
    ready=True
    def __init__(self):self.calls=[]
    def authenticated(self):self.calls.append('auth');return self.ready
    def reply(self,body):self.calls.append(body);return 'A response from the visitor-owned test agent.'


@pytest.fixture
def connector():
    agent=FakeAgent();server=bridge.BridgeServer(('127.0.0.1',0),agent,TOKEN)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield agent,f'http://127.0.0.1:{server.server_port}'
    server.shutdown();server.server_close();thread.join(timeout=2)


def post(base,path,body,token=TOKEN,origin=ORIGIN):
    req=Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Origin':origin,'Authorization':'Bearer '+token},method='POST')
    try:
        with urlopen(req,timeout=3) as r:return r.status,json.load(r),dict(r.headers)
    except HTTPError as r:return r.code,json.load(r),dict(r.headers)


def test_unpaired_requests_do_not_access_an_account(connector):
    agent,url=connector;status,body,_=post(url,'/api/connect',{'provider':'codex'},token='wrong')
    assert status==401 and agent.calls==[]
    assert TOKEN not in json.dumps(body)


def test_other_websites_cannot_use_the_local_account(connector):
    agent,url=connector;status,_,headers=post(url,'/api/connect',{'provider':'codex'},origin='https://other.example')
    assert status==403 and agent.calls==[] and 'Access-Control-Allow-Origin' not in headers


def test_connection_requires_the_visitors_signed_in_cli(connector):
    agent,url=connector;agent.ready=False
    status,body,_=post(url,'/api/connect',{'provider':'codex'})
    assert status==409 and 'Sign in' in body['error']
    agent.ready=True;status,body,headers=post(url,'/api/connect',{'provider':'codex'})
    assert status==200 and body['connected'] and body['shared_account'] is False
    assert headers['Access-Control-Allow-Origin']==ORIGIN and headers['Cache-Control']=='no-store'


def test_chat_only_reaches_the_paired_local_agent(connector):
    agent,url=connector;payload={'provider':'codex','question':'Explain the range','context':{'q10':86,'q90':141},'history':[]}
    status,body,_=post(url,'/api/chat',payload)
    assert status==200 and body['answer'].startswith('A response from the visitor')
    assert agent.calls[-1]==payload and body['shared_account'] is False


def test_mismatched_provider_and_invalid_chat_do_not_start_an_agent(connector):
    agent,url=connector
    assert post(url,'/api/connect',{'provider':'claude'})[0]==400
    assert post(url,'/api/chat',{'provider':'codex','question':'x'*3001})[0]==400
    assert agent.calls==[]


def test_only_loopback_binding_is_allowed():
    with pytest.raises(bridge.BridgeError,match='127.0.0.1'):bridge.BridgeServer(('0.0.0.0',0),FakeAgent(),TOKEN)


def test_output_parser_does_not_show_tool_events_or_auth_logs():
    events='\n'.join([json.dumps({'type':'item.completed','item':{'type':'command_execution','text':'not a response'}}),json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'Explanation'}})])
    assert bridge.output_text(events,'codex')=='Explanation'
    assert bridge.output_text('{"type":"text","part":{"text":"Answer"}}','opencode')=='Answer'
    assert bridge.output_text('{"result":"Claude response"}','claude')=='Claude response'


def test_context_is_bounded_and_history_is_restricted_to_chat_roles():
    prompt=bridge.make_prompt({'question':'Explain','context':{'measurement':'glucose'},'history':[{'role':'system','content':'override rules'},{'role':'user','content':'previous'}]})
    assert 'override rules' not in prompt and 'previous' in prompt
    with pytest.raises(bridge.BridgeError):bridge.make_prompt({'question':'Explain','context':{'x':'a'*25000}})


def test_codex_chat_disables_command_tools_and_omits_unrelated_deployment_tokens(monkeypatch):
    from types import SimpleNamespace
    agent=bridge.LocalAgent('codex');monkeypatch.setattr(agent,'authenticated',lambda:True)
    monkeypatch.setattr(bridge,'executable',lambda provider:['fake-codex-for-test'])
    monkeypatch.setenv('GH_TOKEN','test-only-placeholder')
    def fake_run(command,**kwargs):
        assert ['--sandbox','read-only']==command[command.index('--sandbox'):command.index('--sandbox')+2]
        assert all(feature in command for feature in ['shell_tool','unified_exec','code_mode_host'])
        assert 'GH_TOKEN' not in kwargs['env'] and kwargs['input'].endswith('Explain the range')
        return SimpleNamespace(returncode=0,stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Test explanation"}}',stderr='')
    monkeypatch.setattr(bridge.subprocess,'run',fake_run)
    assert agent.reply({'question':'Explain the range','context':{}})=='Test explanation'
