"""Local research dashboard: retrospective predictions and SQLite journal."""
import csv
from contextlib import closing
from functools import lru_cache
import json
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/dashboard/journal.sqlite3'
CSV = ROOT / 'outputs/model_campaign_v2/workspace/final_evaluation/predictions.csv'
HTML = ROOT / 'dashboard/index.html'
CONFIG = ROOT / 'outputs/model_campaign_v2/workspace/config/hourly_task_candidates.json'
LABELS = {'HR':'Frecuencia cardíaca','O2Sat':'Saturación de oxígeno','Temp':'Temperatura','SBP':'Presión sistólica','MAP':'Presión arterial media','DBP':'Presión diastólica','Resp':'Frecuencia respiratoria','EtCO2':'CO₂ espirado','BaseExcess':'Exceso de bases','HCO3':'Bicarbonato','FiO2':'Oxígeno inspirado','pH':'pH sanguíneo','PaCO2':'CO₂ arterial','SaO2':'Saturación arterial','AST':'AST','BUN':'Nitrógeno ureico','Alkalinephos':'Fosfatasa alcalina','Calcium':'Calcio','Chloride':'Cloro','Creatinine':'Creatinina','Bilirubin_direct':'Bilirrubina directa','Glucose':'Glucosa','Lactate':'Lactato','Magnesium':'Magnesio','Phosphate':'Fosfato','Potassium':'Potasio','Bilirubin_total':'Bilirrubina total','TroponinI':'Troponina I','Hct':'Hematocrito','Hgb':'Hemoglobina','PTT':'Tiempo de tromboplastina','WBC':'Leucocitos','Fibrinogen':'Fibrinógeno','Platelets':'Plaquetas'}
GROUPS = {'cardiovascular_vitals':'Corazón y circulación','respiratory_oxygenation':'Respiración y oxígeno','thermoregulation':'Temperatura','acid_base':'Equilibrio ácido-base','renal_energy_metabolism':'Riñón y metabolismo','electrolytes_minerals':'Electrolitos y minerales','hepatobiliary_enzymes':'Hígado y bilirrubina','cardiac_laboratory':'Marcadores cardíacos','metabolic_lactate':'Lactato','cbc_erythrocytes':'Glóbulos rojos','cbc_leukocytes':'Glóbulos blancos','cbc_platelets':'Plaquetas','hemostasis_adhesion':'Coagulación'}
WORKSPACE = ROOT / 'outputs/model_campaign_v2/workspace'
sys.path.insert(0, str(ROOT / 'src'))


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute('CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, kind TEXT NOT NULL, description TEXT NOT NULL, episode_id TEXT NOT NULL)')
    return db


@lru_cache(maxsize=1)
def prediction_index():
    result = {}
    with CSV.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['method'] in ('group_temporal_residual_calibrated', 'group_temporal_residual_raw', 'ensemble_calibrated', 'linear_quantile_calibrated'):
                key = (row['episode_id'], row['variable'], int(row['horizon_hours']), row['method'])
                item = {k: float(row[k]) for k in ('origin_hour','target_hour','observed','q10','q50','q90')}
                item['calibration_source'] = row['calibration_source']
                result.setdefault(key, []).append(item)
    for rows in result.values(): rows.sort(key=lambda x:x['target_hour'])
    return result


def episodes():
    names = {key[0] for key in prediction_index()}
    return sorted(names, key=lambda name:(-len(series(name, 'HR', 24)), -len(series(name, 'HR', 1)), name))


def series(episode, variable, horizon=1, method='group_temporal_residual_calibrated'):
    return prediction_index().get((episode, variable, horizon, method), [])


@lru_cache(maxsize=1)
def nasa_data():
    import math
    references = {}
    for assay,path in [('CMP',ROOT/'data/raw/OSD-575/LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv'),('CBC',ROOT/'data/raw/OSD-569/LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv')]:
        with path.open(encoding='utf-8-sig',newline='') as source:
            for r in csv.DictReader(source):
                try:
                    low,high = float(r['RANGE_MIN']),float(r['RANGE_MAX'])
                    if math.isfinite(low) and math.isfinite(high) and low<=high:
                        references[(assay,r['SUBJECT_ID'],r.get('timepoint',r.get('TEST_DATE')),r['ANALYTE'],r['UNITS'])]=(low,high)
                except (ValueError,KeyError): pass
    with (ROOT/'data/processed/inspiration4_observations.csv').open(encoding='utf8',newline='') as f:
        observations = [r for r in csv.DictReader(f) if r['value'] and math.isfinite(float(r['value']))]
    for row in observations:
        bounds = references.get((row['assay'],row['subject_id'],row['timepoint'],row['original_analyte'] or row['analyte'],row['unit']))
        row['reference_min'],row['reference_max'] = bounds if bounds else (None,None)
        row['reference_status'] = ('below_reference' if float(row['value'])<bounds[0] else 'above_reference' if float(row['value'])>bounds[1] else 'within_reference') if bounds else 'no_reference'
    with (ROOT/'outputs/training/predictions.csv').open(encoding='utf8',newline='') as f:
        predictions = [r for r in csv.DictReader(f) if r['method']=='grouped_transformer']
    variables = {}
    translated = {'GLUCOSE':'Glucosa','CREATININE':'Creatinina','POTASSIUM':'Potasio','SODIUM':'Sodio','CALCIUM':'Calcio','CHLORIDE':'Cloro','HEMOGLOBIN':'Hemoglobina','HEMATOCRIT':'Hematocrito','PLATELET COUNT':'Plaquetas','WHITE BLOOD CELL COUNT':'Leucocitos','RED BLOOD CELL COUNT':'Eritrocitos','ALBUMIN':'Albúmina','PROTEIN; TOTAL':'Proteínas totales','BILIRUBIN; TOTAL':'Bilirrubina total','ALKALINE PHOSPHATASE':'Fosfatasa alcalina','CARBON DIOXIDE':'Dióxido de carbono','UREA NITROGEN (BUN)':'Nitrógeno ureico'}
    for row in observations:
        variables[row['variable']] = {'id':row['variable'],'label':translated.get(row['analyte'],row['analyte']),'unit':row['unit'],'assay':row['assay']}
    return observations,predictions,sorted(variables.values(),key=lambda r:(r['assay'] not in ('CMP','CBC'),r['assay'],r['label']))


@lru_cache(maxsize=1)
def final_data():
    import numpy as np
    with np.load(WORKSPACE / 'role_data/final/packed.npz', allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


@lru_cache(maxsize=1)
def model_artifacts():
    import numpy as np
    import torch
    from astronaut_twin.hourly_architecture_trials import load_trial_checkpoint
    def read(path):
        with np.load(path, allow_pickle=False) as f: return {k:f[k] for k in f.files}
    torch.set_num_threads(2)
    base = WORKSPACE / 'outputs/hourly_baselines'
    model, state = load_trial_checkpoint(WORKSPACE / 'outputs/hourly_architecture_trials/group_temporal_residual.pt')
    return model, state, read(base/'scalers.npz'), read(base/'linear_quantile.npz'), read(base/'persistence.npz'), read(WORKSPACE/'calibration/group_temporal_residual.npz')


def origins_for(episode):
    import numpy as np
    data = final_data()
    matches = np.flatnonzero(data['episode_ids'] == episode)
    if len(matches) != 1: return []
    samples = np.flatnonzero(data['sample_episode'] == matches[0])
    return [int(x) for x in data['times'][data['origins'][samples]]]


@lru_cache(maxsize=64)
def forecast_prediction(episode, origin, method):
    import numpy as np
    import torch
    from astronaut_twin.hourly_architecture_trials import prepare_frozen_trial_data
    from astronaut_twin.hourly_baselines import predict_linear
    if method not in ('group_temporal_residual_calibrated','group_temporal_residual_raw'):
        raise ValueError('La vista por instante usa el Transformer por grupo')
    data = final_data()
    ep = np.flatnonzero(data['episode_ids'] == episode)
    if len(ep) != 1: raise ValueError('Caso desconocido')
    sample = np.flatnonzero((data['sample_episode'] == ep[0]) & (data['times'][data['origins']] == origin))
    if len(sample) != 1: raise ValueError('Instante no disponible')
    sample = int(sample[0])
    packed = dict(data)
    for name in ('origins','history_start','sample_episode'): packed[name] = data[name][[sample]]
    # Future labels never reach preparation or inference.
    packed['target_rows'] = np.full((1,4),-1,dtype=np.int64)
    model, state, scalers, linear, bands, calibration = model_artifacts()
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    raw = predict_linear(baseline, linear, bands)
    base_z = torch.from_numpy((raw-neural.mean[None,None,:,None])/neural.scale[None,None,:,None])
    args, _, _ = neural.batch([0], state['groups'])
    with torch.no_grad():
        prediction = model(args, torch.from_numpy(baseline.features), base_z).numpy()[0]
    prediction = prediction * neural.scale[None,:,None] + neural.mean[None,:,None]
    if method.endswith('_calibrated'):
        prediction[:,:,0] -= calibration['expansion']; prediction[:,:,2] += calibration['expansion']
    return prediction


def forecast_window(episode, variable, origin, method):
    import numpy as np
    data = final_data()
    ep = np.flatnonzero(data['episode_ids'] == episode)
    col = np.flatnonzero(data['variables'] == variable)
    if len(ep) != 1 or len(col) != 1: raise ValueError('Caso o variable desconocidos')
    v = int(col[0]); first = int(data['episode_offsets'][ep[0]]); end = int(data['episode_offsets'][ep[0]+1])
    prediction = forecast_prediction(episode, origin, method)
    calibration = model_artifacts()[-1]
    forecasts = [{'horizon':int(h),'hour':int(origin+h),'q10':float(prediction[i,v,0]),'q50':float(prediction[i,v,1]),'q90':float(prediction[i,v,2]),
                  'calibration_source':str(calibration['source'][i,v]) if method.endswith('_calibrated') else 'raw'} for i,h in enumerate(data['horizons'])]
    clock = data['times'][first:end]; values = data['values'][first:end,v]
    history = [{'hour':int(t),'value':float(x)} for t,x in zip(clock,values) if origin-23<=t<=origin and np.isfinite(x)]
    future = [{'hour':int(t),'value':float(x)} for t,x in zip(clock,values) if origin<t<=origin+24 and np.isfinite(x)]
    meta = next(x for x in json.loads(CONFIG.read_text(encoding='utf8'))['variables'] if x['source_key']==variable)
    return {'origin':origin,'history':history,'future_observed':future,'forecast':forecasts,'unit':meta['source_unit_as_documented'],
            'unit_review':meta.get('unit_review_required',False),'context_hours':24,'method':method}


def overview_at(episode, origin, horizon, method):
    """Known measurements and forecasts share one issuance time; never use later actuals."""
    import numpy as np
    if horizon not in (1,4,8,24): raise ValueError('Horizonte inválido')
    if method not in ('group_temporal_residual_calibrated','group_temporal_residual_raw','ensemble_calibrated','linear_quantile_calibrated'):
        raise ValueError('Modelo desconocido')
    if origin not in origins_for(episode): raise ValueError('Instante no disponible')
    data=final_data()
    ep=int(np.flatnonzero(data['episode_ids']==episode)[0])
    first,end=map(int,data['episode_offsets'][ep:ep+2])
    clock=data['times'][first:end]
    prediction=forecast_prediction(episode,origin,method) if method.startswith('group_temporal') else None
    h=int(np.flatnonzero(data['horizons']==horizon)[0])
    result=[]
    for key in ('HR','O2Sat','Resp','Temp','Glucose','SBP','Creatinine','Hgb'):
        col=int(np.flatnonzero(data['variables']==key)[0])
        values=data['values'][first:end,col]
        known=np.flatnonzero((clock<=origin)&np.isfinite(values))
        saved=next((r for r in series(episode,key,horizon,method) if r['origin_hour']==origin),None) if prediction is None else None
        quantiles=prediction[h,col] if prediction is not None else [saved['q10'],saved['q50'],saved['q90']] if saved else None
        result.append({'variable':key,'value':float(values[known[-1]]) if len(known) else None,
                       'hour':int(clock[known[-1]]) if len(known) else None,'origin':origin,'horizon':horizon,'target_hour':origin+horizon,
                       'forecast':dict(zip(('q10','q50','q90'),map(float,quantiles))) if quantiles is not None else None})
    return result


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, body, mime='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/': return self.respond(200, (ROOT/'dashboard/presentation.html').read_bytes(), 'text/html; charset=utf-8')
        if u.path == '/research': return self.respond(200, HTML.read_bytes(), 'text/html; charset=utf-8')
        if u.path in ('/presentation.js','/presentation.css','/assistant-connection.js'):
            return self.respond(200,(ROOT/'dashboard'/u.path[1:]).read_bytes(),'text/javascript; charset=utf-8' if u.path.endswith('.js') else 'text/css; charset=utf-8')
        if u.path in ('/presentation/architecture.svg','/presentation/architecture.drawio'):
            return self.respond(200,(ROOT/'presentation'/u.path.rsplit('/',1)[1]).read_bytes(),'image/svg+xml' if u.path.endswith('.svg') else 'application/xml')
        if u.path == '/api/presentation':
            from astronaut_twin.presentation_forecasts import presentation_payload
            return self.respond(200,presentation_payload())
        if u.path in ('/expanded.js','/expanded.css'):
            return self.respond(200,(ROOT/'dashboard'/u.path[1:]).read_bytes(),'text/javascript; charset=utf-8' if u.path.endswith('.js') else 'text/css; charset=utf-8')
        if u.path.startswith('/api/expanded/'):
            from astronaut_twin import source_dashboard as expanded
            q=parse_qs(u.query)
            try:
                if u.path=='/api/expanded/catalog':return self.respond(200,expanded.catalog())
                if u.path=='/api/expanded/origins':return self.respond(200,expanded.origins(q.get('case',[''])[0]))
                if u.path=='/api/expanded/forecast':return self.respond(200,expanded.forecast(q.get('case',[''])[0],int(q.get('origin',[''])[0]),q.get('method',['ensemble'])[0],int(q.get('horizon',['1'])[0])))
                return self.respond(404,{'error':'Recurso desconocido'})
            except (ValueError,IndexError) as exc:return self.respond(400,{'error':str(exc)})
            except FileNotFoundError:return self.respond(503,{'error':'La campaña ampliada aún no está preparada'})
        if u.path == '/api/nasa':
            obs,pred,variables = nasa_data()
            q = parse_qs(u.query);subject=q.get('subject',['C001'])[0];var=q.get('variable',['CMP::GLUCOSE'])[0]
            return self.respond(200,{'subjects':sorted({r['subject_id'] for r in obs}), 'variables':variables,
                'visits':['L-92','L-44','L-3','R+1','R+45','R+82','R+194'],
                'observations':[{'visit':r['timepoint'],'value':float(r['value']),'unit':r['unit'],'source_file':r['source_file'],'reference_min':r['reference_min'],'reference_max':r['reference_max'],'reference_status':r['reference_status']} for r in sorted(obs,key=lambda x:int(x['visit_index'])) if r['subject_id']==subject and r['variable']==var],
                'predictions':[{'origin':r['origin'],'target':r['target_visit'],'horizon_days':float(r['horizon_days']),'split':r['split'],**{k:float(r[k]) for k in ('q10','q50','q90')}} for r in pred if r['subject_id']==subject and r['variable']==var],
                'source':'NASA OSD-575 / OSD-569 · Inspiration4', 'model_status':'Modelo espacial exploratorio, cuatro participantes; no validación clínica ni telemetría horaria'})
        if u.path == '/api/episodes': return self.respond(200, episodes())
        if u.path == '/api/cases':
            cases = []
            for i, episode in enumerate(episodes()):
                rows = series(episode, 'HR', 1)
                hours = int(rows[-1]['target_hour']) if rows else 0
                cases.append({'id': episode, 'label': f'Caso {i+1:03d}', 'hours': hours, 'points': len(rows)})
            return self.respond(200, cases)
        if u.path == '/api/origins':
            return self.respond(200, origins_for(parse_qs(u.query).get('episode',[''])[0]))
        if u.path == '/api/forecast':
            q = parse_qs(u.query)
            try:
                return self.respond(200, forecast_window(q.get('episode',[''])[0], q.get('variable',['HR'])[0], int(q.get('origin',[''])[0]), q.get('method',['group_temporal_residual_calibrated'])[0]))
            except (ValueError, IndexError) as exc: return self.respond(400, {'error':str(exc)})
        if u.path == '/api/variables':
            config = json.loads(CONFIG.read_text(encoding='utf-8'))
            return self.respond(200, [{'id': v['source_key'], 'label': LABELS.get(v['source_key'],v['source_key']), 'unit': v['source_unit_as_documented'], 'group': v['group'], 'group_label': GROUPS.get(v['group'],v['group']), 'unit_review': v.get('unit_review_required', False)} for v in config['variables']])
        if u.path == '/api/overview':
            q=parse_qs(u.query)
            try:
                episode=q.get('episode',[''])[0]
                origin=int(q.get('origin',[str(origins_for(episode)[-1])])[0])
                return self.respond(200,overview_at(episode,origin,int(q.get('horizon',['1'])[0]),q.get('method',['group_temporal_residual_calibrated'])[0]))
            except (ValueError,IndexError) as exc: return self.respond(400,{'error':str(exc)})
        if u.path == '/api/series':
            q = parse_qs(u.query)
            try:
                horizon = int(q.get('horizon',['1'])[0])
                if horizon not in (1,4,8,24): raise ValueError()
                return self.respond(200, series(q.get('episode',[''])[0], q.get('variable',['HR'])[0], horizon, q.get('method',['group_temporal_residual_calibrated'])[0]))
            except ValueError: return self.respond(400, {'error':'Horizonte inválido'})
        if u.path == '/api/entries':
            with closing(connect()) as db:
                episode = parse_qs(u.query).get('episode',[''])[0]
                rows = db.execute('SELECT id,created_at,kind,description FROM entries WHERE episode_id = ? ORDER BY id DESC LIMIT 100',(episode,)).fetchall()
            return self.respond(200, [dict(zip(('id','created_at','kind','description'),r)) for r in rows])
        return self.respond(404, {'error':'No encontrado'})

    def do_POST(self):
        if self.path != '/api/entries': return self.respond(404, {'error':'No encontrado'})
        try:
            origin = self.headers.get('Origin')
            if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'): return self.respond(403, {'error':'Origen no permitido'})
            if not self.headers.get('Content-Type','').lower().startswith('application/json'): return self.respond(400, {'error':'Se requiere JSON'})
            n = int(self.headers.get('Content-Length','0'))
            if n < 1 or n > 4096: raise ValueError('Tamaño incorrecto')
            body = json.loads(self.rfile.read(n))
            kind, description, episode = body['kind'], body['description'].strip(), body['episode']
            valid_nasa = episode in ('NASA:C001','NASA:C002','NASA:C003','NASA:C004')
            if kind not in ('medición','comida','actividad','nota') or not description or len(description)>1000 or (not valid_nasa and episode not in episodes()): raise ValueError('Datos incorrectos')
            with closing(connect()) as db:
                db.execute('INSERT INTO entries(kind,description,episode_id) VALUES (?,?,?)',(kind,description,episode))
                db.commit()
            return self.respond(201, {'ok':True})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError): return self.respond(400, {'error':'Datos incorrectos'})


if __name__ == '__main__':
    connect().close()
    print('Panel local: http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765), Handler).serve_forever()
