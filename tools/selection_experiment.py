"""Read-only E/F feasibility experiment; not a shipped ranking or a musical quality test.

Run from the repository: .venv/bin/python tools/selection_experiment.py
Only writes the evidence and blinded listening manifests under .planning/research.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.planning/research'
SEED = 20261026
K = 5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(s, duration):
    # Existing descriptors only; each categorical token has equal importance.
    return set([f'class:{s["class"]}', f'time:{min(3, int(4 * s["source_start_s"] / max(duration, 1)))}']
               + [f'character:{v}' for v in s.get('character') or []])


def diverse(rows, duration):
    remaining = list(rows)
    selected, seen = [], set()
    while remaining and len(selected) < K:
        # Deterministic novelty, then source-time separation; ID breaks remaining ties.
        def score(s):
            gap = min((abs(s['source_start_s'] - t['source_start_s']) for t in selected), default=0)
            return (-len(features(s, duration) - seen), -gap, s['id'])
        best = min(remaining, key=score)
        selected.append(best)
        seen |= features(best, duration)
        remaining.remove(best)
    return selected


def stats(rows, duration):
    return {'ids': [s['id'] for s in rows], 'classes': len({s['class'] for s in rows}),
            'source_quartiles': len({min(3, int(4*s['source_start_s']/max(duration, 1))) for s in rows}),
            'character_tokens': sorted({x for s in rows for x in s.get('character') or []})}


def tail_ratio(path):
    # Lower energy near the tail may indicate less abrupt truncation, but also silence/weak content.
    # Deliberately NOT called cleanliness or a quality score.
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    tail = y[-max(1, int(sr*.02)):]
    return round(float(np.sqrt(np.mean(tail**2)) / max(float(np.sqrt(np.mean(y**2))), 1e-10)), 6)


def main():
    rng = random.Random(SEED)
    base = json.loads((OUT/'2026-10-26-agent-retrieval-baseline.json').read_text())
    report = {'seed': SEED, 'k': K, 'method': 'Read-only technical feasibility; no listening judgments.',
              'model_inference': False, 'indexes': [], 'shortlist_tasks': [], 'exemplar_pairs': [],
              'similarity_tasks': [], 'cli_checks': []}
    trials, key, protected = [], {}, {}

    def protect(path):
        path = path.resolve()
        protected.setdefault(str(path), digest(path))

    def trial(task, methods, song, anchor=None):
        tid = f'T{len(trials)+1:03}'
        items = list(methods.items())
        rng.shuffle(items)
        variants, mapping = {}, {}
        for label, (method, rows) in zip('ABCD', items):
            variants[label] = []
            for s in rows:
                path = song/s['path']
                protect(path)
                variants[label].append({'id': s['id'], 'path': str(path), 'sha256': digest(path)})
            mapping[label] = method
        trials.append({'trial': tid, 'task': task, 'anchor': anchor, 'variants': variants,
                       'judgments': {'preferred': None, 'cut_cleanliness': None, 'musical_usefulness': None,
                                     'variety': None, 'auditions_until_selection': None, 'seconds_until_selection': None,
                                     'missed_useful_alternatives': None, 'notes': None}})
        key[tid] = mapping
        return tid

    for item in base['indexes']:
        song = Path(item['song_dir']); ip = song/'index.json'
        protect(ip); d = json.loads(ip.read_text()); ss = d['samples']; duration = d['source']['duration_s']
        ep = song/'embeddings/clap.npy'; mp = song/'embeddings/clap.json'
        protect(ep); protect(mp); m = json.loads(mp.read_text()); emb = np.load(ep, allow_pickle=False)
        valid = (emb.shape == (len(ss), 512) and np.isfinite(emb).all()
                 and len(m['rows']) == len(ss) and m['paths'] == [s['path'] for s in ss]
                 and all(s.get('embedding_ref') == {'file':'embeddings/clap.npy','row':i} for i,s in enumerate(ss)))
        norms = np.linalg.norm(emb, axis=1)
        valid = bool(valid and np.all(norms > 0))
        report['indexes'].append({'song_root':str(song),'index_revision':digest(ip),'samples':len(ss),
             'embedding_mapping_valid':valid,'embedding_sha256':digest(ep),'metadata_sha256':digest(mp),
             'encoder': {k:m[k] for k in ('model','revision','dim','sample_rate','max_seconds')},
             'min_norm':float(norms.min()),'max_norm':float(norms.max()),
             'longer_than_encoder_window':sum(s['duration_s']>m['max_seconds'] for s in ss)})
        assert valid, song
        from grepchopflip.enrich.wavchunks import audio_digest
        audio_matches = all(audio_digest(song/s['path']) == m['rows'][i] for i,s in enumerate(ss))
        report['indexes'][-1]['all_embedding_audio_digests_match'] = audio_matches
        assert audio_matches, song
        if d['separation']['split'] != 'all':
            continue
        by_id = {s['id']:i for i,s in enumerate(ss)}
        for stem, kind in [('drums','oneshot'),('bass','oneshot'),('other','phrase'),('vocals','phrase')]:
            pool = sorted((s for s in ss if s['stem']==stem and s['kind']==kind and s['needs_review'] is False),key=lambda s:s['id'])
            reps = [s for s in pool if s['is_dedup_representative'] is not False]
            if not reps:
                continue
            label = f'{song.name}/{stem}/{kind}'
            baseline = reps[:K]; varied = diverse(reps,duration)
            # A source-time condition available today through --start/--end.
            region = [s for s in reps if s['source_end_s'] > duration/2 and s['source_start_s'] < 3*duration/4][:K]
            for condition, expected, extra in [('current',baseline,[]), ('middle_late_region',region,['--start',str(duration/2),'--end',str(3*duration/4)])]:
                cmd=[str(Path(sys.executable).parent/'gcf'),'samples',str(song),'--stem',stem,'--kind',kind,'--limit',str(K),'--json']+extra
                started=time.perf_counter(); run=subprocess.run(cmd,capture_output=True,check=True)
                result=json.loads(run.stdout); ids=[s['id'] for s in result['samples']]
                assert ids == [s['id'] for s in expected], (label,condition,ids)
                report['cli_checks'].append({'task':label,'condition':condition,'command':cmd,'exit_code':run.returncode,
                    'stdout_bytes':len(run.stdout),'wall_ms':round((time.perf_counter()-started)*1000,3),'ids':ids})
            tid=trial(label+' shortlist',{'current':baseline,'metadata_diverse':varied,'middle_late_region':region},song)
            report['shortlist_tasks'].append({'trial':tid,'task':label,'pool':len(pool),'representatives':len(reps),
                'current':stats(baseline,duration),'metadata_diverse':stats(varied,duration),'middle_late_region':stats(region,duration),
                'changed_ids':len(set(s['id'] for s in varied)-set(s['id'] for s in baseline))})
            groups=defaultdict(list)
            for s in pool:
                groups[s.get('dedup_group') or s['id']].append(s)
            tested=0
            for gid,members in sorted(groups.items()):
                if len(members)<2 or tested>=2:
                    continue
                rep=next((s for s in members if s['is_dedup_representative']),None)
                if rep is None: continue
                scores={s['id']:tail_ratio(song/s['path']) for s in members}
                alt=min(members,key=lambda s:(scores[s['id']],s['id']))
                tested+=1
                # Record ties/no change; avoid fake A/B comparisons of identical audio.
                pair={'task':label,'group':gid,'members':len(members),'current':rep['id'],
                      'low_tail_proxy':alt['id'],'current_tail_ratio':scores[rep['id']],
                      'alternate_tail_ratio':scores[alt['id']],'changed':alt['id']!=rep['id']}
                if pair['changed']:
                    pair['trial']=trial(label+' fixed-group exemplar',{'current': [rep],'low_tail_proxy':[alt]},song)
                report['exemplar_pairs'].append(pair)
            # Three deterministic anchors per task, avoiding reliance on the first ID only.
            for ai in sorted(set([0,len(reps)//2,len(reps)-1])):
                anchor=reps[ai]
                candidates=[s for s in reps if s['id']!=anchor['id'] and s.get('dedup_group')!=anchor.get('dedup_group')]
                t=time.perf_counter()
                vec=emb[by_id[anchor['id']]]/norms[by_id[anchor['id']]]
                sim={s['id']:float(np.dot(vec,emb[by_id[s['id']]]/norms[by_id[s['id']]])) for s in candidates}
                ranked=sorted(candidates,key=lambda s:(-sim[s['id']],s['id']))[:K]
                elapsed=(time.perf_counter()-t)*1000
                exact=[s for s in candidates if s['class']==anchor['class'] and
                       set(anchor.get('character') or []) <= set(s.get('character') or [])][:K]
                st={'task':label,'anchor':anchor['id'],'candidate_count':len(candidates),'warm_score_sort_ms':round(elapsed,3),
                    'cosine': [{'id':s['id'],'score':round(sim[s['id']],6),'class':s['class']} for s in ranked],
                    'metadata_ids':[s['id'] for s in exact],
                    'overlap':len({s['id'] for s in ranked}&{s['id'] for s in exact}),
                    'cross_class':sum(s['class']!=anchor['class'] for s in ranked)}
                if ai==len(reps)//2:
                    ap=song/anchor['path'];protect(ap)
                    st['trial']=trial(label+' alternatives',{'metadata':exact,'cosine':ranked},song,
                                      {'id':anchor['id'],'path':str(ap),'sha256':digest(ap)})
                report['similarity_tasks'].append(st)
    changed=[p for p,sha in protected.items() if digest(Path(p)) != sha]
    assert not changed,changed
    report['integrity']={'protected_files':len(protected),'changed':changed,
                         'scope':'All seven indexes/embedding files and every listening candidate/anchor WAV; other WAVs not hashed.'}
    report['human_judgments']={'collected':0,'preferred_method':None,'time_to_accepted_selection':None}
    (OUT/'2026-10-26-selection-experiments.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'2026-10-26-listening-trials.json').write_text(json.dumps({'seed':SEED,'trials':trials},indent=2)+'\n')
    (OUT/'2026-10-26-listening-key.json').write_text(json.dumps(key,indent=2)+'\n')
    print(json.dumps({'indexes':len(report['indexes']),'shortlist_tasks':len(report['shortlist_tasks']),
         'exemplar_pairs':len(report['exemplar_pairs']),'changed_exemplars':sum(x['changed'] for x in report['exemplar_pairs']),
         'similarity_tasks':len(report['similarity_tasks']),'listening_trials':len(trials),'integrity':report['integrity']}))

if __name__=='__main__':main()
