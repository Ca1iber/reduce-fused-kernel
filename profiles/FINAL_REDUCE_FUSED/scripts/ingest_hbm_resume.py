from pathlib import Path
import json,csv,re,hashlib,shutil
v=Path(__file__).resolve().parents[1];repo=v.parents[1]
usage=v/'raw/hbm_native/usage_16g.csv'
with usage.open()as f:rows=list(csv.DictReader(f));fields=list(rows[0])
records={(x['version'],x['variant'],x['workload']):x for x in rows}
with (v/'raw/summary_16g.csv').open()as f:
    references={(x['variant'].lower(),x['workload']):x for x in csv.DictReader(f)}
latest=v/'raw/benchmark_rerun_20261009T105946/comparison_16g.csv'
if latest.exists():
    with latest.open()as f:
        current={(x['variant'].lower(),x['workload']):float(x['current_us'])for x in csv.DictReader(f)}
else:current={}
allowed=json.loads((v/'meta/hbm_native/verified_batches.json').read_text())['batches']
pending=json.loads((v/'meta/hbm_native/current_remaining.json').read_text())['batch']
if pending not in allowed:allowed.append(pending)
accepted=[];rejected=[]
for batch in allowed:
    folder=v/'raw/hbm_native'/batch/'mcprofiler_output'
    status_file=v/'meta/hbm_native'/(batch+'_status.json')
    status=json.loads(status_file.read_text())if status_file.exists()else{}
    command=status.get('command',[])
    plan_path=None
    for part in command:
        if '--plan' in str(part):
            import shlex
            words=shlex.split(part)
            if '--plan' in words:plan_path=Path(words[words.index('--plan')+1])
    plan=json.loads(plan_path.read_text())if plan_path and plan_path.exists()else[]
    for f in sorted(folder.glob('*_kernel.txt.json')):
        match=re.fullmatch(r'\d+_hbm_(v000|final)_(base|xsf|fp8|quantized)_(tiny|h3072|h7168|prefill)_kernel\.txt\.json',f.name)
        if match:version,variant,workload=match.groups()
        elif len(plan)==1:version,variant,workload=[plan[0][x]for x in ['version','variant','workload']]
        else:continue
        name=f'hbm_{version}_{variant}_{workload}'
        case=json.loads((v/'raw/hbm_native'/name/'case.json').read_text())
        d=json.loads(f.read_text());e=next(x for x in d['Summary']if x['name']=='RoofLine')['value'];p=e['processed_data'];roof=e['data']
        cta=int(float(str(next(x for x in d['CE Statistics']if x['name']=='WORKGROUPS')['value']).replace(',','')))
        expected_cta=case['T']*(case['H']//case['config'].get('tile_hidden',case['H']))
        q=int(references[variant,workload]['modeled_bytes']);ratio=p['all_memacs']/q
        measured_us=p['during']/(roof['core_clk']*1000)
        reference_us=current.get((variant,workload),float(references[variant,workload]['current_us']))if version=='final'else float(references[variant,workload]['baseline_us'])
        consistent=(cta==expected_cta and 0.8<=ratio<=1.3 and 0<measured_us<max(10,reference_us*3))
        check=dict(batch=batch,case=name,CTA=cta,expected_CTA=expected_cta,byte_ratio=ratio,native_us=measured_us,reference_us=reference_us)
        if not consistent:rejected.append(check);continue
        picture=Path(e['filename']);image=v/'analysis/hbm_native'/('verified_'+name+'.png');shutil.copy2(picture,image)
        row=dict(version=version,variant=variant,workload=workload,hbm_usage_pct=100*roof['case_bandwith']/roof['MAX_Bandwith'],native_bandwidth_GBs=roof['case_bandwith'],native_roof_GBs=roof['MAX_Bandwith'],native_cycles=p['during'],native_core_clk_GHz=roof['core_clk'],native_bytes=p['all_memacs'],source_record=str(f.relative_to(repo)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),image=str(image.relative_to(repo)),reused=False)
        records[version,variant,workload]=row;accepted.append(check)
ordered=sorted(records.values(),key=lambda x:(x['version'],['base','xsf','fp8','quantized'].index(x['variant']),['tiny','h3072','h7168','prefill'].index(x['workload'])))
with usage.open('w',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(ordered)
status=dict(total=len(ordered),expected=32,reused=8,new=len(ordered)-8,complete=len(ordered)==32,accepted=accepted,rejected=rejected,whitelisted_batches=allowed)
(v/'meta/hbm_native/ingestion_status.json').write_text(json.dumps(status,indent=2)+'\n')
print('VALID_RECORDS',len(ordered),'REJECTED_THIS_PASS',len(rejected),flush=True)
for x in ordered:print(x['version'],x['variant'],x['workload'],f"{float(x['hbm_usage_pct']):.2f}%",flush=True)
