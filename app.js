(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const sign = (value, digits = 2) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${number(Math.abs(value), digits)}`;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const mean = values => values.length && values.every(finite) ? values.reduce((a,b) => a+b,0)/values.length : null;
  const keys = ['pristine','v1','v2-r1','v2-r2','v2-r3'];
  const labels = {pristine:'Pristine',v1:'V1 edit','v2-r1':'Astra · round 1','v2-r2':'Astra · round 2','v2-r3':'Astra · round 3'};
  const shortLabels = {pristine:'Pristine',v1:'V1 edit','v2-r1':'V2 · round 1','v2-r2':'V2 · round 2','v2-r3':'V2 · round 3'};
  const subtitles = {pristine:'Original checkpoint',v1:'Historical selected edit','v2-r1':'Development candidate','v2-r2':'Development candidate','v2-r3':'Development candidate'};
  let data, methods, state = {route:'results',model:'v1',view:'quality',scope:'heldout',metric:'prompt_strict',session:null,request:1,trace:'response_ms'};
  try {
    data = JSON.parse($('#research-data').textContent);
    methods = JSON.parse($('#methodology-data').textContent);
    if (!data.v1?.heldout?.metrics || !data.v2?.execution) throw new Error('Missing published evidence');
  } catch (error) { $('#app-error').hidden = false; return; }
  const resource = () => data.resource_study;
  const rounds = data.v2.execution.rounds.filter(r => r.metrics);
  const methodSource = path => {
    if (/^https:\/\//.test(path)) return path;
    const clean = path.replace(/^\.\.\//,'');
    const known = {'reports/ifeval-browser-v1-final/README.md':data.sources.v1_report,'reports/ifeval-browser-v1-final/result.json':data.sources.v1_data,'docs/ifeval-v1-reproduction.md':data.sources.reproduction,'reports/ifeval-browser-v2-collaboration-run/README.md':data.sources.v2_execution,'docs/ifeval-v2-final-data.md':data.sources.final_data_audit};
    if (known[clean]) return known[clean];
    if (clean === 'reports/ifeval-browser-v2-collaboration-run/result.json') return data.sources.v2_execution.replace(/README\.md$/,'result.json');
    return `${data.sources.repository_base || 'https://github.com/Arhaan2/weight-space-lab/blob/codex/results-site/'}${clean}`;
  };
  const qualityMetrics = () => [
    {key:'prompt_strict',label:'Strict prompt accuracy',unit:'%',digits:2,direction:'higher'},
    {key:'prompt_loose',label:'Loose prompt accuracy',unit:'%',digits:2,direction:'higher'},
    ...(state.scope === 'heldout' ? [{key:'instruction_strict',label:'Strict instruction accuracy',unit:'%',digits:2,direction:'higher'},{key:'instruction_loose',label:'Loose instruction accuracy',unit:'%',digits:2,direction:'higher'}] : []),
    {key:'tokens',label:'Output tokens',unit:'tokens',digits:0,direction:'context'}
  ];
  const resourceMetrics = [
    {key:'response',label:'Mean response time',unit:'s / request',digits:2,direction:'lower',get:s=>s.measured_requests>0&&finite(s.response_ms_sum)?s.response_ms_sum/s.measured_requests/1000:null,description:'Mean full-response duration per measured request, excluding warm-ups. Session means are averaged equally.'},
    {key:'throughput',label:'Mean full-response throughput',unit:'tokens / s',digits:2,direction:'higher',get:s=>s.response_ms_sum>0&&finite(s.scored_output_tokens)?s.scored_output_tokens/(s.response_ms_sum/1000):null,description:'Calculate each session’s output tokens / summed response seconds, then average the two rates equally. This is not pooled throughput or decode-only speed; full response time includes prefill and overhead.'},
    {key:'native_peak_footprint_gb',label:'Sampled native footprint',unit:'GB',digits:3,direction:'lower',description:'Mean of per-session sampled peaks for the owned Chrome process tree, in decimal GB. Shared-memory attribution is imperfect; this is not physical VRAM.'},
    {key:'cpu_capacity_percent',label:'CPU capacity',unit:'%',digits:2,direction:'context',coverage:'cpu_coverage',description:'Active-interval weighted percentage of total logical-CPU capacity for the owned Chrome process tree. This is an activity rate, not cumulative CPU work; interpret it with response duration.'},
    {key:'gpu_time_ms_per_second',label:'Driver-accounted GPU time',unit:'ms / s',digits:2,direction:'context',coverage:'gpu_coverage',description:'Experimental AGX driver-accounted command-buffer time for the owned Chrome process tree. Includes browser rendering; concurrent queues can overlap. Not GPU utilization, occupancy or model-only time.'},
    {key:'host_power_watts',label:'Host power',unit:'W',digits:2,direction:'context',coverage:'power_coverage',description:'Host-wide sampled combined power during covered active intervals. Power is a rate, not energy; interpret it with duration. Includes unrelated machine activity and is not model-only.'},
    {key:'scored_output_tokens',label:'Output tokens',unit:'tokens / session',digits:0,direction:'context',description:'Mean sampled output tokens per session across measured requests. Warm-up tokens excluded.'},
    {key:'load_ms',label:'Model load time',unit:'s',digits:2,direction:'lower',get:s=>finite(s.load_ms)?s.load_ms/1000:null,description:'Model load duration per session. Interpret with the fixed cache policy in the protocol.'},
    {key:'gpu_buffer_descriptor_bytes_sampled_max',label:'GPUBuffer descriptors',unit:'GB',digits:3,direction:'context',get:s=>finite(s.gpu_buffer_descriptor_bytes_sampled_max)?s.gpu_buffer_descriptor_bytes_sampled_max/1e9:null,description:'Mean per-session sampled maximum of live GPUBuffer descriptor bytes. Allocation bookkeeping, not measured physical GPU residency.'},
    {key:'wasm_address_space_bytes_sampled_max',label:'WASM address space',unit:'MB',digits:2,direction:'context',get:s=>finite(s.wasm_address_space_bytes_sampled_max)?s.wasm_address_space_bytes_sampled_max/1e6:null,description:'Mean per-session sampled maximum of WASM linear-memory address space in decimal MB. Not native process footprint.'}
  ];
  const metrics = () => state.view === 'resources' ? resourceMetrics : qualityMetrics();
  const metric = () => metrics().find(m=>m.key===state.metric) || metrics()[0];
  const sessionsFor = key => (resource()?.sessions || []).filter(s=>s.artifact_key===key);
  const completed = session => session.status==='completed' && session.summary?.status==='complete' && session.summary.measured_requests===60 && session.summary.scheduled_requests===60;
  function quality(key, metricKey = state.metric) {
    if (state.scope === 'heldout') {
      if (!['pristine','v1'].includes(key)) return {value:null,n:0};
      const arm=key==='pristine'?'baseline':'candidate';
      if (metricKey==='tokens') return {value:data.v1.heldout.arms[arm].accounting.scored_generated_tokens,n:128};
      const reading=data.v1.heldout.metrics[metricKey][arm];
      return {value:reading.value*100,n:reading.denominator,count:reading.numerator};
    }
    const record=key==='pristine'?data.v1.development.baseline:key==='v1'?data.v1.development.candidate:null;
    const r=key.startsWith('v2-')?rounds.find(r=>key===`v2-r${r.round}`)?.metrics:null;
    if (!record&&!r) return {value:null,n:0};
    const n=record?.total??r.denominator;
    if(metricKey==='tokens') return {value:record?.output_tokens??r.scored_output_tokens,n};
    const count=metricKey==='prompt_loose'?(record?.loose??r.loose_passes):(record?.strict??r.strict_passes);
    return {value:count/n*100,count,n};
  }
  function reading(key) {
    if (state.view==='quality') return quality(key);
    const m=metric(), all=sessionsFor(key), complete=all.filter(completed);
    const values=complete.map(s=>m.get?m.get(s.summary):s.summary[m.key]).filter(finite);
    const hasPair=complete.length===2&&values.length===2;
    return {value:hasPair?mean(values):null,n:values.length,values,sessions:all,complete:complete.length,coverage:m.coverage?complete.map(s=>s.summary[m.coverage]).filter(finite):[]};
  }
  function readHash() {
    const [route,query='']=location.hash.slice(1).split('?');
    const params=new URLSearchParams(query);
    state.route=['results','methodology','evidence'].includes(route)?route:'results';
    if(params.has('model')&&keys.includes(params.get('model')))state.model=params.get('model');
    if(params.has('view')&&['quality','resources'].includes(params.get('view')))state.view=params.get('view');
    if(params.has('scope')&&['heldout','development'].includes(params.get('scope')))state.scope=params.get('scope');
    if(params.has('metric'))state.metric=params.get('metric');
    const session=Number(params.get('session')),request=Number(params.get('request'));
    if(Number.isInteger(session)&&session>=1&&session<=10)state.session=session;
    if(Number.isInteger(request)&&request>=1&&request<=60)state.request=request;
    if(['response_ms','output_tokens'].includes(params.get('trace')))state.trace=params.get('trace');
    if(!metrics().some(m=>m.key===state.metric))state.metric=metrics()[0].key;
  }
  function saveHash() {
    const params=new URLSearchParams({model:state.model,view:state.view,scope:state.scope,metric:state.metric});
    if(state.view==='resources'&&state.session){params.set('session',state.session);params.set('request',state.request);params.set('trace',state.trace);}
    history.replaceState(null,'',`#${state.route}${state.route==='results'?'?'+params:''}`);
  }
  function announce(text) { $('#announcement').textContent=text; }
  function renderNavigation() {
    $$('.route-view').forEach(el=>el.classList.toggle('active',el.id===state.route));
    $$('[data-route]').forEach(el=>{if(el.dataset.route===state.route)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
    document.title=`${state.route[0].toUpperCase()+state.route.slice(1)} · Weight Space Lab`;
  }
  function selectArtifact(key) {
    const focused=document.activeElement;
    const focusRegion=focused?.closest('#artifact-list')?'#artifact-list':focused?.closest('#comparison-chart')?'#comparison-chart':focused?.closest('#measurement-table')?'#measurement-table':null;
    state.model=key;
    if(state.view==='quality'&&state.scope==='heldout'&&key.startsWith('v2-')){state.scope='development';if(state.metric.startsWith('instruction'))state.metric='prompt_strict';}
    render();saveHash();
    if(focusRegion)document.querySelector(`${focusRegion} [data-model="${key}"]`)?.focus({preventScroll:true});
    announce(`${labels[key]}. ${metric().label}: ${formatReading(reading(key))}. ${scopeLabel()}.`);
  }
  function scopeLabel() {return state.view==='resources'?'Independent standard-workload resource pilot':state.scope==='heldout'?'V1 held-out set · 128 prompts / 173 instructions':'Development set · 32 prompts';}
  function formatReading(r,includeUnit=true) {return finite(r.value)?`${number(r.value,metric().digits)}${includeUnit?' '+metric().unit:''}`:'Unavailable';}
  function renderPickers() {
    $('#artifact-list').innerHTML=keys.map(key=>`<button type="button" class="artifact-button" data-model="${key}" aria-pressed="${key===state.model}"><strong>${labels[key]}</strong><small>${subtitles[key]}</small><span aria-hidden="true">${key===state.model?'↗':''}</span></button>`).join('');
    $('#mobile-model').innerHTML=keys.map(key=>`<option value="${key}"${key===state.model?' selected':''}>${shortLabels[key]}</option>`).join('');
    $$('[data-view]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.view===state.view)));
    $('#scope-field').hidden=state.view==='resources';$('#scope').value=state.scope;
    $('#metric').innerHTML=metrics().map(m=>`<option value="${m.key}"${m.key===state.metric?' selected':''}>${m.label}</option>`).join('');
  }
  function niceMax(max){if(max<=0)return 1;const power=10**Math.floor(Math.log10(max));const step=[1,2,2.5,5,10].find(s=>s*power>max/4)||10;return Math.ceil(max/(step*power))*step*power;}
  function continuationSummary() {
    const continuation=resource()?.continuation;if(!continuation)return '';
    const gap=continuation.inter_session_gap_after_position6_seconds;
    return `Documented continuation${finite(gap)?` · ${number(gap/60,2)}-minute gap`:'. Gap duration not recorded'}. This was not an uninterrupted run.`;
  }
  function renderResourceContext() {
    const context=$('#resource-context');
    context.hidden=state.view!=='resources'||!resource()?.continuation;
    context.innerHTML=context.hidden?'':`<p>${escape(continuationSummary())} <a href="#methodology">Recovery details</a></p>`;
  }
  function renderChart() {
    const m=metric(), readings=keys.map(key=>({key,...reading(key)}));
    const allValues=readings.flatMap(r=>[r.value,...(r.values||[])]).filter(finite);
    const max=state.view==='quality'&&state.metric!=='tokens'?100:niceMax(Math.max(...allValues,0)*1.04);
    $('#study-label').textContent=scopeLabel();$('#chart-title').textContent=m.label;
    $('#comparison-chart').innerHTML=readings.map(r=>`<button type="button" class="chart-row${finite(r.value)?'':' unavailable'}" data-model="${r.key}" data-artifact="${r.key}" aria-pressed="${r.key===state.model}" aria-label="${escape(labels[r.key]+': '+formatReading(r))}"><span class="chart-name">${shortLabels[r.key]}</span><span class="chart-track" aria-hidden="true">${finite(r.value)?`<span class="chart-bar" style="width:${r.value/max*100}%"></span>`:''}${(r.values||[]).map(v=>`<span class="session-point" title="Session: ${number(v,m.digits)} ${escape(m.unit)}" style="left:${v/max*100}%"></span>`).join('')}</span><span class="chart-value">${finite(r.value)?number(r.value,m.digits)+(m.unit==='%'?'%':''):'—'}</span></button>`).join('')+`<div class="chart-axis" aria-hidden="true"><span>0</span><span>${number(max/2,max<10?1:0)}</span><span>${number(max,max<10?1:0)}${m.unit==='%'?'%':''}</span></div>`;
    $('#chart-caption').textContent=state.view==='resources'?`Bars: mean of two completed sessions. Dots: individual sessions. Unit: ${m.unit}. Missing means remain unavailable until both sessions complete.`:state.metric==='tokens'?'Sampled output tokens over scored prompts. Lower output alone does not establish lower computation or energy.':'Each bar starts at zero. Exact counts and denominators are available below.';
    let note;
    if(state.view==='resources')note=`${m.description} Two sessions per artifact form a descriptive pilot; no resource-equivalence claim is established.`;
    else if(state.scope==='heldout')note=state.metric==='tokens'?'Historical V1 output observations: the edited run stopped after 94 prompts and resumed for 34. The run order and interruption prevent a controlled efficiency claim.':'V1’s paired confidence interval includes zero. The 128-prompt subset is not the full IFEval benchmark and is now exposed.';
    else note='Development-only evidence. The saved baseline and V1 used Chrome 156; new Astra candidates used Chrome 157. Repeated prompt exposure and the browser change limit causal and generalization claims.';
    $('#scope-note').innerHTML=`${escape(note)} <a href="#methodology">Read methods</a>`;
    renderSelected(readings.find(r=>r.key===state.model),readings[0]);
  }
  function renderSelected(r,baseline) {
    const m=metric();let delta='Reference artifact',signal='neutral';
    if(state.model!=='pristine') {
      if(state.view==='resources' && resource()?.matched_environment!==true) {
        delta=resource()?.status==='in_progress'?'Comparison pending: full-schedule environment verification is pending':'Comparison unavailable: matched environment not established';
      } else if(finite(r.value)&&finite(baseline.value)) {
        const diff=r.value-baseline.value;
        delta=`<strong>${sign(diff,m.digits)} ${state.view==='quality'&&state.metric!=='tokens'?'percentage points':escape(m.unit)}</strong> vs pristine`;
        if(diff!==0&&m.direction!=='context')signal=(m.direction==='higher'?diff>0:diff<0)?'positive':'negative';
      } else delta='No matched observation to compare';
    }
    let context=state.view==='quality'?(r.count!==undefined?`${number(r.count)} of ${r.n} ${state.metric.startsWith('instruction')?'instructions':'prompts'} passed.`:r.n?`${r.n} scored prompts. Sampled output tokens.`:'This artifact has no historical held-out evaluation.'):`${r.complete||0} of 2 sessions completed. ${r.n||0} available readings for this metric.`;
    if(state.view==='resources'&&r.coverage?.length)context+=` Active-time coverage: ${r.coverage.map(v=>number(v*100,1)+'%').join(' / ')}.`;
    if(state.view==='resources'&&r.values?.length)context+=` Individual readings: ${r.values.map(v=>number(v,m.digits)).join(' / ')} ${m.unit}.`;
    let qualityReference='';
    if(state.view==='resources') {
      const historical=state.model==='pristine'?data.v1.development.baseline:state.model==='v1'?data.v1.development.candidate:null;
      const candidate=rounds.find(row=>state.model===`v2-r${row.round}`)?.metrics;
      const passes=historical?.strict??candidate?.strict_passes;
      const denominator=historical?.total??candidate?.denominator;
      if(passes!==undefined)qualityReference=`<div class="quality-reference"><p>Historical development quality</p><strong>${passes} / ${denominator} strict passes</strong><p>Separate quality study. Chrome 156 baseline; Chrome 157 V2 candidates. <a href="#methodology">Scope &amp; caveats</a></p></div>`;
    }
    let interval='';
    if(state.view==='quality'&&state.scope==='heldout'&&state.model==='v1'&&state.metric!=='tokens') {
      const result=data.v1.heldout.metrics[state.metric], [low,high]=result.ci95_percentage_points, extent=Math.max(Math.abs(low),Math.abs(high),Math.abs(result.difference_percentage_points))*1.2;
      const at=x=>(x+extent)/(2*extent)*100;
      interval=`<div class="interval"><p class="interval-title">Paired 95% interval · change vs pristine</p><div class="interval-track" aria-hidden="true"><span class="interval-zero"></span><span class="interval-range" style="left:${at(low)}%;width:${at(high)-at(low)}%"></span><span class="interval-dot" style="left:${at(result.difference_percentage_points)}%"></span></div><div class="interval-values"><span>${sign(low,2)} pp</span><span>0</span><span>${sign(high,2)} pp</span></div><p class="interval-caption">Includes zero; improvement is unestablished.</p></div>`;
    }
    $('#selected-detail').innerHTML=`<p class="selected-label">Selected artifact</p><h3 class="selected-name">${labels[state.model]}</h3>${finite(r.value)?`<p class="selected-value">${number(r.value,m.digits)}<small>${escape(m.unit)}</small></p>`:'<p class="detail-empty">No measurement available</p>'}<p class="selected-delta" data-signal="${signal}">${delta}</p><p class="detail-context">${escape(context)}</p>${interval}${qualityReference}`;
  }
  function renderTable() {
    $('#table-scope').textContent=state.view==='resources'?'Two-session means':state.scope==='heldout'?'Historical V1':'Development only';
    const headers=state.view==='resources'?['Artifact',metric().unit,'Sessions','Range']:['Artifact','Strict','Loose','Output tokens'];
    const rows=keys.map(key=>{
      let cells;
      if(state.view==='resources') {const r=reading(key);cells=[formatReading(r,false),`${r.complete||0} / 2`,r.values?.length?`${number(Math.min(...r.values),metric().digits)}–${number(Math.max(...r.values),metric().digits)}`:'—'];}
      else {const a=quality(key,'prompt_strict'),b=quality(key,'prompt_loose'),t=quality(key,'tokens');cells=[a.n?`${a.count} / ${a.n}`:'—',b.n?`${b.count} / ${b.n}`:'—',number(t.value)];}
      return `<tr class="${key===state.model?'selected-row':''}"><th scope="row"><button type="button" class="row-select" data-model="${key}">${shortLabels[key]}</button></th>${cells.map((value,i)=>`<td data-label="${escape(headers[i+1])}">${escape(value)}</td>`).join('')}</tr>`;
    }).join('');
    $('#measurement-table').innerHTML=`<table class="data-table"><caption class="visually-hidden">${escape(scopeLabel())} comparison</caption><thead><tr>${headers.map(h=>`<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
  }
  function editInfo(key) {
    if(methods.edit_operations?.[key])return methods.edit_operations[key];
    if(key==='pristine')return {operations:[],changed_source_entries:0,artifact_sha256:data.v1.heldout.arms.baseline.deployed_artifact_sha256};
    if(key==='v1')return {operations:data.v1.edit.layers.map(layer=>({tensor:`model.language_model.layers.${layer}.mlp.down_proj.weight`,columns_inclusive:data.v1.edit.columns,factor:data.v1.edit.scale})),changed_source_entries:data.v1.edit.changed_bf16_entries,artifact_sha256:data.v1.heldout.arms.candidate.deployed_artifact_sha256};
    const r=rounds.find(r=>key===`v2-r${r.round}`);return {operations:r?.weight_operations||[],changed_source_entries:r?.source_edit?.changed_stored_entries,artifact_sha256:r?.artifact_sha256};
  }
  function renderEdit() {
    const info=editInfo(state.model), operations=info.operations||[];
    $('#edit-version').textContent=shortLabels[state.model];
    if(!operations.length){$('#edit-detail').innerHTML='<p class="edit-description">The unchanged source checkpoint is the reference for every edit. Candidates share the same architecture, runtime and quantization.</p><p class="operation-foot">0 source entries changed.</p><a class="provenance-link" href="#evidence">Inspect artifact identities ↗</a>';return;}
    const width=methods.model?.architecture?.intermediate_size;
    $('#edit-detail').innerHTML=`<p class="edit-description">${state.model==='v1'?'Historical manual proposal; Astra / Ultra attribution is unverified.':'Fresh-context gpt-6-astra proposal, applied independently to the pristine checkpoint.'}</p>${operations.map(op=>{
      const range=op.columns_inclusive||[Math.min(...op.columns),Math.max(...op.columns)], layer=op.tensor.match(/layers\.(\d+)/)?.[1];
      const map=width?`<div class="weight-map" aria-label="Layer ${layer}, columns ${range[0]} through ${range[1]} of ${width} scaled by ${op.factor}"><span class="weight-slice" style="left:${range[0]/width*100}%;width:${(range[1]-range[0]+1)/width*100}%"></span></div>`:'';
      return `<div class="edit-operation"><div class="edit-operation-head"><strong>Layer ${escape(layer)}</strong><span>× ${number(op.factor,2)}</span></div>${map}<code>Columns ${range[0]}–${range[1]} · down_proj</code></div>`;
    }).join('')}<p class="operation-foot">${number(info.changed_source_entries)} stored source entries changed.${width?` Maps show the actual 0–${width-1} column range.`:''}</p><details><summary>Exact edit operations</summary><pre>${escape(JSON.stringify(operations,null,2))}</pre></details><a class="provenance-link" href="#evidence">Inspect artifact identities ↗</a>`;
  }
  function renderSessions() {
    const panel=$('#session-panel');panel.hidden=state.view!=='resources';if(panel.hidden)return;
    const all=sessionsFor(state.model),m=metric();$('#session-count').textContent=`${shortLabels[state.model]} · ${all.length} recorded`;
    if(!all.length){$('#session-table').innerHTML='<p class="empty-state">No session summary is published for this artifact yet. The chart shows no inferred or placeholder measurements.</p>';return;}
    const rows=all.map(s=>{const summary=s.summary||{},v=m.get?m.get(summary):summary[m.key],coverage=m.coverage?summary[m.coverage]:null;return `<tr><th scope="row">Session ${s.position}<span class="cell-subline">${escape(s.status.replaceAll('_',' '))}</span></th><td data-label="Requests">${number(summary.measured_requests)} / ${number(summary.scheduled_requests)}</td><td data-label="${escape(m.unit)}">${number(v,m.digits)}</td><td data-label="Output tokens">${number(summary.scored_output_tokens)}</td><td data-label="Active coverage">${m.coverage?(finite(coverage)?number(coverage*100,1)+'%':'—'):'Not applicable'}</td></tr>`;}).join('');
    $('#session-table').innerHTML=`<table class="data-table"><caption class="visually-hidden">Individual standard-workload sessions for ${labels[state.model]}</caption><thead><tr><th scope="col">Session</th><th scope="col">Requests</th><th scope="col">${escape(m.unit)}</th><th scope="col">Output tokens</th><th scope="col">Active coverage</th></tr></thead><tbody>${rows}</tbody></table><p class="small-note">Session values are retained separately. Missing telemetry is unavailable, never zero. Warm-ups are excluded from measured-request token and response totals.</p><div id="request-explorer" class="request-explorer"></div>`;
    renderRequestExplorer();
  }

  function observedRequests(session) {return (session?.requests||[]).filter(row=>row.warmup===false);}
  function activeSession() {return sessionsFor(state.model).find(session=>session.position===state.session);}
  function renderRequestExplorer() {
    const container=$('#request-explorer');if(!container)return;
    const sessions=sessionsFor(state.model);
    let selected=sessions.find(session=>session.position===state.session);
    if(!selected)selected=sessions.find(session=>observedRequests(session).length)||sessions[0];
    if(!selected){container.innerHTML='';return;}
    state.session=selected.position;
    const requests=observedRequests(selected);
    state.request=Math.max(1,Math.min(state.request,requests.length||1));
    container.innerHTML=`<div class="section-heading"><div><h3>Inspect a session</h3><p class="small-note">Measured requests only · warm-up excluded</p></div><button type="button" class="text-button" id="export-requests"${requests.length?'':' disabled'}>Download session CSV ↓</button></div><div class="request-controls"><label>Session<select id="resource-session">${sessions.map(session=>`<option value="${session.position}"${session.position===state.session?' selected':''}>Session ${session.position} · ${escape(session.status.replaceAll('_',' '))}</option>`).join('')}</select></label>${requests.length?`<label>Trace<select id="request-metric"><option value="response_ms"${state.trace==='response_ms'?' selected':''}>Response duration · ms</option><option value="output_tokens"${state.trace==='output_tokens'?' selected':''}>Output tokens</option></select></label>`:''}</div>${requests.length?`<div id="request-traces" class="request-traces"></div><p class="small-note" id="request-trace-caption"></p><div class="request-scrubber"><button type="button" id="previous-request" class="outline-button" aria-label="Previous measured request">←</button><label for="request-index">Request <output id="request-position" for="request-index">${state.request}</output><input id="request-index" type="range" min="1" max="${requests.length}" step="1" value="${state.request}" aria-label="Measured request in execution order"></label><button type="button" id="next-request" class="outline-button" aria-label="Next measured request">→</button></div><div id="request-detail" class="request-detail" aria-live="polite" aria-atomic="true"></div>`:'<p class="empty-state">No measured request record is published for this session yet. Pending or stopped sessions are not filled with inferred observations.</p>'}`;
    if(requests.length)renderRequestTrace();
  }
  function renderRequestTrace() {
    const session=activeSession(),requests=observedRequests(session);if(!requests.length)return;
    const selected=requests[state.request-1],field=state.trace,unit=field==='response_ms'?'ms':'tokens',max=niceMax(Math.max(...requests.map(row=>finite(row[field])?row[field]:0),0)*1.05);
    const traces=[0,1,2].map(windowIndex=>{
      const start=windowIndex*20+1,end=start+19;
      const rows=requests.filter(row=>row.request_index>=start&&row.request_index<=end);
      const marks=rows.map(row=>{if(!finite(row[field]))return '';const height=row[field]/max*65,x=36+(row.request_index-start)*12.5;return `<rect x="${x}" y="${85-height}" width="9" height="${height}" fill="${row===selected?'var(--accent)':'#858e82'}"><title>Request ${row.request_index}: ${number(row[field],field==='response_ms'?2:0)} ${unit}</title></rect>`;}).join('');
      const marker=selected.request_index>=start&&selected.request_index<=end?`<path d="M${40.5+(selected.request_index-start)*12.5} 89v5" stroke="var(--accent)" stroke-width="2"/>`:'';
      return `<div class="request-window"><p>Window ${windowIndex+1}<span>Requests ${start}–${end}</span></p><svg viewBox="0 0 300 112" role="img" aria-label="${field==='response_ms'?'Response duration':'Output tokens'} for requests ${start} through ${end}"><path d="M34 20H288M34 52.5H288" stroke="var(--rule)"/><path d="M34 85H288" stroke="var(--control)"/><text x="29" y="24" text-anchor="end">${number(max)}</text><text x="29" y="56.5" text-anchor="end">${number(max/2,max%2?1:0)}</text><text x="29" y="89" text-anchor="end">0</text>${marks}${marker}<text x="36" y="108">${start}</text><text x="283" y="108" text-anchor="end">${end}</text>${rows.length?'':'<text x="160" y="50" text-anchor="middle">No observed requests</text>'}</svg></div>`;
    }).join('');
    $('#request-traces').innerHTML=traces;
    const gaps=session.summary?.idle_gaps_ms;
    $('#request-trace-caption').textContent=`Horizontal position is request order, not elapsed time. Vertical unit: ${unit}. All three windows share the same scale.${gaps?.length?' Observed idle gaps between windows: '+gaps.map(value=>number(value/1000,2)+' s').join(' and ')+'.':' Idle-gap durations are unavailable in this session summary.'}`;
    $('#request-position').textContent=`${selected.request_index} of 60`;
    const range=$('#request-index');range.value=state.request;range.setAttribute('aria-valuetext',`Request ${selected.request_index} of 60, ${number(selected[field],field==='response_ms'?2:0)} ${unit}`);
    $('#previous-request').disabled=state.request<=1;$('#next-request').disabled=state.request>=requests.length;
    const fields=[['Response duration',number(selected.response_ms,2)+' ms'],['Output tokens',number(selected.output_tokens)],['Finish reason',selected.finish_reason??'—'],['First output',finite(selected.first_output_ms)?number(selected.first_output_ms,2)+' ms':'—'],['Reported decode rate',finite(selected.decode_tokens_per_second)?number(selected.decode_tokens_per_second,2)+' tokens/s':'—'],['Observation validity',selected.valid===true?'Valid':selected.valid===false?'Invalid':'Unavailable']];
    $('#request-detail').innerHTML=fields.map(([label,value])=>`<div><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`).join('');
  }
  function exportRequests() {
    const session=activeSession();if(!session)return;
    const fields=['request_index','valid','response_ms','input_tokens','output_tokens','finish_reason','first_output_ms','decode_tokens_per_second','started_ms','ended_ms'];
    const rows=[['artifact','session',...fields],...observedRequests(session).map(row=>[state.model,session.position,...fields.map(field=>row[field]??'')])];
    downloadCSV(rows,`weight-space-lab-${state.model}-session-${session.position}-requests.csv`);
    announce(`Session ${session.position} measured request CSV downloaded. Warm-up and response text excluded.`);
  }

  function pretty(key){return key.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());}
  function detailList(object){return `<dl>${Object.entries(object||{}).map(([k,v])=>`<div><dt>${escape(pretty(k))}</dt><dd>${escape(typeof v==='object'?JSON.stringify(v):String(v))}</dd></div>`).join('')}</dl>`;}
  function continuationMethodsHTML() {
    const continuation=resource()?.continuation;if(!continuation)return '';
    const recovery=continuation.recovery||{};
    const facts={'Preserved completed positions':(continuation.inherited_completed_positions||[]).join(', '),'Continuation positions':(continuation.remaining_positions||[]).join(', '),'Failure stage':continuation.failure_stage,'Failure type':continuation.failure_type,'Recovered file role':recovery.file_role,'Recovered bytes':number(recovery.size_bytes),'Recovered SHA-256':recovery.sha256,'Recovery method':recovery.method,'Gap after position 6':finite(continuation.inter_session_gap_after_position6_seconds)?number(continuation.inter_session_gap_after_position6_seconds,2)+' seconds':'Not yet recorded','Original calendar and total limits':continuation.original_calendar_and_total_limits_unchanged===true?'Unchanged':'Not established','Replaced completed sessions':continuation.replaced_completed_sessions,'Model dispatch retries':continuation.model_dispatch_retries};
    return `<details><summary>Documented resource continuation</summary><div class="method-body"><p>${escape(continuation.interruption_note)}</p>${detailList(facts)}<p>Hardware and runtime matching does not establish constant background host activity or uninterrupted conditions. Both R3 observations remain in the descriptive comparison.</p></div></details>`;
  }
  function resourceEnvironmentHTML() {
    const report=resource();if(!report?.environment)return '';
    const environment=report.environment,hardware=environment.hardware||{},bytes=Number(hardware['hw.memsize']);
    const values={'Processor':hardware['machdep.cpu.brand_string']??'Unavailable','Logical CPUs':hardware['hw.logicalcpu']??'Unavailable','Physical / unified memory':finite(bytes)&&bytes>0?`${number(bytes/2**30,bytes%(2**30)?2:0)} GiB (${number(bytes)} bytes)`:'Unavailable','Hardware model':hardware['hw.model']??'Unavailable','macOS':environment.macos??'Unavailable','Chrome':environment.browser_version??'Unavailable','Runtime':environment.runtime??'Unavailable','Quantization':environment.quantization??'Unavailable'};
    const verification=report.matched_environment===true?'Matched hardware and runtime verified across the completed schedule.':report.status==='in_progress'?'Hardware and runtime matching remains pending until the full schedule completes.':'Hardware and runtime matching was not established for the full schedule.';
    return `<details><summary>Resource environment</summary><div class="method-body"><p>${escape(verification)}</p>${detailList(values)}<p>Memory is the operating system’s reported physical / unified capacity, converted from bytes to GiB. It is not a measurement of model memory or physical GPU residency.</p></div></details>`;
  }
  function renderMethods() {
    const studies=(methods.studies||[]).map(study=>`<details><summary>${escape(study.title)}</summary><div class="method-body"><p>${escape(study.question)}</p><p><strong>Scope.</strong> ${escape(study.scope)}</p><p><strong>Selection.</strong> ${escape(study.selection_rule)}</p>${study.aggregation?`<p><strong>Aggregation.</strong> ${escape(study.aggregation)}</p>`:''}${study.public_workload_prompt?`<p><strong>Public resource workload prompt.</strong> “${escape(study.public_workload_prompt)}”</p>`:''}${detailList(study.generation_settings)}<ul>${(study.limitations||[]).map(s=>`<li>${escape(s)}</li>`).join('')}</ul><p>${(study.sources||[]).map(s=>`<a href="${escape(methodSource(s.url))}">${escape(s.label)}</a>`).join(' · ')}</p></div></details>`).join('');
    const telemetry=Object.values(methods.telemetry_definitions||{}).map(def=>`<div><dt>${escape(def.label)}${def.unit?' · '+escape(def.unit):''}</dt><dd>${escape(def.definition)}${def.scope?' '+escape(def.scope):''}${def.coverage?' Coverage: '+escape(def.coverage):''}${def.limitations?' '+escape(Array.isArray(def.limitations)?def.limitations.join(' '):def.limitations):''}</dd></div>`).join('');
    $('#methodology-detail').innerHTML=studies+continuationMethodsHTML()+resourceEnvironmentHTML()+`<details><summary>Scoring and uncertainty</summary><div class="method-body">${detailList(methods.scoring)}</div></details><details><summary>Telemetry definitions and coverage</summary><div class="method-body"><dl>${telemetry}</dl></div></details><details><summary>Reproduction and preserved failures</summary><div class="method-body"><ul>${[...(methods.reproduction||[]),...(methods.preserved_failures||[])].map(s=>`<li>${escape(s)}</li>`).join('')}</ul></div></details>`;
    if(resource()) {
      const count=(resource().sessions||[]).filter(completed).length;
      $('#resource-method-state').textContent=pretty(resource().status||'reported');
      $('#resource-method-status').textContent=`${count} of 10 planned sessions are complete in the public report. Quality eligibility was not required. Two sessions per artifact form a descriptive pilot, not an equivalence test.`;
    }
  }
  function renderSources() {
    const sources=[['V1 held-out report','Historical quality, paired outcomes and uncertainty',data.sources.v1_report],['V2 collaboration report','Three candidate evaluations and selection outcome',data.sources.v2_execution],['Quality and resource protocol','Frozen objective and conditional comparison',data.sources.v2_protocol],['Untouched-data audit','Exposure accounting for final-quality claims',data.sources.final_data_audit],['Reproduction guide','Source pins, transformations and evaluation',data.sources.reproduction],['Full research methodology','Scopes, artifact identities and counter definitions',methodSource('../docs/research-methodology.md')]];
    if(resource())sources.splice(2,0,['All-artifact resource report','Individual sessions, telemetry coverage and aggregates',data.sources.resource_study||methodSource('../reports/resource-all-edits-v1/result.json')]);
    $('#source-list').innerHTML=sources.map(([title,description,url])=>`<a href="${escape(url)}"><span>${escape(title)}</span><small>${escape(description)}</small><span aria-hidden="true">↗</span></a>`).join('');
    $('#identity-table').innerHTML=`<table class="data-table hash-table"><thead><tr><th scope="col">Artifact</th><th scope="col">Deployed artifact SHA-256</th></tr></thead><tbody>${keys.map(key=>`<tr><th scope="row">${labels[key]}</th><td data-label="Deployed artifact SHA-256" class="hash">${escape(editInfo(key).artifact_sha256||'Unavailable')}</td></tr>`).join('')}</tbody></table>`;
  }
  function render() {renderNavigation();renderPickers();renderResourceContext();renderChart();renderTable();renderEdit();renderSessions();}
  function exportCSV() {
    const m=metric(),rows=[['artifact','study','metric','value','unit','sample_count','individual_session_values']];
    keys.forEach(key=>{const r=reading(key);rows.push([key,scopeLabel(),m.label,r.value??'',m.unit,r.n,(r.values||[]).join(';')]);});
    downloadCSV(rows,`weight-space-lab-${state.view}-${state.metric}.csv`);announce('Current comparison exported as CSV.');
  }
  function downloadCSV(rows,filename) {
    const csv=rows.map(row=>row.map(cell=>'"'+String(cell).replaceAll('"','""')+'"').join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
  }
  $('.skip-link').addEventListener('click',event=>{event.preventDefault();const target=$(`#${state.route}-heading`)||$('#main');target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start'});});
  document.addEventListener('click',event=>{const button=event.target.closest('[data-model]');if(button)selectArtifact(button.dataset.model);if(event.target.closest('#export-requests'))exportRequests();const step=event.target.closest('#previous-request')?-1:event.target.closest('#next-request')?1:0;if(step){state.request=Math.max(1,Math.min(observedRequests(activeSession()).length,state.request+step));renderRequestTrace();saveHash();}});
  document.addEventListener('change',event=>{if(event.target.id==='resource-session'){state.session=Number(event.target.value);state.request=1;renderRequestExplorer();$('#resource-session')?.focus({preventScroll:true});saveHash();}if(event.target.id==='request-metric'){state.trace=event.target.value;renderRequestTrace();saveHash();}});
  document.addEventListener('input',event=>{if(event.target.id==='request-index'){state.request=Number(event.target.value);renderRequestTrace();saveHash();}});
  $$('[data-view]').forEach(button=>button.addEventListener('click',()=>{state.view=button.dataset.view;state.metric=state.view==='resources'?'response':'prompt_strict';render();saveHash();announce(`${state.view==='resources'?'Resource':'Quality'} measurements selected.`);}));
  $('#scope').addEventListener('change',event=>{state.scope=event.target.value;if(!metrics().some(m=>m.key===state.metric))state.metric='prompt_strict';render();saveHash();announce(scopeLabel());});
  $('#metric').addEventListener('change',event=>{state.metric=event.target.value;render();saveHash();announce(`${metric().label}: ${formatReading(reading(state.model))} for ${labels[state.model]}.`);});
  $('#mobile-model').addEventListener('change',event=>selectArtifact(event.target.value));
  $('#export-csv').addEventListener('click',exportCSV);
  $('#copy-link').addEventListener('click',async()=>{saveHash();try{await navigator.clipboard.writeText(location.href);$('#copy-link').textContent='Link copied';announce('View link copied.');setTimeout(()=>$('#copy-link').textContent='Copy view link',1800);}catch{$('#copy-link').textContent='Use the address bar';announce('Copy this view from the browser address bar.');}});
  $$('.print-button').forEach(button=>button.addEventListener('click',()=>window.print()));
  window.addEventListener('hashchange',()=>{const prior=state.route;readHash();render();if(prior!==state.route){window.scrollTo(0,0);const heading=$(`#${state.route}-heading`);heading.setAttribute('tabindex','-1');heading.focus({preventScroll:true});announce(`${pretty(state.route)} view.`);}});
  let printDetails=[];
  window.addEventListener('beforeprint',()=>{printDetails=$$('.methods-details details').map(el=>[el,el.open]);printDetails.forEach(([el])=>el.open=true);});
  window.addEventListener('afterprint',()=>{printDetails.forEach(([el,open])=>el.open=open);printDetails=[];});
  if(!location.hash && ['complete','completed'].includes(resource()?.status)) {state.view='resources';state.metric='response';}
  readHash();renderMethods();renderSources();render();
  $('.interactive-workspace').hidden=false;document.documentElement.classList.add('js-ready');
  $('#snapshot-date').textContent=new Date(data.updated+'T12:00:00').toLocaleDateString('en-GB',{day:'numeric',month:'long',year:'numeric'});
})();
