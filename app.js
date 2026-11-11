(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const sign = (value, digits = 2) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${number(Math.abs(value), digits)}`;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const mean = values => values.length && values.every(finite) ? values.reduce((a,b) => a+b,0)/values.length : null;
  const historicalKeys = ['pristine','v1','v2-r1','v2-r2','v2-r3'];
  let keys = [...historicalKeys];
  const labels = {pristine:'Pristine',v1:'V1 edit','v2-r1':'Astra · round 1','v2-r2':'Astra · round 2','v2-r3':'Astra · round 3','astra-r4':'Astra · round 4'};
  const shortLabels = {pristine:'Pristine',v1:'V1 edit','v2-r1':'Astra 1','v2-r2':'Astra 2','v2-r3':'Astra 3','astra-r4':'Astra 4'};
  const pointLabels = {pristine:'P',v1:'V1','v2-r1':'A1','v2-r2':'A2','v2-r3':'A3','astra-r4':'A4'};
  const subtitles = {pristine:'Original checkpoint',v1:'Historical selected edit','v2-r1':'Development candidate','v2-r2':'Development candidate','v2-r3':'Development candidate','astra-r4':'Fresh-baseline candidate'};
  let data, methods, state = {route:'results',model:'v1',view:'overview',cohort:'historical',scope:'development',metric:'response',qualityMetric:'prompt_strict',zoom:true,session:null,request:1,trace:'response_ms',record:null};
  try {
    data = JSON.parse($('#research-data').textContent);
    methods = JSON.parse($('#methodology-data').textContent);
    if (!data.v1?.heldout?.metrics || !data.v2?.execution) throw new Error('Missing published evidence');
  } catch (error) { $('#app-error').hidden = false; return; }
  const fresh = () => data.fresh_comparison;
  const artifactKey = artifact => artifact.key || artifact.artifact_key;
  const freshArtifacts = () => Array.isArray(fresh()?.artifacts) ? fresh().artifacts : [];
  const allKeys = [...historicalKeys,...freshArtifacts().map(artifactKey).filter(key=>key&&!historicalKeys.includes(key))];
  freshArtifacts().forEach(artifact=>{const key=artifactKey(artifact);if(!labels[key])labels[key]=artifact.label||key;if(!shortLabels[key])shortLabels[key]=artifact.label||key;if(!pointLabels[key])pointLabels[key]=key;if(!subtitles[key])subtitles[key]='Measured weight artifact';});
  const resource = () => state.cohort==='fresh' ? fresh()?.resource : data.resource_study;
  const cohortLabel = () => state.cohort==='fresh'?'Fresh baseline comparison':'Historical observations';
  function syncCohort() {if(state.cohort==='fresh'&&!fresh())state.cohort='historical';keys=state.cohort==='fresh'?[...allKeys]:[...historicalKeys];if(!keys.includes(state.model))state.model='pristine';}
  const freshQuality = key => fresh()?.quality?.artifacts?.find(row=>row.artifact_key===key);
  const rounds = data.v2.execution.rounds.filter(r => r.metrics);
  const methodSource = path => {
    if (/^https:\/\//.test(path)) return path.replace(/\/README\.md$/,'/result.json');
    const clean = path.replace(/^\.\.\//,'');
    const known = {'reports/ifeval-browser-v1-final/README.md':data.sources.v1_report,'reports/ifeval-browser-v1-final/result.json':data.sources.v1_data,'docs/ifeval-v1-reproduction.md':data.sources.reproduction,'reports/ifeval-browser-v2-collaboration-run/README.md':data.sources.v2_execution,'docs/ifeval-v2-final-data.md':data.sources.final_data_audit};
    if (known[clean]) return known[clean].replace(/\/README\.md$/,'/result.json');
    if (clean === 'reports/ifeval-browser-v2-collaboration-run/result.json') return data.sources.v2_execution.replace(/README\.md$/,'result.json');
    return `${data.sources.repository_base || 'https://github.com/Arhaan2/weight-space-lab/blob/codex/results-site/'}${clean}`.replace(/\/README\.md$/,'/result.json');
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
    {key:'wasm_address_space_bytes_sampled_max',label:'WASM address space',unit:'MB',digits:2,direction:'context',get:s=>finite(s.wasm_address_space_bytes_sampled_max)?s.wasm_address_space_bytes_sampled_max/1e6:null,description:'Mean per-session sampled maximum of WASM linear-memory address space in decimal MB. Not native process footprint.'},
    {key:'response_ms_sum',label:'Measured response duration',unit:'s / session',digits:2,direction:'lower',get:s=>finite(s.response_ms_sum)?s.response_ms_sum/1000:null,description:'Sum of full-response durations for the 60 measured requests, averaged across two sessions. Excludes model loading, warm-ups, gaps and idle intervals.'},
    {key:'session_ms',label:'Profile duration',unit:'s',digits:2,direction:'context',get:s=>finite(s.session_ms)?s.session_ms/1000:null,description:'Mean measured profile duration, including the prescribed baseline, idle and recovery phases. This is separate from response time and controller-stage elapsed time, and is not cumulative CPU or GPU work.'}
  ];
  const metrics = () => state.view !== 'quality' ? resourceMetrics : qualityMetrics();
  const metric = () => metrics().find(m=>m.key===state.metric) || metrics()[0];
  const sessionsFor = key => (resource()?.sessions || []).filter(s=>s.artifact_key===key);
  const completed = session => session.status==='completed' && session.summary?.status==='complete' && session.summary.measured_requests===60 && session.summary.scheduled_requests===60;
  function quality(key, metricKey = state.metric, scope = state.scope) {
    if(state.cohort==='fresh') {
      const row=freshQuality(key),r=row?.metrics;
      if(!r||!['completed','complete'].includes(row.status))return {value:null,n:0};
      const n=r.denominator;
      if(metricKey==='tokens')return {value:r.scored_output_tokens??null,n};
      const count=metricKey==='prompt_loose'?r.loose_passes:r.strict_passes;
      return {value:n>0&&finite(count)?count/n*100:null,count,n};
    }
    if (scope === 'heldout') {
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
  function resourceReading(key,m=metric()) {
    const all=sessionsFor(key), complete=all.filter(completed);
    const values=complete.map(s=>m.get?m.get(s.summary):s.summary[m.key]).filter(finite);
    const hasPair=complete.length===2&&values.length===2;
    return {value:hasPair?mean(values):null,n:values.length,values,sessions:all,complete:complete.length,coverage:m.coverage?complete.map(s=>s.summary[m.coverage]).filter(finite):[]};
  }
  function activityDurationContext(key) {
    if(!['cpu_capacity_percent','gpu_time_ms_per_second','host_power_watts'].includes(state.metric))return '';
    const duration=resourceReading(key,resourceMetrics.find(m=>m.key==='response_ms_sum'));
    const session=resourceReading(key,resourceMetrics.find(m=>m.key==='session_ms'));
    if(!finite(duration.value)&&!finite(session.value))return '';
    return `Duration context: ${finite(duration.value)?`${number(duration.value,2)} s of measured responses per session`:''}${finite(duration.value)&&finite(session.value)?'; ':''}${finite(session.value)?`${number(session.value,2)} s for the measured profile`:''}. These durations are not a model-only integration interval for the activity counters.`;
  }
  function reading(key) {return state.view==='quality'?quality(key):resourceReading(key);}
  function readHash() {
    const [route,query='']=location.hash.slice(1).split('?');
    const params=new URLSearchParams(query);
    state.route=['results','methodology','evidence'].includes(route)?route:'results';
    if(params.has('cohort')&&['fresh','historical'].includes(params.get('cohort')))state.cohort=params.get('cohort');
    syncCohort();
    if(params.has('model')&&keys.includes(params.get('model')))state.model=params.get('model');
    if(params.has('view')&&['overview','quality','resources'].includes(params.get('view')))state.view=params.get('view');
    if(params.has('scope')&&['heldout','development'].includes(params.get('scope')))state.scope=params.get('scope');
    if(params.has('metric'))state.metric=params.get('metric');
    if(['prompt_strict','prompt_loose'].includes(params.get('quality')))state.qualityMetric=params.get('quality');
    if(params.has('zoom'))state.zoom=params.get('zoom')!=='false';
    state.record=params.get('record');
    const session=Number(params.get('session')),request=Number(params.get('request'));
    if(Number.isInteger(session)&&session>=1&&session<=100)state.session=session;
    if(Number.isInteger(request)&&request>=1&&request<=60)state.request=request;
    if(['response_ms','output_tokens'].includes(params.get('trace')))state.trace=params.get('trace');
    if(state.cohort==='fresh')state.scope='development';
    if(!metrics().some(m=>m.key===state.metric))state.metric=metrics()[0].key;
  }
  function saveHash() {
    const params=new URLSearchParams({model:state.model,view:state.view,cohort:state.cohort,scope:state.scope,metric:state.metric});
    if(state.view==='overview'){params.set('quality',state.qualityMetric);params.set('zoom',String(state.zoom));}
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
    const focusRegion=['#artifact-list','#comparison-chart','#measurement-table','#overview-table','#tradeoff-plot'].find(selector=>focused?.closest(selector));
    state.model=key;
    if(state.view==='quality'&&state.scope==='heldout'&&key.startsWith('v2-')){state.scope='development';if(state.metric.startsWith('instruction'))state.metric='prompt_strict';}
    render();saveHash();
    if(focusRegion)document.querySelector(`${focusRegion} [data-model="${key}"]`)?.focus({preventScroll:true});
    announce(`${labels[key]}. ${metric().label}: ${formatReading(reading(key))}. ${scopeLabel()}.`);
  }
  function scopeLabel() {return state.view==='overview'?`${cohortLabel()} · quality and resources`:state.view==='resources'?`${cohortLabel()} · standard resource workload`:state.cohort==='fresh'?'Fresh comparison · 32 exposed development prompts':state.scope==='heldout'?'V1 held-out set · 128 prompts / 173 instructions':'Historical development set · 32 prompts';}
  function formatReading(r,includeUnit=true) {return finite(r.value)?`${number(r.value,metric().digits)}${includeUnit?' '+metric().unit:''}`:'Unavailable';}
  function renderPickers() {
    $('.workspace-toolbar').classList.toggle('overview-toolbar',state.view==='overview');
    $('#artifact-count').textContent=`${keys.length} versions`;
    $('#artifact-list').innerHTML=keys.map(key=>`<button type="button" class="artifact-button" data-model="${key}" aria-pressed="${key===state.model}"><strong>${labels[key]}</strong><small>${subtitles[key]}</small><span aria-hidden="true">${key===state.model?'↗':''}</span></button>`).join('');
    $('#mobile-model').innerHTML=keys.map(key=>`<option value="${key}"${key===state.model?' selected':''}>${shortLabels[key]}</option>`).join('');
    $$('[data-view]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.view===state.view)));
    $('#scope-field').hidden=state.view!=='quality'||state.cohort==='fresh';$('#scope').value=state.scope;
    $('#metric').innerHTML=metrics().map(m=>`<option value="${m.key}"${m.key===state.metric?' selected':''}>${m.label}</option>`).join('');
    $('#cohort').innerHTML=`${fresh()?'<option value="fresh">Fresh baseline comparison</option>':''}<option value="historical">Historical observations</option>`;
    $('#cohort').value=state.cohort;
    $('#cohort-context').textContent=state.cohort==='fresh'?'All versions are scheduled against a newly measured pristine baseline. Missing or incomplete measurements stay unavailable.':'Quality and resource values came from separate studies. Historical quality used different Chrome versions; this is descriptive context.';
  }
  function niceMax(max){if(max<=0)return 1;const power=10**Math.floor(Math.log10(max));const step=[1,2,2.5,5,10].find(s=>s*power>max/4)||10;return Math.ceil(max/(step*power))*step*power;}
  function continuationSummary() {
    const continuation=resource()?.continuation;if(!continuation)return '';
    const gap=continuation.inter_session_gap_after_position6_seconds;
    return `Documented continuation${finite(gap)?` · ${number(gap/60,2)}-minute gap`:'. Gap duration not recorded'}. This was not an uninterrupted run.`;
  }
  function renderResourceContext() {
    const context=$('#resource-context');
    context.hidden=!['resources','overview'].includes(state.view)||!resource()?.continuation;
    context.innerHTML=context.hidden?'':`<p>${escape(continuationSummary())} <a href="#methodology">Recovery details</a></p>`;
  }
  function overviewRows() {
    return keys.map(key=>({key,quality:quality(key,state.qualityMetric,'development'),resource:resourceReading(key),strict:quality(key,'prompt_strict','development'),loose:quality(key,'prompt_loose','development'),tokens:quality(key,'tokens','development')}));
  }
  function renderOverview() {
    const visible=state.view==='overview';$('#overview-panel').hidden=!visible;
    $('#single-metric-panel').hidden=visible;$('.evidence-grid').hidden=visible;
    if(!visible)return;
    $('#overview-cohort').textContent=cohortLabel();
    $('#overview-quality').value=state.qualityMetric;
    Array.from($('#overview-quality').options).forEach(option=>{option.textContent=option.value==='prompt_strict'?'Strict prompts':'Loose prompts';});
    const optionLabels={response:'Response time',throughput:'Throughput',native_peak_footprint_gb:'Native footprint',cpu_capacity_percent:'CPU capacity',gpu_time_ms_per_second:'GPU time',host_power_watts:'Host power',scored_output_tokens:'Output tokens',load_ms:'Model load time',gpu_buffer_descriptor_bytes_sampled_max:'GPU descriptors',wasm_address_space_bytes_sampled_max:'WASM address',response_ms_sum:'Response total',session_ms:'Profile duration'};
    $('#overview-resource').innerHTML=resourceMetrics.map(m=>`<option value="${m.key}"${m.key===state.metric?' selected':''}>${optionLabels[m.key]||m.label}</option>`).join('');
    $('#overview-zoom').checked=state.zoom;
    const rows=overviewRows(),m=metric(),selected=rows.find(row=>row.key===state.model),baseline=rows.find(row=>row.key==='pristine');
    $('#overview-table-scope').textContent=state.cohort==='fresh'?'Fresh measurements only':'Historical measurements';
    const columns=[['Strict passes',row=>row.strict.n?`${row.strict.count} / ${row.strict.n}`:'—'],['Loose passes',row=>row.loose.n?`${row.loose.count} / ${row.loose.n}`:'—'],['Quality tokens',row=>number(row.tokens.value)],['Response · s/req',row=>number(resourceReading(row.key,resourceMetrics[0]).value,2)],['Native peak · GB',row=>number(resourceReading(row.key,resourceMetrics[2]).value,3)],['CPU · %',row=>number(resourceReading(row.key,resourceMetrics[3]).value,2)],['GPU · ms/s',row=>number(resourceReading(row.key,resourceMetrics[4]).value,2)]];
    $('#overview-table').innerHTML=`<table class="data-table overview-table"><caption class="visually-hidden">All ${keys.length} artifacts, ${escape(cohortLabel())}. Resource values are means of two completed sessions.</caption><thead><tr><th scope="col">Version</th>${columns.map(([title])=>`<th scope="col">${escape(title)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr class="${row.key===state.model?'selected-row':''}"><th scope="row"><button type="button" class="row-select" data-model="${row.key}" aria-pressed="${row.key===state.model}"><span class="version-symbol" aria-hidden="true">${pointLabels[row.key]}</span>${shortLabels[row.key]}</button>${state.cohort==='fresh'&&(!row.strict.n||row.resource.complete!==2)?`<span class="cell-subline">${row.strict.n?'Quality complete':'Quality unavailable'} · ${row.resource.complete}/2 resource sessions</span>`:''}</th>${columns.map(([title,get])=>`<td data-label="${escape(title)}">${escape(get(row))}</td>`).join('')}</tr>`).join('')}</tbody></table><p class="small-note">Quality tokens cover scored development prompts. Resource columns average two 60-request sessions per version. CPU and experimental GPU counters are activity rates, not total work. Host power is unavailable unless explicitly observed. <a href="#methodology">Measurement definitions</a></p>`;
    const qualityMatched=state.cohort==='fresh'&&fresh()?.quality?.matched_environment===true;
    const resourceMatched=resource()?.matched_environment===true;
    let comparison='Pristine is the reference artifact.';
    if(state.model!=='pristine') {
      const qDelta=finite(selected.quality.value)&&finite(baseline.quality.value)?selected.quality.value-baseline.quality.value:null;
      const rDelta=finite(selected.resource.value)&&finite(baseline.resource.value)?selected.resource.value-baseline.resource.value:null;
      if(state.cohort==='fresh'&&qualityMatched&&resourceMatched&&finite(qDelta)&&finite(rDelta))comparison=`${sign(qDelta,2)} percentage points and ${sign(rDelta,m.digits)} ${m.unit} vs the fresh pristine baseline.`;
      else if(state.cohort==='fresh')comparison='Matched comparison is unavailable until both quality and resource environment checks pass and observations complete.';
      else comparison='Historical quality and resource observations are from separate runs; their pairing does not establish a controlled joint improvement.';
    }
    const q=selected.quality,r=selected.resource;
    $('#overview-detail').innerHTML=`<p class="selected-label">Selected version</p><h3>${escape(labels[state.model])}</h3><dl><div><dt>${state.qualityMetric==='prompt_strict'?'Strict':'Loose'} prompt accuracy</dt><dd>${finite(q.value)?`${number(q.value,2)}<small class="percent-unit">%</small>`:'Unavailable'}${q.n?`<span>${q.count} of ${q.n} prompts</span>`:''}</dd></div><div><dt>${escape(m.label)}</dt><dd>${finite(r.value)?`${number(r.value,m.digits)}<small>${escape(m.unit)}</small>`:'Unavailable'}<span>${r.complete||0} of 2 sessions complete</span></dd></div></dl><p class="overview-delta">${escape(comparison)}</p>${r.values?.length?`<p class="small-note">Individual sessions: ${r.values.map(value=>number(value,m.digits)).join(' / ')} ${escape(m.unit)}.</p>`:''}${activityDurationContext(state.model)?`<p class="small-note">${escape(activityDurationContext(state.model))}</p>`:''}<button type="button" class="outline-button" id="inspect-version">Inspect this version →</button>`;
    const freshNote='Fresh baseline and candidate evaluations use the same exposed development prompts. This can establish a development-score change, not held-out generalization. Two resource sessions per version remain a descriptive pilot.';
    const historicalNote='The figure pairs historical development scores with the separate all-edit resource pilot. Chrome 156 baseline/V1 quality and Chrome 157 Astra quality are not a matched quality cohort. It does not establish a quality–resource improvement.';
    $('#overview-limit').innerHTML=`${escape(state.cohort==='fresh'?freshNote:historicalNote)} ${escape(m.description)} <a href="#methodology">Read the methodology</a>`;
    renderOverviewPlot();
  }
  function axisRange(values,{percent=false,focused=false}={}) {
    if(percent&&!focused)return {min:0,max:100,step:25};
    if(!values.length)return {min:0,max:percent?100:1,step:percent?25:.25};
    const lo=Math.min(...values),hi=Math.max(...values);
    const span=hi-lo||Math.max(Math.abs(hi)*.1,percent?6.25:.01);
    const lower=focused?Math.max(0,lo-span*.25):0,upper=hi+span*.25;
    const raw=(upper-lower)/4,power=10**Math.floor(Math.log10(raw));
    const unit=[1,2,2.5,5,10].find(value=>value*power>=raw)*power;
    return {min:Math.floor(lower/unit)*unit,max:Math.ceil(upper/unit)*unit,step:unit};
  }
  function renderOverviewPlot() {
    if(state.view!=='overview')return;
    const container=$('#tradeoff-plot'),width=Math.max(280,Math.round(container.clientWidth)||640),height=280;
    const m=metric(),rows=overviewRows(),valid=rows.filter(row=>finite(row.quality.value)&&finite(row.resource.value));
    const xRange=axisRange(valid.map(row=>row.resource.value),{focused:state.zoom});
    const yRange=axisRange(valid.map(row=>row.quality.value),{percent:true,focused:state.zoom});
    const left=47,right=width-24,top=20,bottom=222;
    const x=value=>left+(value-xRange.min)/(xRange.max-xRange.min)*(right-left);
    const y=value=>bottom-(value-yRange.min)/(yRange.max-yRange.min)*(bottom-top);
    const xDigits=Math.max(0,Math.min(4,-Math.floor(Math.log10(xRange.step))+(xRange.step/(10**Math.floor(Math.log10(xRange.step)))===2.5?1:0)));
    const yDigits=yRange.step%1?2:0;
    let axes='';
    for(let value=yRange.min;value<=yRange.max+yRange.step/100;value+=yRange.step)axes+=`<path d="M${left} ${y(value)}H${right}" class="plot-grid"/><text x="${left-10}" y="${y(value)+4}" text-anchor="end">${number(value,yDigits)}%</text>`;
    for(let value=xRange.min;value<=xRange.max+xRange.step/100;value+=xRange.step)axes+=`<text x="${x(value)}" y="${bottom+24}" text-anchor="middle">${number(value,xDigits)}</text>`;
    const baseline=valid.find(row=>row.key==='pristine');
    const baselineLines=baseline?`<path d="M${x(baseline.resource.value)} ${top}V${bottom}M${left} ${y(baseline.quality.value)}H${right}" class="plot-reference"/>`:'';
    const groups=new Map();
    valid.forEach(row=>{const identity=`${row.resource.value}|${row.quality.value}`;if(!groups.has(identity))groups.set(identity,[]);groups.get(identity).push(row);});
    // Labels are controls, not observations: only the small SVG dots encode data.
    // Place 44px targets independently so equal/nearby scores stay selectable.
    const points=[...groups.values()].map(group=>({group,ax:x(group[0].resource.value),ay:y(group[0].quality.value)}));
    const placed=[],clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    points.forEach(point=>{
      const candidates=[],minX=left+22,maxX=right-22,minY=top+22,maxY=bottom-22;
      const xs=new Set([clamp(point.ax,minX,maxX),minX,maxX]),ys=new Set([clamp(point.ay-32,minY,maxY),clamp(point.ay+32,minY,maxY),minY,maxY]);
      for(let px=minX;px<=maxX;px+=8)xs.add(px);
      for(let py=minY;py<=maxY;py+=8)ys.add(py);
      xs.forEach(px=>ys.forEach(py=>{
        if(placed.some(other=>Math.abs(px-other.lx)<46&&Math.abs(py-other.ly)<46))return;
        const distance=(px-point.ax)**2+(py-point.ay)**2;
        const obscured=points.filter(other=>Math.abs(px-other.ax)<22&&Math.abs(py-other.ay)<18).length;
        candidates.push({lx:px,ly:py,score:distance+obscured*100000+(py>point.ay?10:0)});
      }));
      candidates.sort((a,b)=>a.score-b.score);
      const position=candidates[0]||{lx:clamp(point.ax,minX,maxX),ly:clamp(point.ay,minY,maxY)};
      Object.assign(point,position);placed.push(point);
    });
    let pointLeaders='',pointDots='';
    const marks=points.map(({group,ax,ay,lx,ly})=>{
      const selected=group.some(row=>row.key===state.model),row=group.find(row=>row.key===state.model)||group[0],names=group.map(row=>labels[row.key]).join(', ');
      const description=`${names}. ${number(row.quality.value,2)} percent accuracy; ${number(row.resource.value,m.digits)} ${m.unit}.${group.length>1?' Coincident observations; activate to cycle through versions.':''}`;
      pointLeaders+=`<path d="M${ax} ${ay}L${lx} ${ly}" class="plot-leader${selected?' selected':''}"/>`;
      pointDots+=`<circle cx="${ax}" cy="${ay}" r="${selected?5:4}" class="plot-observation${selected?' selected':''}" data-observation-models="${group.map(row=>row.key).join('|')}"/>`;
      return `<button type="button" class="plot-point${selected?' selected':''}" data-point-models="${group.map(row=>row.key).join('|')}" data-model="${row.key}" aria-pressed="${selected}" aria-label="${escape(description)}" title="${escape(description)}" style="left:${lx/width*100}%;top:${ly}px"><span class="plot-mark"><span>${group.length>1?group.length:pointLabels[row.key]}</span></span></button>`;
    }).join('');
    const title=state.qualityMetric==='prompt_strict'?'Strict prompt accuracy':'Loose prompt accuracy';
    $('#tradeoff-y-label').textContent=`${title} ↑`;
    container.innerHTML=`<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)} against ${escape(m.label)}. ${valid.length} measured versions. ${state.zoom?'Axes focus on observed values.':'Axes begin at zero.'}">${axes}<path d="M${left} ${bottom}H${right}" class="plot-baseline"/>${baselineLines}${pointLeaders}${pointDots}<text x="${(left+right)/2}" y="${height-7}" text-anchor="middle" class="plot-axis-title">${escape(m.label)} · ${escape(m.unit)}</text>${valid.length?'':`<text x="${(left+right)/2}" y="125" text-anchor="middle">No complete paired observations</text>`}</svg>${marks}`;
    const missing=rows.filter(row=>!finite(row.quality.value)||!finite(row.resource.value));
    $('#tradeoff-caption').textContent=`Select a label or table row. Small dots are measured values; label connectors prevent overlap. P = pristine; A = Astra round. ${state.zoom?'Axes focus on the observed range.':'Axes begin at zero.'} Dashed guides mark pristine. ${[...groups.values()].some(group=>group.length>1)?'Numbered marks contain coincident observations; activate to cycle. ':''}${missing.length?`Not plotted: ${missing.map(row=>shortLabels[row.key]).join(', ')} (missing complete quality or resource observations).`:'Every version has a plotted observation.'}`;
  }
  function exportOverview() {
    const columns=[['artifact',row=>row.key],['cohort',()=>cohortLabel()],['strict_passes',row=>row.strict.count??''],['loose_passes',row=>row.loose.count??''],['quality_denominator',row=>row.strict.n||''],['quality_output_tokens',row=>row.tokens.value??''],['resource_sessions_complete',row=>row.resource.complete]];
    resourceMetrics.forEach(m=>columns.push([`${m.key} (${m.unit})`,row=>resourceReading(row.key,m).value??'']));
    const rows=[columns.map(([title])=>title),...overviewRows().map(row=>columns.map(([,get])=>get(row)))];
    downloadCSV(rows,`weight-space-lab-all-versions-${state.cohort}.csv`);announce('All-version quality and resource comparison exported.');
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
    else if(state.cohort==='fresh')note='Fresh-run development evidence on the same 32 exposed prompts. A current-browser baseline controls the historical browser-version mismatch, but this is not new held-out confirmation.';
    else note='Development-only evidence. The saved baseline and V1 used Chrome 156; new Astra candidates used Chrome 157. Repeated prompt exposure and the browser change limit causal and generalization claims.';
    $('#scope-note').innerHTML=`${escape(note)} <a href="#methodology">Read methods</a>`;
    renderSelected(readings.find(r=>r.key===state.model),readings[0]);
  }
  function freshPairedOutcomeHTML() {
    if(state.view!=='quality'||state.cohort!=='fresh'||state.model==='pristine'||!['prompt_strict','prompt_loose'].includes(state.metric))return '';
    const reference=freshQuality('pristine')?.item_outcomes,candidate=freshQuality(state.model)?.item_outcomes;
    if(!Array.isArray(reference)||!reference.length||!Array.isArray(candidate)||!candidate.length)return '';
    const indexed=rows=>{const result=new Map();for(const row of rows){if(typeof row.item_id!=='string'||!row.item_id)return null;result.set(row.item_id,result.has(row.item_id)?null:row);}return result;};
    const a=indexed(reference),b=indexed(candidate);if(!a||!b)return '';
    const ids=new Set([...a.keys(),...b.keys()]),field=state.metric==='prompt_loose'?'loose_pass':'strict_pass';
    const counts={improved:0,regressed:0,tied:0,unresolved:0};
    const resolved=row=>['ok','truncated'].includes(row?.status)&&row.grading_status==='graded'&&typeof row[field]==='boolean';
    for(const id of ids){const before=a.get(id),after=b.get(id);if(!resolved(before)||!resolved(after))counts.unresolved++;else if(before[field]===after[field])counts.tied++;else if(after[field])counts.improved++;else counts.regressed++;}
    return `<div class="quality-reference"><p>Paired ${state.metric==='prompt_loose'?'loose':'strict'} outcomes vs fresh pristine</p><strong>${counts.improved} improved · ${counts.regressed} regressed<br>${counts.tied} tied · ${counts.unresolved} unresolved</strong><p>Aligned by ${ids.size} item IDs. Ties include unchanged passes and unchanged failures. Descriptive results on the exposed development set.</p></div>`;
  }
  function renderSelected(r,baseline) {
    const m=metric();let delta='Reference artifact',signal='neutral';
    if(state.model!=='pristine') {
      if((state.view==='resources' && resource()?.matched_environment!==true)||(state.view==='quality'&&state.cohort==='fresh'&&fresh()?.quality?.matched_environment!==true)) {
        delta=resource()?.status==='in_progress'?'Comparison pending: full-schedule environment verification is pending':'Comparison unavailable: matched environment not established';
      } else if(finite(r.value)&&finite(baseline.value)) {
        const diff=r.value-baseline.value;
        delta=`<strong>${sign(diff,m.digits)} ${state.view==='quality'&&state.metric!=='tokens'?'percentage points':escape(m.unit)}</strong> vs pristine`;
        if(diff!==0&&m.direction!=='context')signal=(m.direction==='higher'?diff>0:diff<0)?'positive':'negative';
      } else delta='No matched observation to compare';
    }
    let context=state.view==='quality'?(r.count!==undefined?`${number(r.count)} of ${r.n} ${state.metric.startsWith('instruction')?'instructions':'prompts'} passed.`:r.n?`${r.n} scored prompts. Sampled output tokens.`:state.cohort==='fresh'?'No completed quality measurement is available for this artifact in the fresh comparison.':'This artifact has no historical held-out evaluation.'):`${r.complete||0} of 2 sessions completed. ${r.n||0} available readings for this metric.`;
    if(state.view==='resources'&&r.coverage?.length)context+=` Active-time coverage: ${r.coverage.map(v=>number(v*100,1)+'%').join(' / ')}.`;
    if(state.view==='resources'&&r.values?.length)context+=` Individual readings: ${r.values.map(v=>number(v,m.digits)).join(' / ')} ${m.unit}.`;
    if(state.view==='resources'&&activityDurationContext(state.model))context+=` ${activityDurationContext(state.model)}`;
    let qualityReference='';
    if(state.view==='resources') {
      const historical=state.cohort==='historical'?(state.model==='pristine'?data.v1.development.baseline:state.model==='v1'?data.v1.development.candidate:null):null;
      const candidate=state.cohort==='fresh'?freshQuality(state.model)?.metrics:rounds.find(row=>state.model===`v2-r${row.round}`)?.metrics;
      const passes=historical?.strict??candidate?.strict_passes;
      const denominator=historical?.total??candidate?.denominator;
      if(passes!==undefined)qualityReference=`<div class="quality-reference"><p>${state.cohort==='fresh'?'Fresh development quality':'Historical development quality'}</p><strong>${passes} / ${denominator} strict passes</strong><p>${state.cohort==='fresh'?'Same exposed development set; no new held-out confirmation.':'Separate quality study. Chrome 156 baseline; Chrome 157 V2 candidates.'} <a href="#methodology">Scope &amp; caveats</a></p></div>`;
    }
    let interval='';
    if(state.view==='quality'&&state.scope==='heldout'&&state.model==='v1'&&state.metric!=='tokens') {
      const result=data.v1.heldout.metrics[state.metric], [low,high]=result.ci95_percentage_points, extent=Math.max(Math.abs(low),Math.abs(high),Math.abs(result.difference_percentage_points))*1.2;
      const at=x=>(x+extent)/(2*extent)*100;
      interval=`<div class="interval"><p class="interval-title">Paired 95% interval · change vs pristine</p><div class="interval-track" aria-hidden="true"><span class="interval-zero"></span><span class="interval-range" style="left:${at(low)}%;width:${at(high)-at(low)}%"></span><span class="interval-dot" style="left:${at(result.difference_percentage_points)}%"></span></div><div class="interval-values"><span>${sign(low,2)} pp</span><span>0</span><span>${sign(high,2)} pp</span></div><p class="interval-caption">Includes zero; improvement is unestablished.</p></div>`;
    }
    $('#selected-detail').innerHTML=`<p class="selected-label">Selected artifact</p><h3 class="selected-name">${labels[state.model]}</h3>${finite(r.value)?`<p class="selected-value">${number(r.value,m.digits)}<small>${escape(m.unit)}</small></p>`:'<p class="detail-empty">No measurement available</p>'}<p class="selected-delta" data-signal="${signal}">${delta}</p><p class="detail-context">${escape(context)}</p>${interval}${qualityReference}${freshPairedOutcomeHTML()}`;
  }
  function renderTable() {
    $('#table-scope').textContent=state.view==='resources'?'Two-session means':state.cohort==='fresh'?'Fresh development':state.scope==='heldout'?'Historical V1':'Historical development';
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
    const current=freshArtifacts().find(artifact=>artifactKey(artifact)===key);
    if(current?.weight_operations?.length)return {operations:current.weight_operations,changed_source_entries:current.source_edit?.changed_stored_entries??current.changed_source_entries,artifact_sha256:current.artifact_sha256??current.deployed_artifact_sha256};
    if(methods.edit_operations?.[key])return methods.edit_operations[key];
    if(key==='pristine')return {operations:[],changed_source_entries:0,artifact_sha256:data.v1.heldout.arms.baseline.deployed_artifact_sha256};
    if(key==='v1')return {operations:data.v1.edit.layers.map(layer=>({tensor:`model.language_model.layers.${layer}.mlp.down_proj.weight`,columns_inclusive:data.v1.edit.columns,factor:data.v1.edit.scale})),changed_source_entries:data.v1.edit.changed_bf16_entries,artifact_sha256:data.v1.heldout.arms.candidate.deployed_artifact_sha256};
    const r=rounds.find(r=>key===`v2-r${r.round}`);return {operations:r?.weight_operations||[],changed_source_entries:r?.source_edit?.changed_stored_entries,artifact_sha256:r?.artifact_sha256};
  }
  function renderEdit() {
    const info=editInfo(state.model), operations=info.operations||[];
    $('#edit-version').textContent=shortLabels[state.model];
    if(!operations.length&&state.model!=='pristine'){$('#edit-detail').innerHTML='<p class="edit-description">Verified edit operations are not available for this artifact yet.</p><a class="provenance-link" href="#evidence">Inspect artifact identities ↗</a>';return;}
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
  function detailValue(value) {if(value===null||value===undefined)return 'Unavailable';if(typeof value==='boolean')return value?'Yes':'No';if(Array.isArray(value))return value.map(detailValue).join(', ');if(typeof value==='object')return detailList(value);return escape(typeof value==='number'?number(value,Number.isInteger(value)?0:2):String(value));}
  function detailList(object){return `<dl>${Object.entries(object||{}).map(([k,v])=>`<div><dt>${escape(pretty(k))}</dt><dd>${detailValue(v)}</dd></div>`).join('')}</dl>`;}
  function continuationMethodsHTML() {
    const continuation=data.resource_study?.continuation;if(!continuation)return '';
    const recovery=continuation.recovery||{};
    const facts={'Preserved completed positions':(continuation.inherited_completed_positions||[]).join(', '),'Continuation positions':(continuation.remaining_positions||[]).join(', '),'Failure stage':continuation.failure_stage,'Failure type':continuation.failure_type,'Recovered file role':recovery.file_role,'Recovered bytes':number(recovery.size_bytes),'Recovered SHA-256':recovery.sha256,'Recovery method':recovery.method,'Gap after position 6':finite(continuation.inter_session_gap_after_position6_seconds)?number(continuation.inter_session_gap_after_position6_seconds,2)+' seconds':'Not yet recorded','Original calendar and total limits':continuation.original_calendar_and_total_limits_unchanged===true?'Unchanged':'Not established','Replaced completed sessions':continuation.replaced_completed_sessions,'Model dispatch retries':continuation.model_dispatch_retries};
    return `<details><summary>Documented resource continuation</summary><div class="method-body"><p>${escape(continuation.interruption_note)}</p>${detailList(facts)}<p>Hardware and runtime matching does not establish constant background host activity or uninterrupted conditions. Both R3 observations remain in the descriptive comparison.</p></div></details>`;
  }
  function resourceEnvironmentHTML() {
    const report=data.resource_study;if(!report?.environment)return '';
    const environment=report.environment,hardware=environment.hardware||{},bytes=Number(hardware['hw.memsize']);
    const values={'Processor':hardware['machdep.cpu.brand_string']??'Unavailable','Logical CPUs':hardware['hw.logicalcpu']??'Unavailable','Physical / unified memory':finite(bytes)&&bytes>0?`${number(bytes/2**30,bytes%(2**30)?2:0)} GiB (${number(bytes)} bytes)`:'Unavailable','Hardware model':hardware['hw.model']??'Unavailable','macOS':environment.macos??'Unavailable','Chrome':environment.browser_version??'Unavailable','Runtime':environment.runtime??'Unavailable','Quantization':environment.quantization??'Unavailable'};
    const verification=report.matched_environment===true?'Matched hardware and runtime verified across the completed schedule.':report.status==='in_progress'?'Hardware and runtime matching remains pending until the full schedule completes.':'Hardware and runtime matching was not established for the full schedule.';
    return `<details><summary>Resource environment</summary><div class="method-body"><p>${escape(verification)}</p>${detailList(values)}<p>Memory is the operating system’s reported physical / unified capacity, converted from bytes to GiB. It is not a measurement of model memory or physical GPU residency.</p></div></details>`;
  }
  function renderMethods() {
    const studies=(methods.studies||[]).map(study=>`<details><summary>${escape(study.title)}</summary><div class="method-body"><p>${escape(study.question)}</p><p><strong>Scope.</strong> ${escape(study.scope)}</p><p><strong>Selection.</strong> ${escape(study.selection_rule)}</p>${study.aggregation?`<p><strong>Aggregation.</strong> ${escape(study.aggregation)}</p>`:''}${study.public_workload_prompt?`<p><strong>Public resource workload prompt.</strong> “${escape(study.public_workload_prompt)}”</p>`:''}${detailList(study.generation_settings)}<ul>${(study.limitations||[]).map(s=>`<li>${escape(s)}</li>`).join('')}</ul><p>${(study.sources||[]).map(s=>`<a href="${escape(methodSource(s.url))}">${escape(s.label)}</a>`).join(' · ')}</p></div></details>`).join('');
    const telemetry=Object.values(methods.telemetry_definitions||{}).map(def=>`<div><dt>${escape(def.label)}${def.unit?' · '+escape(def.unit):''}</dt><dd>${escape(def.definition)}${def.scope?' '+escape(def.scope):''}${def.coverage?' Coverage: '+escape(def.coverage):''}${def.limitations?' '+escape(Array.isArray(def.limitations)?def.limitations.join(' '):def.limitations):''}</dd></div>`).join('');
    const modelDetails=`<details><summary>Model and exact interventions</summary><div class="method-body"><p>Every artifact derives from the same pinned Qwen3.5-0.8B source checkpoint. Weight values change; architecture, tokenizer, tensor shapes and compiled graph stay fixed. The browser deployment is the quantized text backbone, excluding vision and MTP.</p>${detailList(methods.model)}<p>Historical V1 scales columns 0–59 in layers 15 and 19 by 0.75. Astra rounds 1–3 independently scale columns 2014–2073 in layer 15 and 3524–3583 in layer 19 by 0.75, 1.25 and 1.50. Each targets 122,880 source coordinates; observed stored-value changes are recorded in the artifact audit. The selected version’s exact operations and deployed identity are available in the Results and Evidence views.</p><p>Source arithmetic rounds directly to BF16 with round-to-nearest/ties-to-even. Quantization uses 32-column groups; a changed group scale can affect decoded values beyond the edited source columns. No smaller model or sparse execution kernel is introduced.</p></div></details>`;
    const editorDetails=`<details><summary>Astra and development feedback</summary><div class="method-body"><p>The historical V1 proposal used a manual returned-JSON handoff. Astra/Ultra was requested, but provider identity and effort were not independently observed. V2 used three actual fresh-context gpt-6-astra collaboration agents. Ultra effort and immutable provider versions are unverified; technical editor isolation is also unverified.</p><p>The development packet supplied approved weight information, the 32 development tasks, pristine outputs and grades, the response schema and the benchmark objective. Later rounds included previous candidate feedback. The sequence is one adaptive search trajectory, not independent editor replications. V2 round 3 used a second delivery turn for the same packet after truncation, adding no new scientific feedback.</p><p>Provider token usage and billing are unavailable. The fresh-baseline round has a separate recorded editor identity and scope below. Evaluation costs on the target model and frontier editor costs are different quantities.</p></div></details>`;
    const benchmarkDetails=`<details><summary>Benchmark selection and exposure</summary><div class="method-body">${Object.entries(methods.benchmarks||{}).map(([key,value])=>`<h3>${escape(pretty(key))}</h3>${detailList(value)}`).join('')}<p>The published item IDs define the selected schedules. Private exclusions mean rerunning a selection heuristic may produce a different subset. Disjoint IDs and normalized overlap checks do not establish absence of paraphrase overlap or pretraining exposure. Fresh baseline execution does not make previously supplied prompts untouched.</p></div></details>`;
    $('#methodology-detail').innerHTML=modelDetails+editorDetails+benchmarkDetails+studies+continuationMethodsHTML()+resourceEnvironmentHTML()+`<details><summary>Scoring and uncertainty</summary><div class="method-body">${detailList(methods.scoring)}</div></details><details><summary>Telemetry definitions and coverage</summary><div class="method-body"><dl>${telemetry}</dl></div></details><details><summary>Reproduction and preserved failures</summary><div class="method-body"><ul>${[...(methods.reproduction||[]),...(methods.preserved_failures||[])].map(s=>`<li>${escape(s)}</li>`).join('')}</ul></div></details>`;
    if(data.resource_study) {
      const count=(data.resource_study.sessions||[]).filter(completed).length;
      $('#resource-method-state').textContent=pretty(data.resource_study.status||'reported');
      $('#resource-method-status').textContent=`${count} of 10 planned sessions are complete in the public report. Quality eligibility was not required. Two sessions per artifact form a descriptive pilot, not an equivalence test.`;
    }
  }
  function renderSources() {
    const sources=[['Historical V1 quality','Development and held-out findings, read here','#evidence?record=v1'],['Astra rounds 1–3','Scores, selection and historical limitations','#evidence?record=v2'],['Original resource pilot','All five artifacts and the retained interruption','#evidence?record=resources'],['Experimental methodology','Protocols, scoring and counter definitions on this site','#methodology']];
    if(fresh())sources.unshift(['Fresh baseline and Astra 4','New matched-cohort observations and recorded status','#evidence?record=fresh']);
    $('#source-list').innerHTML=sources.map(([title,description,url])=>`<a href="${escape(url)}"><span>${escape(title)}</span><small>${escape(description)}</small><span aria-hidden="true">→</span></a>`).join('');
    const strict=data.v1.heldout.metrics.prompt_strict,interval=strict.ci95_percentage_points;
    const v1=`<details id="record-v1"><summary>V1 quality record</summary><div class="method-body"><p>The historical V1 edit scaled columns 0–59 in layers 15 and 19 by 0.75. On the 32 development prompts, strict passes changed from ${data.v1.development.baseline.strict} to ${data.v1.development.candidate.strict}. The candidate was then frozen for a 128-prompt held-out comparison.</p><p>Strict held-out passes were ${strict.baseline.numerator}/${strict.baseline.denominator} for pristine and ${strict.candidate.numerator}/${strict.candidate.denominator} for the edit: ${sign(strict.difference_percentage_points,2)} percentage points, with a paired 95% interval of ${sign(interval[0],2)} to ${sign(interval[1],2)} points. The interval includes zero; improvement and equivalence remain unestablished.</p><p>The candidate stopped after 94 responses and separately continued for 34. Both portions remain included. Historical timing does not establish a controlled resource gain. These final prompts are now exposed and cannot be reused as untouched confirmation.</p><p><a href="#results?view=quality&cohort=historical&scope=heldout&metric=prompt_strict&model=v1">Explore the paired quality result</a> · <a href="${escape(data.sources.v1_data)}">Download original V1 JSON</a></p></div></details>`;
    const v2=`<details id="record-v2"><summary>Astra rounds 1–3 record</summary><div class="method-body"><p>Three fresh-context gpt-6-astra agents proposed independent edits to the pristine checkpoint using development feedback. Each targeted 122,880 source coordinates. Selection required more than 16 strict passes out of 32, at most 13,941 scored output tokens and complete technical validation.</p><table class="data-table"><thead><tr><th>Version</th><th>Strict</th><th>Loose</th><th>Quality tokens</th></tr></thead><tbody>${rounds.map(row=>`<tr><th scope="row">Astra ${row.round}</th><td data-label="Strict">${row.metrics.strict_passes} / ${row.metrics.denominator}</td><td data-label="Loose">${row.metrics.loose_passes} / ${row.metrics.denominator}</td><td data-label="Quality tokens">${number(row.metrics.scored_output_tokens)}</td></tr>`).join('')}</tbody></table><p>No edit qualified, so pristine was retained. Fewer output tokens alone do not establish less browser memory, computation or energy. The saved quality baseline ran Chrome 156; the candidates ran Chrome 157. The comparison is development evidence with a browser-version confound, without new held-out confirmation.</p><p>The three agents formed one adaptive search trajectory. Round 3 needed one extra packet-delivery turn; provider token usage, immutable model version and Ultra effort are unverified. Failed preparation and offline recovery remain in the original accounting, with no recovery generation added.</p><p><a href="#results?view=overview&cohort=historical&metric=response">Compare historical versions</a> · <a href="${escape(methodSource(data.sources.v2_execution))}">Download original V2 JSON</a></p></div></details>`;
    const report=data.resource_study,completedSessions=(report?.sessions||[]).filter(completed),gap=report?.continuation?.inter_session_gap_after_position6_seconds;
    const resources=`<details id="record-resources"><summary>Original all-edit resource record</summary><div class="method-body"><p>The separately authorized resource pilot compared pristine, V1 and all three V2 edits without requiring quality qualification. ${completedSessions.length} sessions completed in mirrored order; each contained 60 measured requests and one warm-up, using the public autumn-leaves prompt.</p><p>The execution retained the original six sessions and continued only the four undispatched positions after a ${finite(gap)?number(gap/60,2):'documented'}-minute gap. The slower second Astra 3 observation remains included. Matching recorded hardware and runtime does not establish uninterrupted or constant host conditions.</p><p>Every measured session produced 2,340 tokens. Sampled GPUBuffer descriptor and WASM maxima matched, but these counters do not measure physical VRAM. Native footprint and response time varied. Power was unavailable. Lower CPU/GPU activity rates with longer response times do not establish lower total computation or energy.</p><p>Two sessions on one device form a descriptive pilot, without significance, equivalence or general efficiency claims. Per-session results, request traces and all measurement definitions are available within this site.</p><p><a href="#results?view=resources&cohort=historical&metric=response">Explore every resource session</a> · <a href="resource-study.json" download>Download original resource JSON</a></p></div></details>`;
    const current=fresh()?`<details id="record-fresh"><summary>Fresh baseline and Astra 4 record</summary><div class="method-body"><p>The new comparison is a separate experiment with a newly measured pristine baseline. It evaluates the four Astra artifacts and the V1 edit against that baseline. Historical observations are preserved separately. Current report status: <strong>${escape(pretty(fresh().status||'recorded'))}</strong>.</p><p>Every fresh comparison reading comes from the new report. A missing result is not filled from an older run. The same exposed development prompts are used for quality; fresh execution does not make them a new held-out test.</p><p><a href="#results?view=overview&cohort=fresh&metric=response">Explore the fresh comparison</a> · <a href="#methodology">Read the full experimental method</a> · <a href="fresh-comparison.json" download>Download fresh comparison JSON</a></p></div></details>`:'';
    $('#study-records').innerHTML=current+v1+v2+resources;
    $('#identity-table').innerHTML=`<table class="data-table hash-table"><thead><tr><th scope="col">Artifact</th><th scope="col">Deployed artifact SHA-256</th></tr></thead><tbody>${allKeys.map(key=>`<tr><th scope="row">${labels[key]}</th><td data-label="Deployed artifact SHA-256" class="hash">${escape(editInfo(key).artifact_sha256||'Unavailable')}</td></tr>`).join('')}</tbody></table>`;
  }
  function focusRecord() {
    if(state.route!=='evidence'||!['v1','v2','resources','fresh'].includes(state.record))return;
    const record=$(`#record-${state.record}`);if(!record)return;record.open=true;record.querySelector('summary').focus({preventScroll:true});record.scrollIntoView({block:'start'});
  }
  function render() {syncCohort();renderNavigation();renderPickers();renderResourceContext();renderOverview();if(state.view!=='overview'){renderChart();renderTable();renderEdit();}renderSessions();}
  function exportCSV() {
    if(state.view==='overview'){exportOverview();return;}
    const m=metric(),rows=[['artifact','study','metric','value','unit','sample_count','individual_session_values']];
    keys.forEach(key=>{const r=reading(key);rows.push([key,scopeLabel(),m.label,r.value??'',m.unit,r.n,(r.values||[]).join(';')]);});
    downloadCSV(rows,`weight-space-lab-${state.view}-${state.metric}.csv`);announce('Current comparison exported as CSV.');
  }
  function downloadCSV(rows,filename) {
    const csv=rows.map(row=>row.map(cell=>'"'+String(cell).replaceAll('"','""')+'"').join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
  }
  $('.skip-link').addEventListener('click',event=>{event.preventDefault();const target=$(`#${state.route}-heading`)||$('#main');target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start'});});
  document.addEventListener('click',event=>{const point=event.target.closest('[data-point-models]');const button=event.target.closest('[data-model]');if(point){const models=point.dataset.pointModels.split('|'),index=models.indexOf(state.model);selectArtifact(models[(index+1)%models.length]);}else if(button)selectArtifact(button.dataset.model);if(event.target.closest('#inspect-version')){state.view='resources';render();saveHash();$('#chart-title').setAttribute('tabindex','-1');$('#chart-title').focus({preventScroll:true});}if(event.target.closest('#export-requests'))exportRequests();const step=event.target.closest('#previous-request')?-1:event.target.closest('#next-request')?1:0;if(step){state.request=Math.max(1,Math.min(observedRequests(activeSession()).length,state.request+step));renderRequestTrace();saveHash();}});
  document.addEventListener('change',event=>{if(event.target.id==='resource-session'){state.session=Number(event.target.value);state.request=1;renderRequestExplorer();$('#resource-session')?.focus({preventScroll:true});saveHash();}if(event.target.id==='request-metric'){state.trace=event.target.value;renderRequestTrace();saveHash();}});
  document.addEventListener('input',event=>{if(event.target.id==='request-index'){state.request=Number(event.target.value);renderRequestTrace();saveHash();}});
  $$('[data-view]').forEach(button=>button.addEventListener('click',()=>{state.view=button.dataset.view;state.metric=state.view==='quality'?'prompt_strict':'response';render();saveHash();announce(`${state.view==='overview'?'All-version':state.view==='resources'?'Resource':'Quality'} measurements selected.`);}));
  $('#cohort').addEventListener('change',event=>{state.cohort=event.target.value;state.session=null;state.request=1;if(state.cohort==='fresh')state.scope='development';if(!metrics().some(m=>m.key===state.metric))state.metric=metrics()[0].key;render();saveHash();announce(`${cohortLabel()} selected. Historical measurements are not used to fill missing fresh observations.`);});
  $('#overview-quality').addEventListener('change',event=>{state.qualityMetric=event.target.value;renderOverview();saveHash();announce('Quality axis updated.');});
  $('#overview-resource').addEventListener('change',event=>{state.metric=event.target.value;renderOverview();saveHash();announce(`${metric().label} selected for the resource axis.`);});
  $('#overview-zoom').addEventListener('change',event=>{state.zoom=event.target.checked;renderOverviewPlot();saveHash();announce(state.zoom?'Axes focus on the observed range.':'Axes begin at zero.');});
  $('#scope').addEventListener('change',event=>{state.scope=event.target.value;if(!metrics().some(m=>m.key===state.metric))state.metric='prompt_strict';render();saveHash();announce(scopeLabel());});
  $('#metric').addEventListener('change',event=>{state.metric=event.target.value;render();saveHash();announce(`${metric().label}: ${formatReading(reading(state.model))} for ${labels[state.model]}.`);});
  $('#mobile-model').addEventListener('change',event=>selectArtifact(event.target.value));
  $('#export-csv').addEventListener('click',exportCSV);
  $('#copy-link').addEventListener('click',async()=>{saveHash();try{await navigator.clipboard.writeText(location.href);$('#copy-link').textContent='Link copied';announce('View link copied.');setTimeout(()=>$('#copy-link').textContent='Copy view link',1800);}catch{$('#copy-link').textContent='Use the address bar';announce('Copy this view from the browser address bar.');}});
  $$('.print-button').forEach(button=>button.addEventListener('click',()=>window.print()));
  window.addEventListener('hashchange',()=>{const prior=state.route;readHash();render();if(prior!==state.route){window.scrollTo(0,0);const heading=$(`#${state.route}-heading`);heading.setAttribute('tabindex','-1');heading.focus({preventScroll:true});announce(`${pretty(state.route)} view.`);}focusRecord();});
  let printDetails=[];
  window.addEventListener('beforeprint',()=>{printDetails=$$('.methods-details details').map(el=>[el,el.open]);printDetails.forEach(([el])=>el.open=true);});
  window.addEventListener('afterprint',()=>{printDetails.forEach(([el,open])=>el.open=open);printDetails=[];});
  if(!location.hash && ['complete','completed'].includes(fresh()?.status)) {state.cohort='fresh';state.view='overview';state.metric='response';state.model=allKeys.includes('astra-r4')?'astra-r4':'pristine';}
  readHash();renderMethods();renderSources();render();
  $('.interactive-workspace').hidden=false;document.documentElement.classList.add('js-ready');renderOverviewPlot();focusRecord();
  if(typeof ResizeObserver==='function'){let frame;new ResizeObserver(()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(renderOverviewPlot);}).observe($('#tradeoff-plot'));}
  $('#snapshot-date').textContent=new Date(data.updated+'T12:00:00').toLocaleDateString('en-GB',{day:'numeric',month:'long',year:'numeric'});
})();
