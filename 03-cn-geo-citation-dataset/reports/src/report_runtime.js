(function(){
  'use strict';

  const C = {
    brand:'#1b365d', brand2:'#2d5a8a', pale:'#eef2f7', mid:'#d6e1ee',
    ink:'#141413', warm:'#3d3d3a', olive:'#504e49', stone:'#6b6a64',
    border:'#e8e6dc', line:'#d8d5c8', warn:'#b46b45', ok:'#65745b', white:'#ffffff'
  };
  const palette = [C.brand,C.brand2,'#65745b','#8c6f56','#6e7d94','#9a806d','#7a8871','#6b6a64'];
  const heatColors = ['#f0f2f5','#d6e1ee','#8ba0b8','#4c7098','#1b365d'];
  const platformOrder = REPORT.platforms.map(d=>d.platform_code);
  const platformName = Object.fromEntries(REPORT.platforms.map(d=>[d.platform_code,d.platform_name_cn]));
  const fmt = new Intl.NumberFormat('zh-CN');
  const fmt1 = new Intl.NumberFormat('zh-CN',{maximumFractionDigits:1});
  const state = {scope:'dedup',platform:'ALL',sourceType:'ALL',dimension:'query_intent',label:'ALL',freshness:'ALL',preferenceProduct:'ALL',preferenceRankScope:'common',preferenceTopN:10,selectedPreferenceSource:null,platformScaleView:'count',sourceTypesView:'count',labelScaleView:'count'};
  const charts = new Map();
  const factories = new Map();
  const preferenceChartIds = ['chartSourcePreference','chartAnchorMigration','chartTerminalTilt','chartPreferenceTypes'];

  function n(v){ return fmt.format(Number(v||0)); }
  function compact(v){
    const x=Number(v||0); if(x>=1000000)return fmt1.format(x/1000000)+'M';
    if(x>=1000)return fmt1.format(x/1000)+'k'; return fmt.format(x);
  }
  function pct(v,digits=1){ return (Number(v||0)*100).toFixed(digits)+'%'; }
  function signedPct(v,digits=1){ const value=Number(v||0);return (value>0?'+':'')+(value*100).toFixed(digits)+'%'; }
  function metric(d){ return Number(state.scope==='raw'?d.raw_count:d.dedup_count)||0; }
  function metricLabel(){ return state.scope==='raw'?'原始记录':'精确去重'; }
  function esc(value){ return String(value??'').replace(/[&<>'"]/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[s])); }
  function short(value,max=22){ const s=String(value??''); return s.length>max?s.slice(0,max-1)+'…':s; }
  function byMetricDesc(a,b){ return metric(b)-metric(a); }
  function axisCategory(extra={}){ return Object.assign({axisLine:{lineStyle:{color:C.line}},axisTick:{show:false},axisLabel:{color:C.stone,fontSize:11,hideOverlap:true},splitLine:{show:false}},extra); }
  function axisValue(extra={}){ return Object.assign({axisLine:{show:false},axisTick:{show:false},axisLabel:{color:C.stone,fontSize:10},splitLine:{lineStyle:{color:C.border,type:'dashed'}}},extra); }
  function heatCells(data,max){const ceiling=Math.max(1,Number(max)||1);return data.map(cell=>({value:cell,label:{color:Number(cell[2]||0)>=ceiling*.42?'#fff':C.ink}}));}
  function baseOption(){ return {animationDuration:480,animationEasing:'cubicOut',backgroundColor:'transparent',color:palette,textStyle:{fontFamily:'"PingFang SC","Microsoft YaHei",system-ui,sans-serif',color:C.warm},aria:{enabled:true,decal:{show:false}},tooltip:{trigger:'item',confine:true,backgroundColor:'#fff',borderColor:C.line,borderWidth:1,textStyle:{color:C.ink,fontSize:12},extraCssText:'box-shadow:0 8px 24px rgba(20,20,19,.08);'}}; }
  function withBase(option){ return Object.assign(baseOption(),option); }
  function register(id,factory){ factories.set(id,factory); }
  function validatePreferenceLinkage(){
    const rankedPairs=new Set(REPORT.sourceTopRanks.filter(d=>d.rank<=20).map(d=>d.product_family+'|'+d.source_id));
    const tiltPairs=new Set(REPORT.terminalTilt.map(d=>d.product_family+'|'+d.source_id));
    const missing=[...rankedPairs].filter(key=>!tiltPairs.has(key));
    if(missing.length)throw new Error('Top 20 信源缺少终端倾向联动：'+missing.slice(0,3).join('、'));
    const endpointRows=new Map(REPORT.anchorSourceMigration.map(d=>[d.platform_code,d]));
    const endpointOrder=['DB','DOUBA','DP','DPA','TXYB','TXYBA','TYQW','TYQWA'];
    if(endpointOrder.some((code,index)=>!endpointRows.has(code)||Boolean(endpointRows.get(code).is_anchor_endpoint)!==(index<4)))throw new Error('锚点迁移端点分组不完整');
    if(REPORT.preferenceMeta.anchor_top20_carryover.length!==8)throw new Error('锚点承接摘要必须覆盖八个端点');
  }
  function selectPreferenceSource(sourceId,productFamily=null){
    if(!sourceId)return;
    if(productFamily&&state.preferenceProduct!=='ALL'&&state.preferenceProduct!==productFamily){state.preferenceProduct=productFamily;document.getElementById('preferenceProductFilter').value=productFamily;}
    state.selectedPreferenceSource=state.selectedPreferenceSource===sourceId?null:sourceId;
    document.getElementById('preferenceSourceSelect').value=state.selectedPreferenceSource||'';
    drawPreferenceViews();
  }
  function draw(id){
    const el=document.getElementById(id); if(!el)return;
    let chart=charts.get(id); if(!chart){ chart=echarts.init(el,null,{renderer:'canvas'});charts.set(id,chart); }
    el.setAttribute('aria-busy','true');
    try{
      chart.setOption(factories.get(id)(),true);
      if(id==='chartSourcePreference'||id==='chartAnchorMigration'){
        chart.off('click');
        chart.on('click',params=>{
          const sourceId=params.data&&params.data.sourceId;
          const detail=params.data&&params.data.detail;
          const share=detail&&(detail.weighted_share??detail.share);
          if(!sourceId||!detail||share<=0)return;
          selectPreferenceSource(sourceId,detail.product_family);
        });
      }
      el.setAttribute('aria-busy','false');
    }
    catch(error){ el.setAttribute('aria-busy','false'); el.textContent='图表加载失败：'+error.message; console.error(id,error); }
  }
  function drawPreferenceViews(){ preferenceChartIds.forEach(draw);updatePreferenceSummary();updateAnchorCarryover();updatePreferenceRankTable();updatePreferenceTable(); }
  function drawAll(){ factories.forEach((_,id)=>draw(id)); updateTables(); updateFindings(); updateSummary(); updatePreferenceSummary(); updateAnchorCarryover();updatePreferenceRankTable();updatePreferenceTable();updateTerminalRateGrid();updateMetricViewButtons(); }

  function initMeta(){
    const o=REPORT.overview;
    document.getElementById('kpiCitations').textContent=n(o.raw_citations);
    document.getElementById('kpiDedup').textContent=n(o.dedup_citations);
    document.getElementById('kpiQuestions').textContent=n(o.questions);
    document.getElementById('kpiPlatforms').textContent=n(o.platforms);
    document.getElementById('kpiSources').textContent=n(o.sources);
    document.getElementById('kpiPages').textContent=n(o.pages);
    document.getElementById('preferenceCommonQuestions').textContent=n(REPORT.preferenceMeta.balanced_question_count);
    document.getElementById('preferenceEndpointCount').textContent=n(REPORT.preferenceMeta.endpoint_count);
    document.getElementById('preferenceProductCount').textContent=n(REPORT.preferenceMeta.product_count);
    document.getElementById('preferenceQualifiedSources').textContent=n(REPORT.preferenceMeta.qualified_source_count);
    const coverage=REPORT.classificationCoverage.find(d=>d.platform_code==='ALL');
    document.getElementById('classificationCoverage').textContent=pct(coverage.classification_coverage);
    document.getElementById('classificationRuleShare').textContent=pct(coverage.rule_share);
    document.getElementById('classificationUnclassified').textContent=pct(coverage.unclassified_share);
    document.getElementById('classificationUnnormalized').textContent=pct(coverage.unnormalized_share);
    document.getElementById('rateDedupRetention').textContent=pct(o.dedup_citations/o.raw_citations);
    document.getElementById('rateClassificationCoverage').textContent=pct(coverage.classification_coverage);
    document.getElementById('rateCommonQuestionCoverage').textContent=pct(REPORT.preferenceMeta.common_scope_question_count/o.questions);
    document.getElementById('rateTop10Share').textContent=pct((REPORT.sourcePareto.find(d=>d.rank===10)||{}).cumulative_share||0);
    const carryoverByCode=new Map(REPORT.preferenceMeta.anchor_top20_carryover.map(d=>[d.platform_code,d.source_count/REPORT.preferenceMeta.anchor_pool_size]));
    const anchorRange=codes=>{const values=codes.map(code=>carryoverByCode.get(code)||0);return Math.min(...values)===Math.max(...values)?pct(values[0]):pct(Math.min(...values))+' 至 '+pct(Math.max(...values));};
    document.getElementById('rateDoubaoAnchor').textContent=anchorRange(['DB','DOUBA']);
    document.getElementById('rateDeepSeekAnchor').textContent=anchorRange(['DP','DPA']);
    document.getElementById('releaseDate').textContent=o.release_date;
    document.getElementById('generatedAt').textContent=new Date(REPORT.meta.generated_at).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
    const unclassified=REPORT.ecosystemUnclassifiedSummary;
    const ecosystemTotal=REPORT.ecosystem.reduce((sum,row)=>sum+row.dedup_count,0);
    const assignedEcosystems=REPORT.ecosystem.filter(row=>row.ecosystem!=='未归属生态');
    const assignedSources=assignedEcosystems.reduce((sum,row)=>sum+row.source_count,0);
    const unclassifiedShare=unclassified.dedup_count/ecosystemTotal;
    document.getElementById('ecosystemHeadline').textContent=n(assignedSources)+' 个已归属信源承载 '+pct(1-unclassifiedShare)+' 引用，未归属生态部分呈分散长尾';
    document.getElementById('ecosystemUnclassifiedShare').textContent=pct(unclassifiedShare);
    document.getElementById('ecosystemLowFrequencyShare').textContent=pct(unclassified.low_frequency_source_count/unclassified.source_count);
    document.getElementById('ecosystemTop50Share').textContent=pct(unclassified.top50_dedup_count/unclassified.dedup_count);
    document.getElementById('ecosystemNote').textContent='“未归属生态”仅描述集团或内容生态标签尚未标注。其 '+n(unclassified.source_count)+' 个域名贡献 '+pct(unclassifiedShare)+' 信源表去重引用，Top 50 占该部分引用的 '+pct(unclassified.top50_dedup_count/unclassified.dedup_count)+'。信源类型分类覆盖率与未分类长尾见 04.5。';
    const pipeline=[
      {value:o.raw_citations,label:'原始引用观察',delta:'保留采集事实'},
      {value:o.dedup_citations,label:'精确去重记录',delta:'移除 '+n(o.raw_citations-o.dedup_citations)+' 条额外重复'},
      {value:o.valid_urls,label:'有效 URL 记录',delta:pct(o.valid_urls/o.raw_citations)+' 可进入页面规范化'},
      {value:o.pages,label:'规范页面',delta:'按 canonical_url 聚合'}
    ];
    const root=document.getElementById('pipeline'); root.textContent='';
    pipeline.forEach(item=>{
      const div=document.createElement('div');div.className='pipe-step';
      const value=document.createElement('div');value.className='pipe-num';value.textContent=n(item.value);
      const label=document.createElement('div');label.className='pipe-label';label.textContent=item.label;
      const delta=document.createElement('div');delta.className='pipe-delta';delta.textContent=item.delta;
      div.append(value,label,delta);root.appendChild(div);
    });
    document.getElementById('titledPageCount').textContent=n(o.titled_pages)+' 个标题已提供页面';
    document.getElementById('dateKnownShare').textContent=pct(o.dated_pages/o.pages);
    document.getElementById('dateUnknownShare').textContent=pct(o.unknown_date_pages/o.pages);
    document.getElementById('dateConflictShare').textContent=pct(o.conflicting_date_pages/o.pages);
    const processingRoot=document.getElementById('processingStatus');processingRoot.textContent='';
    REPORT.processingStatus.forEach(item=>{
      const div=document.createElement('div');div.className='processing-item';
      const value=document.createElement('b');value.textContent=n(item.affected_records)+' · '+pct(item.ratio);
      const title=document.createElement('strong');title.textContent=item.item;
      const note=document.createElement('span');note.textContent='分母：'+item.denominator+'。'+item.handling;
      div.append(value,title,note);processingRoot.appendChild(div);
    });
    const dl=document.getElementById('dictionaryList'); dl.textContent='';
    REPORT.dictionary.forEach(item=>{const dt=document.createElement('dt');dt.textContent=item.table_name;const dd=document.createElement('dd');dd.textContent=item.table_name_cn+'：'+item.recommended_use_cn;dl.append(dt,dd);});
  }

  function addOptions(select,items,labelFn,valueFn){
    items.forEach(item=>{const op=document.createElement('option');op.value=valueFn(item);op.textContent=labelFn(item);select.appendChild(op);});
  }
  function initFilters(){
    const platform=document.getElementById('platformFilter');
    addOptions(platform,REPORT.platforms,d=>d.platform_name_cn,d=>d.platform_code);
    const sourceType=document.getElementById('sourceTypeFilter');
    const types=[...new Set(REPORT.sourceTypes.filter(d=>d.platform_code==='ALL').map(d=>d.source_type_cn))].sort((a,b)=>a.localeCompare(b,'zh-CN'));
    addOptions(sourceType,types,d=>d,d=>d);
    const dimension=document.getElementById('dimensionFilter');
    const available=[...new Set(REPORT.labelStats.map(d=>d.label_dimension))];
    available.sort((a,b)=>(REPORT.meta.dimension_names[a]||a).localeCompare(REPORT.meta.dimension_names[b]||b,'zh-CN'));
    addOptions(dimension,available,d=>REPORT.meta.dimension_names[d]||d,d=>d);
    if(!available.includes(state.dimension))state.dimension=available[0];
    dimension.value=state.dimension;
    const preferenceProduct=document.getElementById('preferenceProductFilter');
    addOptions(preferenceProduct,REPORT.terminalPairSummary,d=>d.product_family,d=>d.product_family);
    const preferenceSource=document.getElementById('preferenceSourceSelect');const matrixSources=[...new Map(REPORT.sourcePreference.map(d=>[d.source_id,d])).values()].sort((a,b)=>a.source_rank-b.source_rank);addOptions(preferenceSource,matrixSources,d=>preferenceSourceLabel(d,48),d=>d.source_id);
    updateLabelOptions();
    ['scopeFilter','platformFilter','sourceTypeFilter','dimensionFilter','labelFilter','freshnessFilter'].forEach(id=>{
      document.getElementById(id).addEventListener('change',()=>{
        state.scope=document.getElementById('scopeFilter').value;
        state.platform=document.getElementById('platformFilter').value;
        state.sourceType=document.getElementById('sourceTypeFilter').value;
        const nextDimension=document.getElementById('dimensionFilter').value;
        if(nextDimension!==state.dimension){state.dimension=nextDimension;state.label='ALL';updateLabelOptions();}
        else state.label=document.getElementById('labelFilter').value;
        state.freshness=document.getElementById('freshnessFilter').value;
        drawAll();
      });
    });
    document.getElementById('resetFilters').addEventListener('click',()=>{
      Object.assign(state,{scope:'dedup',platform:'ALL',sourceType:'ALL',dimension:'query_intent',label:'ALL',freshness:'ALL'});
      document.getElementById('scopeFilter').value=state.scope;document.getElementById('platformFilter').value=state.platform;
      document.getElementById('sourceTypeFilter').value=state.sourceType;document.getElementById('dimensionFilter').value=state.dimension;
      document.getElementById('freshnessFilter').value=state.freshness;updateLabelOptions();drawAll();
    });
    document.getElementById('preferenceProductFilter').addEventListener('change',event=>{
      state.preferenceProduct=event.target.value;
      state.selectedPreferenceSource=null;
      document.getElementById('preferenceSourceSelect').value='';
      drawPreferenceViews();
    });
    document.querySelectorAll('[data-rank-scope]').forEach(button=>button.addEventListener('click',()=>{
      state.preferenceRankScope=button.dataset.rankScope;
      updatePreferenceRankTable();
    }));
    document.querySelectorAll('[data-top-n]').forEach(button=>button.addEventListener('click',()=>{
      state.preferenceTopN=Number(button.dataset.topN);
      updatePreferenceRankTable();
    }));
    document.getElementById('preferenceRankTable').addEventListener('click',event=>{
      const button=event.target.closest('.rank-source-button');
      if(button)selectPreferenceSource(button.dataset.sourceId,button.dataset.productFamily);
    });
    preferenceSource.addEventListener('change',()=>{if(preferenceSource.value)selectPreferenceSource(preferenceSource.value);else{state.selectedPreferenceSource=null;drawPreferenceViews();}});
    const scroll=document.querySelector('.preference-rank-scroll');const sticky=document.getElementById('preferenceRankSticky');scroll.addEventListener('scroll',()=>{sticky.scrollLeft=scroll.scrollLeft;},{passive:true});
    document.querySelectorAll('[data-metric-view]').forEach(button=>button.addEventListener('click',()=>{
      const key=button.dataset.metricView+'View';state[key]=button.dataset.metricValue;
      const chartId={platformScale:'chartPlatformScale',sourceTypes:'chartSourceTypes',labelScale:'chartLabelScale'}[button.dataset.metricView];
      if(chartId)draw(chartId);updateMetricViewButtons();
    }));
  }
  function updateMetricViewButtons(){
    document.querySelectorAll('[data-metric-view]').forEach(button=>{button.ariaPressed=String(state[button.dataset.metricView+'View']===button.dataset.metricValue);});
  }
  function updateLabelOptions(){
    const el=document.getElementById('labelFilter');el.textContent='';
    const all=document.createElement('option');all.value='ALL';all.textContent='全部标签';el.appendChild(all);
    REPORT.labelStats.filter(d=>d.label_dimension===state.dimension).forEach(d=>{const op=document.createElement('option');op.value=d.label_value;op.textContent=d.label_cn;el.appendChild(op);});
    el.value=state.label;
  }
  function updateSummary(){
    const platform=state.platform==='ALL'?'全部平台':platformName[state.platform];
    const type=state.sourceType==='ALL'?'全部信源类型':state.sourceType;
    const dim=REPORT.meta.dimension_names[state.dimension]||state.dimension;
    const label=state.label==='ALL'?'全部标签':(REPORT.labelStats.find(d=>d.label_dimension===state.dimension&&d.label_value===state.label)||{}).label_cn;
    const fresh=state.freshness==='ALL'?'全部发布时间状态':state.freshness;
    document.getElementById('scopeSummary').textContent=`当前：${metricLabel()} · ${platform} · ${type} · ${dim}/${label} · ${fresh}`;
  }

  function updateTerminalRateGrid(){
    const root=document.getElementById('terminalRateGrid');root.textContent='';
    const families=[...new Set(REPORT.terminalPairs.map(d=>d.product_family))];
    families.forEach(family=>{
      const web=REPORT.terminalPairs.find(d=>d.product_family===family&&d.terminal==='web');const mobile=REPORT.terminalPairs.find(d=>d.product_family===family&&d.terminal==='mobile');
      const webValue=metric(web);const mobileValue=metric(mobile);const total=webValue+mobileValue;
      const cell=document.createElement('div');cell.className='rate-detail';
      const title=document.createElement('strong');title.textContent=preferenceFamilyLabel(family);
      const value=document.createElement('b');value.textContent=pct(mobileValue/Math.max(1,total));
      const label=document.createElement('span');label.textContent='移动端样本引用占比';
      const meta=document.createElement('small');meta.textContent='移动/电脑 '+fmt1.format(mobileValue/Math.max(1,webValue))+' 倍 · 问题覆盖差 '+signedPct((mobile.question_count-web.question_count)/Math.max(1,web.question_count));
      cell.append(title,value,label,meta);root.appendChild(cell);
    });
  }

  function preferenceFamilyLabel(value){ return value==='腾讯元宝'?'元宝':value; }
  function preferenceTerminalLabel(value){ return value==='web'?'电脑端':'移动端'; }
  function preferenceSourceLabel(d,max=24){ return short(d.source_name+(d.domain?' · '+d.domain:''),max); }
  function updatePreferenceSummary(){
    const rows=REPORT.terminalPairSummary.filter(d=>state.preferenceProduct==='ALL'||d.product_family===state.preferenceProduct);
    if(!rows.length)return;
    const questions=rows.map(d=>d.common_question_count);const similarities=rows.map(d=>d.source_jaccard);const qualifiedSimilarities=rows.map(d=>d.qualified_source_jaccard);const shared=rows.reduce((sum,d)=>sum+d.shared_sources,0);
    document.getElementById('pairCommonQuestions').textContent=rows.length===1?n(questions[0]):n(Math.min(...questions))+' 至 '+n(Math.max(...questions));
    document.getElementById('pairSourceJaccard').textContent=rows.length===1?pct(similarities[0]):pct(Math.min(...similarities))+' 至 '+pct(Math.max(...similarities));
    document.getElementById('pairQualifiedSourceJaccard').textContent=rows.length===1?pct(qualifiedSimilarities[0]):pct(Math.min(...qualifiedSimilarities))+' 至 '+pct(Math.max(...qualifiedSimilarities));
    document.getElementById('pairSharedSources').textContent=n(shared)+(rows.length>1?'（合计）':'');
    const scope=state.preferenceProduct==='ALL'?'四个产品的 8 个端点':state.preferenceProduct+'电脑端与移动端';
    const meta=REPORT.preferenceMeta;
    const inferred=meta.inferred_endpoints.length?' · 映射推定：'+meta.inferred_endpoints.join('、'):'';
    document.getElementById('preferenceScopeSummary').textContent=`当前：${scope} · ${n(meta.balanced_question_count)} 个八端共同问题 · 单问题等权 · ${n(meta.qualified_source_count)} 个筛选达标信源${inferred}`;
  }

  function updateAnchorCarryover(){
    const root=document.getElementById('anchorCarryover');root.textContent='';
    const endpointDetails=new Map(REPORT.anchorSourceMigration.map(d=>[d.platform_code,d]));
    REPORT.preferenceMeta.anchor_top20_carryover.forEach(item=>{
      const detail=endpointDetails.get(item.platform_code);const cell=document.createElement('div');cell.className='anchor-carryover-item';cell.dataset.anchor=String(Boolean(detail&&detail.is_anchor_endpoint));
      cell.dataset.endpoint=item.platform_code;cell.setAttribute('role','listitem');
      const value=document.createElement('b');value.textContent=pct(item.source_count/REPORT.preferenceMeta.anchor_pool_size);
      const label=document.createElement('span');label.textContent=preferenceFamilyLabel(detail.product_family)+' '+preferenceTerminalLabel(detail.terminal)+' · '+n(item.source_count)+' / '+n(REPORT.preferenceMeta.anchor_pool_size)+' 进入 Top 20';
      cell.append(value,label);root.appendChild(cell);
    });
  }

  function fillPreferenceRankHead(head,endpoints){
    head.textContent='';const headRow=document.createElement('tr');const rankHead=document.createElement('th');rankHead.className='rank-number';rankHead.scope='col';rankHead.textContent='排名';headRow.appendChild(rankHead);
    endpoints.forEach((endpoint,index)=>{const th=document.createElement('th');th.scope='col';th.dataset.anchor=String(index<4);th.dataset.pairStart=String(index%2===0);th.dataset.pairEnd=String(index%2===1);if(state.preferenceProduct!=='ALL'&&endpoint.product_family!==state.preferenceProduct)th.classList.add('is-dimmed');const family=document.createElement('strong');family.textContent=preferenceFamilyLabel(endpoint.product_family);const terminal=document.createElement('span');terminal.textContent=preferenceTerminalLabel(endpoint.terminal);th.append(family,terminal);headRow.appendChild(th);});head.appendChild(headRow);
  }

  function updatePreferenceRankTable(){
    document.querySelectorAll('[data-rank-scope]').forEach(button=>{button.ariaPressed=String(button.dataset.rankScope===state.preferenceRankScope);});
    document.querySelectorAll('[data-top-n]').forEach(button=>{button.ariaPressed=String(Number(button.dataset.topN)===state.preferenceTopN);});
    const pool=REPORT.sourceTopRanks.filter(d=>d.scope===state.preferenceRankScope&&d.rank<=state.preferenceTopN);
    const selectedPresent=Boolean(state.selectedPreferenceSource&&pool.some(d=>d.source_id===state.selectedPreferenceSource));
    const endpointMap=new Map();pool.forEach(d=>endpointMap.set(d.platform_code,d));const endpoints=[...endpointMap.values()].sort((a,b)=>a.endpoint_order-b.endpoint_order);
    const note=document.getElementById('preferenceRankScopeNote');
    if(state.preferenceRankScope==='common')note.textContent=`当前使用 ${n(REPORT.preferenceMeta.common_scope_question_count)} 个共同问题，八端分母一致，适合比较端点偏好。`;
    else{
      const counts=endpoints.map(d=>d.scope_question_count);note.textContent=`当前使用各端全部有效信源问题，每端 ${n(Math.min(...counts))} 至 ${n(Math.max(...counts))} 个问题，呈现端内真实分布。`;
    }
    fillPreferenceRankHead(document.getElementById('preferenceRankHead'),endpoints);fillPreferenceRankHead(document.getElementById('preferenceRankStickyHead'),endpoints);
    const body=document.getElementById('preferenceRankBody');body.textContent='';
    for(let rank=1;rank<=state.preferenceTopN;rank+=1){
      const tr=document.createElement('tr');const rankCell=document.createElement('th');rankCell.className='rank-number';rankCell.scope='row';rankCell.textContent=String(rank);tr.appendChild(rankCell);
      endpoints.forEach((endpoint,index)=>{
        const row=pool.find(d=>d.platform_code===endpoint.platform_code&&d.rank===rank);const td=document.createElement('td');td.dataset.pairStart=String(index%2===0);td.dataset.pairEnd=String(index%2===1);
        const button=document.createElement('button');button.type='button';button.className='rank-source-button';button.dataset.sourceId=row.source_id;button.dataset.productFamily=row.product_family;button.ariaPressed=String(row.source_id===state.selectedPreferenceSource);button.title=row.source_name+(row.domain?' · '+row.domain:'')+' · '+row.source_type_cn;
        const shouldDim=selectedPresent?row.source_id!==state.selectedPreferenceSource:(state.preferenceProduct!=='ALL'&&row.product_family!==state.preferenceProduct);if(shouldDim)button.classList.add('is-dimmed');
        button.setAttribute('aria-label',`${preferenceFamilyLabel(row.product_family)}${preferenceTerminalLabel(row.terminal)}第 ${rank} 位，${row.source_name}，份额 ${pct(row.share,2)}，${row.source_category_l1_cn}`);
        const name=document.createElement('span');name.className='rank-source-name';name.textContent=row.source_name;
        const meta=document.createElement('span');meta.className='rank-source-meta';const domain=document.createElement('span');domain.className='rank-source-domain';domain.textContent=row.domain||'域名未提供';const share=document.createElement('span');share.className='rank-source-share';share.textContent=pct(row.share,2);meta.append(domain,share);
        const type=document.createElement('span');type.className='rank-source-type';type.textContent=row.source_category_l1_cn;
        button.append(name,meta,type);td.appendChild(button);tr.appendChild(td);
      });body.appendChild(tr);
    }
  }

  register('chartFieldAvailability',()=>{
    const data=[...REPORT.fieldAvailability].reverse();
    return withBase({grid:{left:8,right:52,top:8,bottom:8,containLabel:true},xAxis:axisValue({max:100,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:data.map(d=>d.metric)}),series:[{type:'bar',data:data.map(d=>({value:+(d.ratio*100).toFixed(1),itemStyle:{color:d.ratio>=.98?C.brand:d.ratio>=.8?C.brand2:C.mid}})),barWidth:14,label:{show:true,position:'right',formatter:p=>p.value.toFixed(1)+'%',color:C.warm,fontSize:10},tooltip:{formatter:p=>{const d=data[p.dataIndex];return `<b>${esc(d.metric)}</b><br>${esc(d.availability_group)}<br>可用记录 ${n(d.available_records)}<br>可用率 ${pct(d.ratio)}`}}}]});
  });
  register('chartApplicability',()=>{
    const data=[...REPORT.analysisApplicability].reverse();
    return withBase({grid:{left:8,right:52,top:8,bottom:8,containLabel:true},xAxis:axisValue({max:100,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:data.map(d=>d.analysis)}),series:[{type:'bar',data:data.map(d=>+(d.ratio*100).toFixed(1)),barWidth:15,itemStyle:{color:C.brand2,borderRadius:[0,2,2,0]},label:{show:true,position:'right',formatter:p=>p.value.toFixed(1)+'%',color:C.warm,fontSize:10},tooltip:{formatter:p=>{const d=data[p.dataIndex];return `<b>${esc(d.analysis)}</b><br>可用记录 ${n(d.available_records)}<br>可用率 ${pct(d.ratio)}<br>${esc(d.boundary)}`}}}]});
  });
  register('chartPlatformAvailability',()=>{
    const platforms=REPORT.platforms.map(d=>d.platform_name_cn);const fields=[...new Set(REPORT.platformAvailability.map(d=>d.field))];
    const map=new Map(REPORT.platformAvailability.map(d=>[d.platform_name_cn+'|'+d.field,d.ratio]));
    const data=[];platforms.forEach((p,y)=>fields.forEach((field,x)=>data.push([x,y,+((map.get(p+'|'+field)||0)*100).toFixed(1)])));
    return withBase({grid:{left:10,right:20,top:12,bottom:56,containLabel:true},xAxis:axisCategory({type:'category',data:fields,axisLabel:{rotate:20,color:C.stone,fontSize:10}}),yAxis:axisCategory({type:'category',data:platforms}),visualMap:{min:0,max:100,calculable:false,orient:'horizontal',left:'center',bottom:0,text:['高','低'],inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,100),label:{show:true,formatter:p=>p.value[2].toFixed(1)+'%',fontSize:9},tooltip:{formatter:p=>`<b>${esc(platforms[p.value[1]])}</b><br>${esc(fields[p.value[0]])}可用率 ${p.value[2].toFixed(1)}%`},itemStyle:{borderColor:'#fff',borderWidth:1},emphasis:{itemStyle:{shadowBlur:8,shadowColor:'rgba(20,20,19,.16)'}}}]});
  });
  register('chartPlatformScale',()=>{
    const data=[...REPORT.platforms].sort((a,b)=>metric(a)-metric(b));const total=data.reduce((sum,d)=>sum+metric(d),0);const shareMode=state.platformScaleView==='share';
    return withBase({aria:{enabled:true,description:'十二个平台的样本引用规模与样本引用份额。当前展示'+(shareMode?'样本引用份额':metricLabel()+'数量')+'。'},grid:{left:8,right:128,top:8,bottom:8,containLabel:true},xAxis:axisValue(shareMode?{name:'样本引用份额',axisLabel:{formatter:'{value}%'}}:{name:metricLabel()}),yAxis:axisCategory({type:'category',data:data.map(d=>d.platform_name_cn)}),series:[{name:shareMode?'样本引用份额':metricLabel(),type:'bar',barWidth:19,data:data.map(d=>shareMode?+(metric(d)/Math.max(1,total)*100).toFixed(2):metric(d)),itemStyle:{color:C.brand,borderRadius:[0,2,2,0]},label:{show:true,position:'right',formatter:p=>shareMode?p.value.toFixed(1)+'%':n(p.value)+' · '+pct(metric(data[p.dataIndex])/total),color:C.warm,fontSize:10},tooltip:{formatter:p=>{const d=data[p.dataIndex];return `<b>${esc(d.platform_name_cn)}</b><br>${metricLabel()} ${n(metric(d))}<br>样本引用份额 ${pct(metric(d)/total)}<br>精确去重保留率 ${pct(d.dedup_count/d.raw_count)}<br>题均引用 ${fmt1.format(metric(d)/Math.max(1,d.question_count))}<br>覆盖问题 ${n(d.question_count)} / ${n(REPORT.overview.questions)}`;}}}]});
  });
  register('chartCoverage',()=>{
    const data=[...REPORT.platforms].sort((a,b)=>a.question_count-b.question_count);
    return withBase({grid:{left:8,right:46,top:8,bottom:8,containLabel:true},xAxis:axisValue({max:100,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:data.map(d=>d.platform_name_cn)}),series:[{type:'bar',barWidth:14,data:data.map(d=>+(d.question_count/REPORT.overview.questions*100).toFixed(1)),itemStyle:{color:p=>p.value>=90?C.brand:C.brand2,borderRadius:[0,2,2,0]},label:{show:true,position:'right',formatter:p=>p.value.toFixed(0)+'%',fontSize:9,color:C.warm}}]});
  });
  register('chartPlatformBreadth',()=>withBase({grid:{left:18,right:28,top:18,bottom:36,containLabel:true},xAxis:axisValue({name:'规范信源数',nameLocation:'middle',nameGap:26}),yAxis:axisValue({name:'去重引用',nameGap:48}),series:[{type:'scatter',symbolSize:d=>Math.max(12,Math.sqrt(d[2])/3.2),data:REPORT.platforms.map(d=>[d.source_count,d.dedup_count,d.page_count,d.platform_name_cn]),itemStyle:{color:C.brand,opacity:.78},label:{show:true,position:'top',formatter:p=>short(p.value[3],8),fontSize:10,color:C.warm},tooltip:{formatter:p=>`<b>${esc(p.value[3])}</b><br>信源 ${n(p.value[0])}<br>去重引用 ${n(p.value[1])}<br>页面 ${n(p.value[2])}`}}]}));
  register('chartDensity',()=>{
    const key=state.scope==='raw'?'raw':'dedup';const data=[...REPORT.platformDensity].sort((a,b)=>a[key+'_median']-b[key+'_median']);
    return withBase({grid:{left:8,right:28,top:10,bottom:34,containLabel:true},xAxis:axisValue({name:'每问题引用数',nameLocation:'middle',nameGap:25}),yAxis:axisCategory({type:'category',data:data.map(d=>d.platform_name_cn)}),series:[{type:'boxplot',data:data.map(d=>[d[key+'_min'],d[key+'_q1'],d[key+'_median'],d[key+'_q3'],d[key+'_max']]),itemStyle:{color:C.pale,borderColor:C.brand,borderWidth:1.5},tooltip:{formatter:p=>{const v=p.value;return `<b>${esc(data[p.dataIndex].platform_name_cn)}</b><br>最小 ${n(v[1])}<br>Q1 ${fmt1.format(v[2])}<br>中位数 ${fmt1.format(v[3])}<br>Q3 ${fmt1.format(v[4])}<br>最大 ${n(v[5])}`}}}]});
  });
  register('chartTerminal',()=>{
    const families=[...new Set(REPORT.terminalPairs.map(d=>d.product_family))];
    const get=(f,t)=>REPORT.terminalPairs.find(x=>x.product_family===f&&x.terminal===t);const tooltip=p=>{const family=families[p.dataIndex];const web=get(family,'web');const mobile=get(family,'mobile');const webValue=metric(web);const mobileValue=metric(mobile);return `<b>${esc(preferenceFamilyLabel(family))}</b><br>电脑端 ${n(webValue)} · 移动端 ${n(mobileValue)}<br>移动端样本引用占比 ${pct(mobileValue/Math.max(1,webValue+mobileValue))}<br>移动/电脑 ${fmt1.format(mobileValue/Math.max(1,webValue))} 倍<br>问题覆盖差 ${signedPct((mobile.question_count-web.question_count)/Math.max(1,web.question_count))}`;};
    return withBase({legend:{top:0,textStyle:{color:C.stone,fontSize:11}},grid:{left:12,right:12,top:38,bottom:30,containLabel:true},xAxis:axisCategory({type:'category',data:families.map(preferenceFamilyLabel)}),yAxis:axisValue({name:metricLabel()}),series:[{name:'电脑端',type:'bar',data:families.map(f=>metric(get(f,'web'))),itemStyle:{color:C.brand},tooltip:{formatter:tooltip}},{name:'移动端',type:'bar',data:families.map(f=>metric(get(f,'mobile'))),itemStyle:{color:C.brand2},tooltip:{formatter:tooltip}}]});
  });
  register('chartTopSources',()=>{
    const rows=REPORT.sourceFilterRows.filter(d=>d.platform_filter===state.platform&&d.type_filter===state.sourceType).sort(byMetricDesc).slice(0,20).reverse();
    const totalKey=state.scope==='raw'?'filter_raw_total':'filter_dedup_total';
    return withBase({grid:{left:8,right:132,top:8,bottom:8,containLabel:true},xAxis:axisValue({name:metricLabel()}),yAxis:axisCategory({type:'category',data:rows.map(d=>short(d.source_name,18))}),series:[{type:'bar',barWidth:15,data:rows.map(d=>metric(d)),itemStyle:{color:C.brand,borderRadius:[0,2,2,0]},label:{show:true,position:'right',formatter:p=>compact(p.value)+' · '+pct(metric(rows[p.dataIndex])/Math.max(1,rows[p.dataIndex][totalKey])),fontSize:10,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.source_name)}</b><br>${esc(d.domain||'')}<br>${metricLabel()} ${n(metric(d))}<br>样本引用份额 ${pct(metric(d)/Math.max(1,d[totalKey]))}<br>当前筛选问题覆盖率 ${pct(d.question_count/Math.max(1,d.filter_question_count))}<br>覆盖问题 ${n(d.question_count)} / ${n(d.filter_question_count)} · 页面 ${n(d.page_count)}`}}}]});
  });
  register('chartSourceTypes',()=>{
    const allRows=REPORT.sourceTypes.filter(d=>d.platform_code===state.platform);const total=allRows.reduce((sum,d)=>sum+metric(d),0);const shareMode=state.sourceTypesView==='share';const rows=allRows.sort(byMetricDesc).slice(0,16).reverse();const platformQuestions=state.platform==='ALL'?REPORT.overview.questions:(REPORT.platforms.find(d=>d.platform_code===state.platform)||{}).question_count;
    return withBase({aria:{enabled:true,description:'当前平台信源类型的引用规模与样本引用构成。当前展示'+(shareMode?'构成率':metricLabel()+'数量')+'。'},grid:{left:8,right:108,top:8,bottom:8,containLabel:true},xAxis:axisValue(shareMode?{name:'样本引用构成率',axisLabel:{formatter:'{value}%'}}:{name:metricLabel()}),yAxis:axisCategory({type:'category',data:rows.map(d=>d.source_type_cn)}),series:[{type:'bar',barWidth:14,data:rows.map(d=>({value:shareMode?+(metric(d)/Math.max(1,total)*100).toFixed(2):metric(d),itemStyle:{color:state.sourceType===d.source_type_cn?C.warn:(state.sourceType==='ALL'?C.brand:C.mid)}})),label:{show:true,position:'right',formatter:p=>shareMode?p.value.toFixed(1)+'%':compact(p.value)+' · '+pct(metric(rows[p.dataIndex])/Math.max(1,total)),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.source_type_cn)}</b><br>${metricLabel()} ${n(metric(d))}<br>样本引用构成率 ${pct(metric(d)/Math.max(1,total))}<br>问题覆盖率 ${pct(d.question_count/Math.max(1,platformQuestions))}<br>问题 ${n(d.question_count)} / ${n(platformQuestions)} · 信源 ${n(d.source_count)}`;}}}]});
  });
  register('chartPareto',()=>{
    const rows=REPORT.sourcePareto.slice(0,50);
    return withBase({legend:{top:0,textStyle:{color:C.stone,fontSize:10}},grid:{left:12,right:46,top:38,bottom:34,containLabel:true},xAxis:axisCategory({type:'category',data:rows.map(d=>d.rank),axisLabel:{interval:4}}),yAxis:[axisValue({name:'去重引用'}),axisValue({name:'累计份额',min:0,max:100,axisLabel:{formatter:'{value}%'}})],series:[{name:'单个信源',type:'bar',data:rows.map(d=>d.dedup_count),itemStyle:{color:C.brand},barMaxWidth:14},{name:'累计份额',type:'line',yAxisIndex:1,data:rows.map(d=>+(d.cumulative_share*100).toFixed(1)),symbol:'none',lineStyle:{color:C.warn,width:2}}]});
  });
  register('chartSourcePlatform',()=>{
    const sources=[...new Set(REPORT.sourcePlatform.map(d=>d.source_name))];const platforms=REPORT.platforms.map(d=>d.platform_name_cn);const map=new Map(REPORT.sourcePlatform.map(d=>[d.source_name+'|'+d.platform_name_cn,d.citation_count]));const data=[];
    platforms.forEach((p,x)=>sources.forEach((s,y)=>data.push([x,y,map.get(s+'|'+p)||0])));
    const max=Math.max(...data.map(d=>d[2]));
    return withBase({grid:{left:8,right:18,top:10,bottom:64,containLabel:true},xAxis:axisCategory({type:'category',data:platforms,axisLabel:{rotate:30,fontSize:10}}),yAxis:axisCategory({type:'category',data:sources.map(s=>short(s,18))}),visualMap:{min:0,max,orient:'horizontal',left:'center',bottom:0,inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,max),label:{show:false},itemStyle:{borderColor:'#fff',borderWidth:1}}]});
  });
  register('chartEcosystem',()=>{
    const rows=REPORT.ecosystem.filter(d=>d.ecosystem!=='未归属生态').slice(0,15).reverse();
    return withBase({grid:{left:8,right:72,top:8,bottom:8,containLabel:true},xAxis:axisValue(),yAxis:axisCategory({type:'category',data:rows.map(d=>d.ecosystem)}),series:[{type:'bar',barWidth:14,data:rows.map((d,i)=>({value:d.dedup_count,itemStyle:{color:i===rows.length-1?C.brand:C.brand2}})),label:{show:true,position:'right',formatter:p=>compact(p.value),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.ecosystem)}</b><br>信源 ${n(d.source_count)}<br>页面 ${n(d.page_count)}<br>去重引用 ${n(d.dedup_count)}`}}}]});
  });
  register('chartUnclassifiedSources',()=>{
    const rows=[...REPORT.ecosystemUnclassifiedTop].reverse();
    return withBase({grid:{left:8,right:64,top:8,bottom:8,containLabel:true},xAxis:axisValue(),yAxis:axisCategory({type:'category',data:rows.map(d=>short(d.source_name,13))}),series:[{type:'bar',barWidth:15,data:rows.map(d=>d.dedup_count),itemStyle:{color:C.mid},label:{show:true,position:'right',formatter:p=>compact(p.value),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.source_name)}</b><br>${esc(d.domain||'')}<br>问题 ${n(d.question_count)} · 页面 ${n(d.page_count)}<br>去重引用 ${n(d.dedup_count)}`}}}]});
  });
  register('chartSourcePreference',()=>{
    const rows=REPORT.sourcePreference.filter(d=>state.preferenceProduct==='ALL'||d.product_family===state.preferenceProduct);
    const endpointMap=new Map();const sourceMap=new Map();
    rows.forEach(d=>{endpointMap.set(d.platform_code,d);sourceMap.set(d.source_id,d);});
    const endpoints=[...endpointMap.values()].sort((a,b)=>a.endpoint_order-b.endpoint_order);
    const sources=[...sourceMap.values()].sort((a,b)=>a.source_rank-b.source_rank);
    const xIndex=new Map(endpoints.map((d,i)=>[d.platform_code,i]));const yIndex=new Map(sources.map((d,i)=>[d.source_id,i]));
    const selectedPresent=Boolean(state.selectedPreferenceSource&&sourceMap.has(state.selectedPreferenceSource));
    const data=rows.map(d=>({
      value:[xIndex.get(d.platform_code),yIndex.get(d.source_id),+d.preference_index.toFixed(1)],sourceId:d.source_id,detail:d,
      itemStyle:{opacity:selectedPresent&&state.selectedPreferenceSource!==d.source_id ? .18 : 1},
      label:{color:d.preference_index>=165||d.preference_index<=35?C.white:C.ink}
    }));
    return withBase({
      aria:{enabled:true,description:'四个 AI 产品电脑端和移动端的信源偏好指数矩阵，指数 100 表示该信源在八个端点中的平均水平。'},
      grid:{left:8,right:18,top:12,bottom:82,containLabel:true},
      xAxis:axisCategory({type:'category',data:endpoints.map(d=>preferenceFamilyLabel(d.product_family)+'\n'+preferenceTerminalLabel(d.terminal)),axisLabel:{interval:0,fontSize:10,color:C.stone}}),
      yAxis:axisCategory({type:'category',inverse:true,data:sources.map(d=>preferenceSourceLabel(d)),axisLabel:{fontSize:10,color:C.stone}}),
      visualMap:{min:0,max:200,calculable:false,orient:'horizontal',left:'center',bottom:8,text:['高偏好','低偏好'],inRange:{color:['#b46b45','#ead8c8','#faf9f5','#b4c6d8','#1b365d']},textStyle:{color:C.stone,fontSize:10}},
      series:[{type:'heatmap',data,label:{show:true,formatter:p=>p.value[2]>=130||p.value[2]<=70?Math.round(p.value[2]):'',fontSize:9},itemStyle:{borderColor:C.white,borderWidth:1},emphasis:{itemStyle:{borderColor:C.ink,borderWidth:2}},tooltip:{formatter:p=>{const d=p.data.detail;return `<b>${esc(d.source_name)}</b><br>${esc(d.domain||'域名未提供')}<br>${esc(d.product_family)} · ${preferenceTerminalLabel(d.terminal)}<br>偏好指数 ${fmt1.format(d.preference_index)}<br>等权份额 ${pct(d.weighted_share,2)}<br>引用 ${n(d.citation_count)} · 问题 ${n(d.question_count)}<br>八端样本覆盖问题 ${n(d.source_question_count)}`;}}}]
    });
  });
  register('chartAnchorMigration',()=>{
    const rows=REPORT.anchorSourceMigration;
    const endpointMap=new Map();const sourceMap=new Map();rows.forEach(d=>{endpointMap.set(d.platform_code,d);sourceMap.set(d.source_id,d);});
    const endpoints=[...endpointMap.values()].sort((a,b)=>a.endpoint_order-b.endpoint_order);const sources=[...sourceMap.values()].sort((a,b)=>a.anchor_order-b.anchor_order);const selectedPresent=Boolean(state.selectedPreferenceSource&&sourceMap.has(state.selectedPreferenceSource));
    const xIndex=new Map(endpoints.map((d,i)=>[d.platform_code,i]));const yIndex=new Map(sources.map((d,i)=>[d.source_id,i]));const ceiling=Math.max(1,...rows.map(d=>d.share*100));
    const endpointLabels=endpoints.map(d=>preferenceFamilyLabel(d.product_family)+'\n'+preferenceTerminalLabel(d.terminal));
    const data=rows.map(d=>({
      value:[xIndex.get(d.platform_code),yIndex.get(d.source_id),+(d.share*100).toFixed(3)],sourceId:d.source_id,detail:d,rankLabel:d.rank==null?'·':String(d.rank),
      itemStyle:{opacity:selectedPresent?(state.selectedPreferenceSource===d.source_id?1:.16):(state.preferenceProduct==='ALL'||state.preferenceProduct===d.product_family?1:.35)},
      label:{color:d.share*100>=ceiling*.5?C.white:C.ink}
    }));
    return withBase({
      aria:{enabled:true,description:'豆包和 DeepSeek 四个锚定端 Top 10 信源在八个端点的排名与等权份额矩阵。'},
      grid:{left:8,right:18,top:52,bottom:68,containLabel:true},
      xAxis:axisCategory({type:'category',position:'top',data:endpointLabels,axisLabel:{interval:0,fontSize:10,color:C.stone,formatter:value=>{const index=endpointLabels.indexOf(value);return `{${index<4?'anchor':'control'}|${value}}`;},rich:{anchor:{color:C.brand,fontWeight:600,lineHeight:15},control:{color:C.stone,lineHeight:15}}}}),
      yAxis:axisCategory({type:'category',inverse:true,data:sources.map(d=>preferenceSourceLabel(d,28)),axisLabel:{fontSize:10,color:C.stone}}),
      visualMap:{min:0,max:ceiling,calculable:false,orient:'horizontal',left:'center',bottom:6,text:['高份额','低份额'],inRange:{color:['#f5f3ed','#ead8c8','#b4c6d8','#4c7098','#1b365d']},textStyle:{color:C.stone,fontSize:10}},
      series:[{type:'heatmap',data,label:{show:true,formatter:p=>p.data.rankLabel,fontSize:10,fontWeight:600},itemStyle:{borderColor:C.white,borderWidth:1},emphasis:{itemStyle:{borderColor:C.ink,borderWidth:2}},tooltip:{formatter:p=>{const d=p.data.detail;return `<b>${esc(d.source_name)}</b><br>${esc(d.domain||'域名未提供')}<br>${esc(d.product_family)} · ${preferenceTerminalLabel(d.terminal)}<br>等权份额 ${pct(d.share,2)}<br>排名 ${d.rank==null?'未出现':n(d.rank)}<br>锚定端 Top 10 出现率 ${pct(d.anchor_top10_occurrences/4)}（${n(d.anchor_top10_occurrences)} / 4）<br>引用 ${n(d.citation_count)} · 问题 ${n(d.question_count)}<br>一级类型 ${esc(d.source_category_l1_cn)}<br>二级类型 ${esc(d.source_type_cn)}`;}}}]
    });
  });
  register('chartTerminalTilt',()=>{
    const pool=REPORT.terminalTilt.filter(d=>state.preferenceProduct==='ALL'||d.product_family===state.preferenceProduct);
    let rows=[];
    if(state.preferenceProduct==='ALL'){
      [...new Set(pool.map(d=>d.product_family))].forEach(family=>rows.push(...pool.filter(d=>d.product_family===family).sort((a,b)=>Math.abs(b.delta_pp)-Math.abs(a.delta_pp)).slice(0,4)));
    }else rows=pool.sort((a,b)=>Math.abs(b.delta_pp)-Math.abs(a.delta_pp)).slice(0,15);
    if(state.selectedPreferenceSource){
      const selected=pool.filter(d=>d.source_id===state.selectedPreferenceSource);
      selected.forEach(d=>{if(!rows.some(row=>row.product_family===d.product_family&&row.source_id===d.source_id))rows.push(d);});
    }
    rows.sort((a,b)=>a.product_family.localeCompare(b.product_family,'zh-CN')||a.delta_pp-b.delta_pp);
    const labels=rows.map(d=>(state.preferenceProduct==='ALL'?preferenceFamilyLabel(d.product_family)+' · ':'')+preferenceSourceLabel(d,20));
    const ceiling=Math.max(1,...rows.flatMap(d=>[d.web_share*100,d.mobile_share*100]));const selectedPresent=Boolean(state.selectedPreferenceSource&&pool.some(d=>d.source_id===state.selectedPreferenceSource));
    const barData=(terminal)=>rows.map(d=>({value:+((terminal==='web'?-1:1)*d[terminal+'_share']*100).toFixed(2),detail:d,sourceId:d.source_id,itemStyle:{opacity:selectedPresent&&state.selectedPreferenceSource!==d.source_id ? .18 : 1}}));
    return withBase({
      aria:{enabled:true,description:'共同问题范围内的电脑端与移动端信源份额镜像对比。'},legend:{top:0,textStyle:{color:C.stone,fontSize:10}},
      grid:{left:8,right:18,top:38,bottom:42,containLabel:true},
      xAxis:axisValue({min:-ceiling,max:ceiling,name:'单问题等权份额',nameLocation:'middle',nameGap:28,axisLabel:{formatter:value=>Math.abs(value).toFixed(0)+'%',color:C.stone,fontSize:10}}),
      yAxis:axisCategory({type:'category',data:labels,axisLabel:{fontSize:10,color:C.stone}}),
      series:[
        {name:'电脑端',type:'bar',stack:'terminal',data:barData('web'),barWidth:13,itemStyle:{color:C.brand},tooltip:{formatter:p=>{const d=p.data.detail;return `<b>${esc(d.source_name)}</b><br>${esc(d.product_family)} · 电脑端<br>等权份额 ${pct(d.web_share,2)}<br>移动端 ${pct(d.mobile_share,2)}<br>端差 ${d.delta_pp>=0?'+':''}${d.delta_pp.toFixed(2)} 个百分点`;}}},
        {name:'移动端',type:'bar',stack:'terminal',data:barData('mobile'),barWidth:13,itemStyle:{color:C.warn},tooltip:{formatter:p=>{const d=p.data.detail;return `<b>${esc(d.source_name)}</b><br>${esc(d.product_family)} · 移动端<br>等权份额 ${pct(d.mobile_share,2)}<br>电脑端 ${pct(d.web_share,2)}<br>端差 ${d.delta_pp>=0?'+':''}${d.delta_pp.toFixed(2)} 个百分点`;}}}
      ]
    });
  });
  register('chartPreferenceTypes',()=>{
    const rows=REPORT.preferenceTypeMix.filter(d=>state.preferenceProduct==='ALL'||d.product_family===state.preferenceProduct);
    const endpointMap=new Map();rows.forEach(d=>endpointMap.set(d.platform_code,d));const endpoints=[...endpointMap.values()].sort((a,b)=>a.endpoint_order-b.endpoint_order);
    const categories=['平台与社区','新闻与媒体','垂直专业内容','商业信息与服务','研究与文档','政府与公共机构','品牌与企业官网','搜索与页面代理','未分类长尾','信源未规范化'];
    const typeColors=['#1b365d','#2d5a8a','#65745b','#8c6f56','#6e7d94','#9a806d','#7a8871','#a99f90','#d6e1ee','#ead8c8'];
    const detailFor=(endpoint,type)=>rows.find(d=>d.platform_code===endpoint.platform_code&&d.source_category_l1_cn===type);
    return withBase({
      aria:{enabled:true,description:'各端点一级信源类型的单问题等权结构，横向堆叠总和为百分之百。'},legend:{top:0,type:'scroll',textStyle:{color:C.stone,fontSize:9}},
      grid:{left:8,right:20,top:54,bottom:24,containLabel:true},xAxis:axisValue({min:0,max:100,axisLabel:{formatter:'{value}%'}}),
      yAxis:axisCategory({type:'category',data:endpoints.map(d=>preferenceFamilyLabel(d.product_family)+' · '+preferenceTerminalLabel(d.terminal)),axisLabel:{fontSize:10,color:C.stone}}),
      series:categories.map((type,i)=>({name:type,type:'bar',stack:'type',barWidth:18,data:endpoints.map(endpoint=>{const detail=detailFor(endpoint,type);return {value:+((detail?detail.weighted_share:0)*100).toFixed(2),detail,type};}),itemStyle:{color:typeColors[i]},tooltip:{formatter:p=>{const d=p.data.detail;if(!d)return `<b>${esc(type)}</b><br>当前端点未观察到该类型`;const breakdown=d.source_type_breakdown.slice(0,8).map(item=>`${esc(item.source_type_cn)} ${pct(item.weighted_share,2)}`).join('<br>');return `<b>${esc(d.product_family)} · ${preferenceTerminalLabel(d.terminal)}</b><br>一级类型 ${esc(d.source_category_l1_cn)}<br>等权份额 ${pct(d.weighted_share,2)}<br>引用 ${n(d.citation_count)} · 问题 ${n(d.question_count)}${breakdown?'<br><br><b>二级类型</b><br>'+breakdown:''}`;}}}))
    });
  });
  register('chartOverlapHeat',()=>{
    const names=REPORT.platforms.map(d=>d.platform_name_cn);const axisNames=names.map(name=>name.replace('腾讯元宝','元宝').replace('DeepSeek','DS').replace('网页版','Web').replace('手机版','移动'));const pairMap=new Map();
    REPORT.overlap.forEach(d=>{const v=+(d.jaccard_similarity*100).toFixed(1);pairMap.set(d.platform_a_name+'|'+d.platform_b_name,v);pairMap.set(d.platform_b_name+'|'+d.platform_a_name,v);});
    const data=[];names.forEach((row,y)=>names.forEach((column,x)=>{if(x===y)data.push({value:[x,y,0],itemStyle:{color:'#f0eee8'},label:{color:C.stone}});else{const value=pairMap.get(column+'|'+row)||0;data.push({value:[x,y,value],label:{color:value>=14?'#fff':C.ink}});}}));
    return withBase({grid:{left:8,right:8,top:8,bottom:72,containLabel:true},xAxis:axisCategory({type:'category',data:axisNames,axisLabel:{rotate:30,fontSize:10}}),yAxis:axisCategory({type:'category',data:axisNames}),visualMap:{min:0,max:35,orient:'horizontal',left:'center',bottom:0,text:['高','低'],inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data,label:{show:true,formatter:p=>p.value[0]===p.value[1]?'自身':(p.value[2]===0?'0':(p.value[2]>=.5?p.value[2].toFixed(1):'')),fontSize:9},itemStyle:{borderColor:'#fff',borderWidth:2},tooltip:{formatter:p=>p.value[0]===p.value[1]?`<b>${esc(names[p.value[0]])}</b><br>平台自身不参与相似度比较`:`<b>${esc(names[p.value[1]])} × ${esc(names[p.value[0]])}</b><br>Jaccard 相似度 ${p.value[2].toFixed(1)}%`}}]});
  });
  register('chartNetwork',()=>{
    const positives=REPORT.overlap.filter(d=>d.jaccard_similarity>0).map(d=>d.jaccard_similarity).sort((a,b)=>a-b);const median=positives[Math.floor(positives.length/2)]||0;
    const nodes=REPORT.platforms.map(d=>({id:d.platform_code,name:d.platform_name_cn,value:d.dedup_count,symbolSize:16+Math.sqrt(d.dedup_count)/15,itemStyle:{color:d.terminal==='mobile'?C.brand2:C.brand}}));
    const links=REPORT.overlap.filter(d=>d.jaccard_similarity>=median&&d.jaccard_similarity>0).map(d=>({source:d.platform_a,target:d.platform_b,value:d.jaccard_similarity,lineStyle:{width:1+d.jaccard_similarity*9,opacity:.42,color:C.brand2}}));
    return withBase({tooltip:{formatter:p=>p.dataType==='edge'?`相似度 ${pct(p.value)}`:`<b>${esc(p.name)}</b><br>去重引用 ${n(p.value)}`},series:[{type:'graph',layout:'circular',circular:{rotateLabel:true},roam:false,data:nodes,links,label:{show:true,position:'right',fontSize:10,color:C.warm},lineStyle:{curveness:.16},emphasis:{focus:'adjacency'}}]});
  });
  register('chartSharedPairs',()=>{
    const rows=[...REPORT.overlap].sort((a,b)=>a.shared_question_page_count-b.shared_question_page_count).slice(-16);
    return withBase({grid:{left:8,right:58,top:8,bottom:8,containLabel:true},xAxis:axisValue(),yAxis:axisCategory({type:'category',data:rows.map(d=>short(d.platform_a_name.replace('网页版','Web').replace('手机版','移动')+' × '+d.platform_b_name.replace('网页版','Web').replace('手机版','移动'),24))}),series:[{type:'bar',barWidth:13,data:rows.map(d=>d.shared_question_page_count),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>compact(p.value),fontSize:9,color:C.warm}}]});
  });
  register('chartSharedUnique',()=>{
    const rows=[...REPORT.sharedUnique].sort((a,b)=>a.shared_pairs/a.total_pairs-b.shared_pairs/b.total_pairs);
    return withBase({aria:{enabled:true,description:'平台共享与独有问题页面组合占比堆叠条形图。每个平台的两部分合计为百分之百。'},legend:{top:0,textStyle:{fontSize:10,color:C.stone}},grid:{left:8,right:18,top:38,bottom:8,containLabel:true},xAxis:axisValue({max:100,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:rows.map(d=>d.platform_name_cn)}),series:[{name:'独有',type:'bar',stack:'x',data:rows.map(d=>+(d.unique_pairs/d.total_pairs*100).toFixed(1)),itemStyle:{color:C.mid}},{name:'共享',type:'bar',stack:'x',data:rows.map(d=>+(d.shared_pairs/d.total_pairs*100).toFixed(1)),itemStyle:{color:C.brand}}]});
  });
  register('chartConsensus',()=>{
    const rows=REPORT.consensusSources.slice(0,16).reverse();
    return withBase({grid:{left:8,right:58,top:8,bottom:8,containLabel:true},xAxis:axisValue(),yAxis:axisCategory({type:'category',data:rows.map(d=>short(d.source_name,17))}),series:[{type:'bar',barWidth:14,data:rows.map(d=>d.consensus_score),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>compact(p.value),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.source_name)}</b><br>平台 ${d.platform_count} · 问题 ${d.question_count}<br>去重引用 ${n(d.dedup_count)}`}}}]});
  });
  register('chartTaxonomy',()=>{
    const groups=[...new Set(REPORT.labelStats.map(d=>d.label_dimension))].map(dim=>({name:REPORT.meta.dimension_names[dim]||dim,value:REPORT.labelStats.filter(d=>d.label_dimension===dim).reduce((s,d)=>s+d.question_count,0),itemStyle:{borderColor:dim===state.dimension?C.warn:'#fff',borderWidth:dim===state.dimension?4:2},children:REPORT.labelStats.filter(d=>d.label_dimension===dim).map(d=>({name:d.label_cn,value:d.question_count}))}));
    return withBase({tooltip:{formatter:p=>`<b>${esc(p.name)}</b><br>问题 ${n(p.value)}`},series:[{type:'treemap',left:0,right:0,top:0,bottom:0,roam:false,nodeClick:false,sort:'desc',visibleMin:4,breadcrumb:{show:false},data:groups,label:{show:true,formatter:p=>short(p.name,9),fontSize:11,color:'#fff',overflow:'truncate'},upperLabel:{show:true,height:28,formatter:'{b}',color:'#fff',fontSize:11},levels:[{itemStyle:{borderColor:'#fff',borderWidth:3,gapWidth:3}},{color:palette,itemStyle:{borderColor:'#fff',borderWidth:2,gapWidth:2},upperLabel:{show:true,height:28,color:'#fff',fontSize:11}},{colorSaturation:[.46,.7],itemStyle:{borderColor:'#fff',borderWidth:1,gapWidth:1},label:{show:true,color:'#fff',fontSize:11,overflow:'truncate'}}]}]});
  });
  register('chartLabelScale',()=>{
    const perQuestionKey=state.scope==='raw'?'avg_raw_citations_per_question':'avg_dedup_citations_per_question';const rateMode=state.labelScaleView==='rate';const value=d=>rateMode?d[perQuestionKey]:metric(d);const rows=REPORT.labelStats.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label)).sort((a,b)=>value(a)-value(b));
    return withBase({aria:{enabled:true,description:'当前问题维度各标签的引用规模和题均引用。当前展示'+(rateMode?'题均引用':metricLabel()+'数量')+'。'},grid:{left:8,right:122,top:8,bottom:8,containLabel:true},xAxis:axisValue({name:rateMode?'题均引用':metricLabel()}),yAxis:axisCategory({type:'category',data:rows.map(d=>d.label_cn)}),series:[{type:'bar',barMaxWidth:20,data:rows.map(d=>+value(d).toFixed(rateMode?2:0)),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>rateMode?fmt1.format(p.value):compact(p.value)+' · 题均 '+fmt1.format(rows[p.dataIndex][perQuestionKey]),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.label_cn)}</b><br>${metricLabel()} ${n(metric(d))}<br>题均引用 ${fmt1.format(d[perQuestionKey])}<br>题均信源 ${fmt1.format(d.avg_sources_per_question)} · 题均页面 ${fmt1.format(d.avg_pages_per_question)}<br>题均平台 ${fmt1.format(d.avg_platforms_per_question)}<br>标签问题 ${n(d.question_count)}`;}}}]});
  });
  register('chartLabelPlatform',()=>{
    const labels=REPORT.labelStats.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label)).map(d=>d.label_cn);const platforms=REPORT.platforms.map(d=>d.platform_name_cn);const rows=REPORT.labelPlatform.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label));const map=new Map(rows.map(d=>[d.platform_name_cn+'|'+d.label_cn,metric(d)]));const data=[];platforms.forEach((p,x)=>labels.forEach((l,y)=>data.push([x,y,map.get(p+'|'+l)||0])));const max=Math.max(1,...data.map(d=>d[2]));
    return withBase({grid:{left:8,right:15,top:8,bottom:74,containLabel:true},xAxis:axisCategory({type:'category',data:platforms,axisLabel:{rotate:32,fontSize:10}}),yAxis:axisCategory({type:'category',data:labels}),visualMap:{min:0,max,orient:'horizontal',left:'center',bottom:0,inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,max),label:{show:true,formatter:p=>p.value[2]?compact(p.value[2]):'',fontSize:9},itemStyle:{borderColor:'#fff',borderWidth:1}}]});
  });
  register('chartLabelDiversity',()=>{
    const rows=REPORT.labelStats.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label));
    return withBase({grid:{left:16,right:28,top:18,bottom:38,containLabel:true},xAxis:axisValue({name:'题均规范信源数',nameLocation:'middle',nameGap:27}),yAxis:axisValue({name:'题均规范页面数',nameGap:44}),series:[{type:'scatter',data:rows.map(d=>[+d.avg_sources_per_question.toFixed(2),+d.avg_pages_per_question.toFixed(2),d.question_count,d.label_cn,d.source_count,d.page_count]),symbolSize:d=>12+Math.sqrt(d[2])*4,itemStyle:{color:C.brand,opacity:.76},label:{show:true,position:'top',formatter:p=>p.value[3],fontSize:10,color:C.warm},tooltip:{formatter:p=>`<b>${esc(p.value[3])}</b><br>题均信源 ${fmt1.format(p.value[0])} · 题均页面 ${fmt1.format(p.value[1])}<br>信源池 ${n(p.value[4])} · 页面池 ${n(p.value[5])}<br>标签问题 ${n(p.value[2])}`}}]});
  });
  register('chartFeaturePrevalence',()=>{
    const rows=[...REPORT.featurePrevalence].reverse();return withBase({grid:{left:8,right:58,top:8,bottom:8,containLabel:true},xAxis:axisValue({max:35,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:rows.map(d=>d.feature)}),series:[{type:'bar',barWidth:18,data:rows.map(d=>+(d.ratio*100).toFixed(1)),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>p.value.toFixed(1)+'%',fontSize:10,color:C.warm}}]});
  });
  register('chartFormats',()=>{
    const rows=REPORT.formatPlatform.filter(d=>d.platform_code===state.platform).sort((a,b)=>b.page_count-a.page_count);const labels={general:'通用',ranking:'榜单',guide:'指南',comparison:'对比',unknown:'标题未提供'};
    return withBase({legend:{top:0,textStyle:{fontSize:10,color:C.stone}},grid:{left:10,right:42,top:38,bottom:30,containLabel:true},xAxis:axisCategory({type:'category',data:rows.map(d=>labels[d.content_format_hint]||d.content_format_hint)}),yAxis:[axisValue({name:'页面数'}),axisValue({name:'每页引用'})],series:[{name:'页面数',type:'bar',data:rows.map(d=>d.page_count),itemStyle:{color:C.brand},barMaxWidth:28},{name:'每页引用',type:'line',yAxisIndex:1,data:rows.map(d=>+(metric(d)/Math.max(1,d.page_count)).toFixed(2)),lineStyle:{color:C.warn,width:2},itemStyle:{color:C.warn}}]});
  });
  function lengthOption(metricName,order){
    const rows=REPORT.lengthPerformance.filter(d=>d.metric===metricName).sort((a,b)=>order.indexOf(a.band)-order.indexOf(b.band));
    const unavailable=metricName==='标题长度'?'标题未提供':'摘要未提供';
    return withBase({legend:{top:0,textStyle:{fontSize:10,color:C.stone}},grid:{left:10,right:42,top:38,bottom:32,containLabel:true},xAxis:axisCategory({type:'category',data:rows.map(d=>d.band)}),yAxis:[axisValue({name:'页面数'}),axisValue({name:'平均引用'})],series:[{name:'页面数',type:'bar',data:rows.map(d=>({value:d.page_count,itemStyle:{color:d.band===unavailable?C.mid:C.brand2}})),barMaxWidth:28,tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.band)}</b><br>页面 ${n(d.page_count)}${d.band===unavailable?'<br>不进入长度表现比较':'<br>平均引用 '+fmt1.format(d.avg_citations)}`}}},{name:'平均引用',type:'line',yAxisIndex:1,data:rows.map(d=>d.band===unavailable?null:+d.avg_citations.toFixed(2)),lineStyle:{color:C.warn,width:2},itemStyle:{color:C.warn}}]});
  }
  register('chartTitleLength',()=>lengthOption('标题长度',['≤20','21 至 30','31 至 40','41 至 50','51 至 60','61+','标题未提供']));
  register('chartSnippetLength',()=>lengthOption('摘要长度',['0','1 至 50','51 至 100','101 至 200','201 至 300','301+','摘要未提供']));
  register('chartFeaturePlatform',()=>{
    const platforms=REPORT.platforms.map(d=>d.platform_name_cn);const features=[...new Set(REPORT.featurePlatform.map(d=>d.feature))];const map=new Map(REPORT.featurePlatform.map(d=>[d.platform_name_cn+'|'+d.feature,d.ratio]));const data=[];platforms.forEach((p,x)=>features.forEach((f,y)=>data.push([x,y,+((map.get(p+'|'+f)||0)*100).toFixed(1)])));
    return withBase({grid:{left:8,right:16,top:10,bottom:72,containLabel:true},xAxis:axisCategory({type:'category',data:platforms,axisLabel:{rotate:32,fontSize:10}}),yAxis:axisCategory({type:'category',data:features}),visualMap:{min:0,max:50,orient:'horizontal',left:'center',bottom:0,inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,50),label:{show:true,formatter:p=>p.value[2]>=1?p.value[2].toFixed(0)+'%':'',fontSize:9},itemStyle:{borderColor:'#fff',borderWidth:1}}]});
  });
  register('chartTopPages',()=>withBase({grid:{left:16,right:20,top:14,bottom:38,containLabel:true},xAxis:axisValue({name:'覆盖问题数',nameLocation:'middle',nameGap:28}),yAxis:axisValue({name:'覆盖平台数',min:0,max:12,nameGap:35}),series:[{type:'scatter',data:REPORT.topPages.map(d=>[d.question_count,d.platform_count,d.deduplicated_citation_count,d.page_title,d.domain]),symbolSize:d=>10+Math.sqrt(d[2])*1.7,itemStyle:{color:C.brand,opacity:.7},tooltip:{formatter:p=>`<b>${esc(short(p.value[3],48))}</b><br>${esc(p.value[4]||'')}<br>问题 ${p.value[0]} · 平台 ${p.value[1]} · 引用 ${n(p.value[2])}`}}]}));
  register('chartFeatureCombos',()=>{
    const rows=[...REPORT.featureCombinations].sort((a,b)=>a.avg_questions-b.avg_questions).slice(-14);
    return withBase({grid:{left:8,right:54,top:8,bottom:8,containLabel:true},xAxis:axisValue({name:'平均覆盖问题数'}),yAxis:axisCategory({type:'category',data:rows.map(d=>d.combination)}),series:[{type:'bar',barWidth:15,data:rows.map(d=>+d.avg_questions.toFixed(2)),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>p.value.toFixed(2),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.combination)}</b><br>页面 ${n(d.page_count)}<br>平均问题 ${fmt1.format(d.avg_questions)}<br>平均平台 ${fmt1.format(d.avg_platforms)}`}}}]});
  });
  register('chartYears',()=>{
    const rows=REPORT.yearDistribution.filter(d=>d.publication_year>=2005);
    return withBase({grid:{left:12,right:18,top:16,bottom:36,containLabel:true},xAxis:axisCategory({type:'category',data:rows.map(d=>d.publication_year),axisLabel:{interval:Math.max(0,Math.floor(rows.length/10)-1)}}),yAxis:axisValue({name:'页面数'}),series:[{type:'bar',data:rows.map(d=>d.page_count),barMaxWidth:30,itemStyle:{color:d=>rows[d.dataIndex].publication_year>=2025?C.brand:C.mid},tooltip:{formatter:p=>`${p.name} 年<br>页面 ${n(p.value)}<br>去重引用 ${n(rows[p.dataIndex].dedup_count)}`}}]});
  });
  register('chartFreshness',()=>{
    const knownBands=['2026','2025','2023 至 2024','2022 及以前'];const order=[...knownBands,'发布时间未知','发布时间冲突'];const rows=order.map(name=>REPORT.freshness.find(d=>d.freshness_band===name)||{freshness_band:name,page_count:0,dedup_count:0});const total=rows.reduce((sum,d)=>sum+d.page_count,0);const knownTotal=rows.filter(d=>knownBands.includes(d.freshness_band)).reduce((sum,d)=>sum+d.page_count,0);
    return withBase({grid:{left:8,right:118,top:8,bottom:8,containLabel:true},xAxis:axisValue({show:false}),yAxis:axisCategory({type:'category',inverse:true,data:rows.map(d=>d.freshness_band)}),series:[{type:'bar',barWidth:20,data:rows.map((d,i)=>({value:d.page_count,itemStyle:{color:state.freshness==='ALL'||state.freshness===d.freshness_band?(knownBands.includes(d.freshness_band)?palette[i]:C.mid):C.mid,opacity:state.freshness==='ALL'||state.freshness===d.freshness_band?1:.42}})),label:{show:true,position:'right',formatter:p=>{const d=rows[p.dataIndex];const known=knownBands.includes(d.freshness_band)?' · 已知时间 '+pct(d.page_count/knownTotal,0):'';return n(d.page_count)+' · 全部 '+pct(d.page_count/total,0)+known;},fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];const known=knownBands.includes(d.freshness_band)?'<br>已知时间样本占比 '+pct(d.page_count/knownTotal):'';return `<b>${esc(d.freshness_band)}</b><br>页面 ${n(d.page_count)}<br>去重引用 ${n(d.dedup_count)}<br>全部页面占比 ${pct(d.page_count/total)}${known}`}}}]});
  });
  register('chartLabelFreshness',()=>{
    const allBands=['2026','2025','2023 至 2024','2022 及以前','发布时间未知','发布时间冲突'];const bands=state.freshness==='ALL'?allBands:[state.freshness];const labels=REPORT.labelStats.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label)).map(d=>d.label_cn);const rows=REPORT.labelFreshness.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label));const map=new Map(rows.map(d=>[d.freshness_band+'|'+d.label_cn,d.page_links]));const totals=new Map(labels.map(label=>[label,allBands.reduce((sum,band)=>sum+(map.get(band+'|'+label)||0),0)]));const data=[];bands.forEach((band,x)=>labels.forEach((label,y)=>{const count=map.get(band+'|'+label)||0;data.push([x,y,+(count/Math.max(1,totals.get(label))*100).toFixed(1),count]);}));
    return withBase({grid:{left:8,right:15,top:8,bottom:64,containLabel:true},xAxis:axisCategory({type:'category',data:bands,axisLabel:{rotate:bands.length>4?18:0}}),yAxis:axisCategory({type:'category',data:labels}),visualMap:{min:0,max:100,orient:'horizontal',left:'center',bottom:0,inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,100),label:{show:true,formatter:p=>p.value[2]>=1?p.value[2].toFixed(0)+'%':'',fontSize:9},tooltip:{formatter:p=>`<b>${esc(labels[p.value[1]])}</b><br>${esc(bands[p.value[0]])}<br>占该标签 ${p.value[2].toFixed(1)}%<br>问题与页面关联 ${n(p.value[3])}`},itemStyle:{borderColor:'#fff',borderWidth:1}}]});
  });
  register('chartSourceFreshness',()=>{
    const totals=new Map();REPORT.sourceTypeFreshness.forEach(d=>totals.set(d.source_type_cn,(totals.get(d.source_type_cn)||0)+d.page_count));const types=[...totals.entries()].sort((a,b)=>b[1]-a[1]).slice(0,12).map(x=>x[0]).reverse();const allBands=['2026','2025','2023 至 2024','2022 及以前','发布时间未知','发布时间冲突'];const bands=state.freshness==='ALL'?allBands:[state.freshness];
    return withBase({aria:{enabled:true,description:'主要信源类型按页面发布时间状态展示内部占比。当前时间口径为'+(state.freshness==='ALL'?'全部发布时间状态':state.freshness)+'。'},legend:{top:0,textStyle:{fontSize:9,color:C.stone}},grid:{left:8,right:18,top:38,bottom:8,containLabel:true},xAxis:axisValue({min:0,max:100,axisLabel:{formatter:'{value}%'}}),yAxis:axisCategory({type:'category',data:types}),series:bands.map((band,i)=>({name:band,type:'bar',stack:'fresh',data:types.map(type=>{const d=REPORT.sourceTypeFreshness.find(x=>x.source_type_cn===type&&x.freshness_band===band);const count=d?d.page_count:0;return {value:+(count/Math.max(1,totals.get(type))*100).toFixed(1),raw:count};}),itemStyle:{color:allBands.indexOf(band)>=4?C.mid:palette[allBands.indexOf(band)]},tooltip:{formatter:p=>`<b>${esc(p.name)}</b><br>${esc(p.seriesName)} ${p.value.toFixed(1)}%<br>页面 ${n(p.data.raw)}`}}))});
  });
  register('chartTitleYear',()=>withBase({legend:{bottom:0,textStyle:{fontSize:10,color:C.stone}},series:[{type:'pie',radius:['38%','68%'],center:['50%','43%'],data:REPORT.titleYearQuality.map((d,i)=>({name:d.status,value:d.page_count,itemStyle:{color:palette[i]}})),label:{formatter:p=>p.percent>=3?p.name+'\n'+p.percent.toFixed(1)+'%':'',fontSize:10,color:C.warm},itemStyle:{borderColor:'#fff',borderWidth:2}}]}));
  register('chartSourceQuadrant',()=>{
    const rows=REPORT.sourceQuadrant;
    return withBase({grid:{left:16,right:24,top:22,bottom:42,containLabel:true},xAxis:axisValue({name:'问题覆盖率',min:0,max:100,nameLocation:'middle',nameGap:30,axisLabel:{formatter:'{value}%'}}),yAxis:axisValue({name:'平台渗透率',min:0,max:100,nameGap:35,axisLabel:{formatter:'{value}%'}}),series:[{type:'scatter',data:rows.map(d=>[+(d.question_count/REPORT.overview.questions*100).toFixed(2),+(d.platform_count/REPORT.overview.platforms*100).toFixed(2),d.dedup_count,d.source_name,d.source_type_cn,d.question_count,d.platform_count]),symbolSize:d=>Math.max(6,Math.min(42,5+Math.sqrt(d[2])/3)),itemStyle:{color:C.brand,opacity:.55},markLine:{silent:true,symbol:['none','none'],lineStyle:{color:C.line,type:'dashed'},data:[{xAxis:+(40/REPORT.overview.questions*100).toFixed(2)},{yAxis:50}],label:{show:false}},tooltip:{formatter:p=>`<b>${esc(p.value[3])}</b><br>${esc(p.value[4]||'未分类')}<br>问题覆盖率 ${p.value[0].toFixed(1)}%（${n(p.value[5])} / ${n(REPORT.overview.questions)}）<br>平台渗透率 ${p.value[1].toFixed(1)}%（${n(p.value[6])} / ${n(REPORT.overview.platforms)}）<br>去重引用 ${n(p.value[2])}`}}]});
  });
  register('chartContentGaps',()=>{
    const rows=REPORT.labelStats.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label));
    return withBase({grid:{left:16,right:24,top:18,bottom:42,containLabel:true},xAxis:axisValue({name:'每问题页面数（供给）',nameLocation:'middle',nameGap:30}),yAxis:axisValue({name:'每问题引用数（需求）',nameGap:44}),series:[{type:'scatter',data:rows.map(d=>[d.page_count/d.question_count,d.dedup_count/d.question_count,d.question_count,d.label_cn]),symbolSize:d=>12+Math.sqrt(d[2])*4,itemStyle:{color:C.brand,opacity:.72},label:{show:true,position:'top',formatter:p=>p.value[3],fontSize:10,color:C.warm},tooltip:{formatter:p=>`<b>${esc(p.value[3])}</b><br>每问题页面 ${fmt1.format(p.value[0])}<br>每问题引用 ${fmt1.format(p.value[1])}`}}]});
  });
  register('chartFormatOpportunity',()=>{
    const formats=['general','ranking','guide','comparison','unknown'];const labels={general:'通用',ranking:'榜单',guide:'指南',comparison:'对比',unknown:'标题未提供'};const platforms=REPORT.platforms.map(d=>d.platform_name_cn);const rows=REPORT.formatPlatform.filter(d=>d.platform_code!=='ALL');const map=new Map(rows.map(d=>[d.platform_code+'|'+d.content_format_hint,d.dedup_count/Math.max(1,d.page_count)]));const data=[];REPORT.platforms.forEach((p,y)=>formats.forEach((f,x)=>data.push([x,y,+((map.get(p.platform_code+'|'+f)||0).toFixed(2))])));const max=Math.max(...data.map(d=>d[2]));
    return withBase({grid:{left:8,right:18,top:10,bottom:64,containLabel:true},xAxis:axisCategory({type:'category',data:formats.map(f=>labels[f])}),yAxis:axisCategory({type:'category',data:platforms}),visualMap:{min:0,max,orient:'horizontal',left:'center',bottom:0,inRange:{color:heatColors},textStyle:{color:C.stone}},series:[{type:'heatmap',data:heatCells(data,max),label:{show:true,formatter:p=>p.value[2]?p.value[2].toFixed(1):'',fontSize:9},itemStyle:{borderColor:'#fff',borderWidth:1}}]});
  });
  register('chartExpansion',()=>{
    const rows=REPORT.expansionCandidates.slice(0,18).reverse();
    return withBase({grid:{left:8,right:122,top:8,bottom:8,containLabel:true},xAxis:axisValue({name:'筛选分'}),yAxis:axisCategory({type:'category',data:rows.map(d=>short(d.source_name,20))}),series:[{type:'bar',barWidth:14,data:rows.map(d=>d.screening_score),itemStyle:{color:C.brand},label:{show:true,position:'right',formatter:p=>compact(p.value)+' · 空白 '+pct(rows[p.dataIndex].potential_platforms/REPORT.overview.platforms,0),fontSize:9,color:C.warm},tooltip:{formatter:p=>{const d=rows[p.dataIndex];return `<b>${esc(d.source_name)}</b><br>问题覆盖率 ${pct(d.question_count/REPORT.overview.questions)}（${n(d.question_count)} / ${n(REPORT.overview.questions)}）<br>平台渗透率 ${pct(d.platform_count/REPORT.overview.platforms)}（${n(d.platform_count)} / ${n(REPORT.overview.platforms)}）<br>平台空白率 ${pct(d.potential_platforms/REPORT.overview.platforms)}（${n(d.potential_platforms)} / ${n(REPORT.overview.platforms)}）<br>去重引用 ${n(d.dedup_count)} · 筛选分 ${n(d.screening_score)}`}}}]});
  });

  function fillTable(id,rows,cells,rowClass){
    const body=document.getElementById(id);body.textContent='';
    if(!rows.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=cells.length;td.textContent='当前筛选条件下暂无记录';tr.appendChild(td);body.appendChild(tr);return;}
    rows.forEach(row=>{const tr=document.createElement('tr');if(rowClass)tr.className=rowClass(row)||'';cells.forEach(cell=>{const td=document.createElement('td');const result=cell(row);td.textContent=result.text;td.className=result.className||'';if(result.title)td.title=result.title;tr.appendChild(td);});body.appendChild(tr);});
  }
  function preferenceIsStable(d){
    const larger=Math.max(d.web_share,d.mobile_share,1e-9);const relativeGap=Math.abs(d.mobile_share-d.web_share)/larger;
    return relativeGap<=.15;
  }
  function preferenceAdvice(d){
    if(d.terminal_preference_index>=150&&d.source_question_count<15)return '高倾向低覆盖';
    if(preferenceIsStable(d))return '双端稳定布局';
    return d.delta_pp>0?'移动端专项验证':'电脑端专项验证';
  }
  function updatePreferenceTable(){
    let rows=REPORT.terminalTilt.filter(d=>state.preferenceProduct==='ALL'||d.product_family===state.preferenceProduct);
    const sortRows=(a,b)=>Number(b.source_id===state.selectedPreferenceSource)-Number(a.source_id===state.selectedPreferenceSource)||b.priority_score-a.priority_score;
    if(state.preferenceProduct==='ALL'){
      rows=REPORT.terminalPairSummary.flatMap(pair=>rows.filter(d=>d.product_family===pair.product_family).sort(sortRows).slice(0,5));
    }else rows=rows.sort(sortRows).slice(0,20);
    fillTable('preferenceTable',rows,[
      d=>({text:preferenceSourceLabel(d,36),className:'truncate',title:d.source_name+(d.domain?' · '+d.domain:'')}),
      d=>({text:d.product_family,className:'nowrap'}),
      d=>({text:preferenceIsStable(d)?'双端接近':(d.delta_pp>0?'移动端':'电脑端'),className:'preference-tag nowrap'}),
      d=>({text:fmt1.format(d.terminal_preference_index),className:'num'}),
      d=>({text:(d.delta_pp>=0?'+':'')+d.delta_pp.toFixed(2)+'pp',className:'num preference-optional'}),
      d=>({text:n(d.source_question_count),className:'num preference-optional'}),
      d=>({text:pct(d.source_question_count/Math.max(1,d.common_question_count)),className:'num preference-optional'}),
      d=>({text:n(d.total_citations),className:'num preference-optional'}),
      d=>({text:preferenceAdvice(d),className:'nowrap'})
    ],d=>d.source_id===state.selectedPreferenceSource?'is-selected':'');
  }
  function updateTables(){
    let sourceRows=REPORT.labelTopSources.filter(d=>d.label_dimension===state.dimension&&(state.label==='ALL'||d.label_value===state.label));
    if(state.label==='ALL')sourceRows=sourceRows.sort((a,b)=>b.dedup_count-a.dedup_count).slice(0,10);else sourceRows=sourceRows.slice(0,8);
    fillTable('labelSourceTable',sourceRows,[d=>({text:d.source_name,className:'truncate',title:d.source_name}),d=>({text:d.label_cn,className:'nowrap'}),d=>({text:n(d.dedup_count),className:'num'}),d=>({text:n(d.question_count),className:'num'})]);
    let whitespace;
    if(state.platform==='ALL')whitespace=REPORT.expansionCandidates.slice(0,8).map(d=>({source_name:d.source_name,source_type_cn:d.source_type_cn,platform_count:d.platform_count,question_count:d.question_count}));
    else whitespace=REPORT.whitespaceSources.filter(d=>d.platform_code===state.platform).slice(0,8);
    fillTable('whitespaceTable',whitespace,[d=>({text:d.source_name,className:'truncate',title:d.source_name}),d=>({text:d.source_type_cn||'未分类',className:'nowrap'}),d=>({text:n(d.platform_count),className:'num'}),d=>({text:n(d.question_count),className:'num'})]);
  }
  function updateFindings(){
    const o=REPORT.overview;const top10=(REPORT.sourcePareto.find(d=>d.rank===10)||{}).cumulative_share||0;
    const maxOverlap=[...REPORT.overlap].sort((a,b)=>b.jaccard_similarity-a.jaccard_similarity)[0];
    const zeroPairs=REPORT.overlap.filter(d=>d.jaccard_similarity===0).length;
    const classification=REPORT.classificationCoverage.find(d=>d.platform_code==='ALL');
    const classifiedShare=classification.classification_coverage;
    const currentPages=REPORT.freshness.filter(d=>['2026','2025'].includes(d.freshness_band)).reduce((s,d)=>s+d.page_count,0);
    const featureYear=REPORT.featurePrevalence.find(d=>d.feature==='标题含年份');
    const pairRows=[...REPORT.terminalPairSummary].sort((a,b)=>a.source_jaccard-b.source_jaccard);const pairLow=pairRows[0];const pairHigh=pairRows[pairRows.length-1];
    const qualifiedJaccards=pairRows.map(d=>d.qualified_source_jaccard);
    const anchorEndpointMap=new Map(REPORT.anchorSourceMigration.map(d=>[d.platform_code,d]));
    const anchorCarryover=[...REPORT.preferenceMeta.anchor_top20_carryover].sort((a,b)=>a.source_count-b.source_count);const anchorLow=anchorCarryover[0];const anchorHigh=anchorCarryover[anchorCarryover.length-1];
    const anchorLabel=item=>{const d=anchorEndpointMap.get(item.platform_code);return preferenceFamilyLabel(d.product_family)+preferenceTerminalLabel(d.terminal);};
    const anchorHighLabels=anchorCarryover.filter(item=>item.source_count===anchorHigh.source_count).map(anchorLabel).join('、');
    const findings=[
      ['精确重复约占 '+pct((o.raw_citations-o.dedup_citations)/o.raw_citations),'默认使用去重口径，可以减少重复采集对平台与信源规模的放大。'],
      ['发布时间元数据覆盖 '+pct(o.publication_metadata_available/o.raw_citations),'字段未知的记录继续参与平台、问题和引用观察统计；时间分析应用独立样本边界。'],
      ['平台问题覆盖从 '+n(Math.min(...REPORT.platforms.map(d=>d.question_count)))+' 到 '+n(Math.max(...REPORT.platforms.map(d=>d.question_count))),'跨平台比较应优先选择覆盖范围相近的平台与问题集合。'],
      ['前 10 个信源贡献 '+pct(top10),'信源生态具有头部集中，长尾仍提供主题与平台差异。'],
      ['八端共同问题等权口径下，可识别信源分类覆盖率 '+pct(classifiedShare),'高贡献域名人工复核与确定性规则已经覆盖主要等权份额，下一步可按贡献继续扩展长尾分类。'],
      ['最高平台对相似度 '+pct(maxOverlap.jaccard_similarity),'当前最高组合为 '+maxOverlap.platform_a_name+' 与 '+maxOverlap.platform_b_name+'，其余多数平台对保持较低重合。'],
      [zeroPairs+' 个平台对没有共享问题与页面组合','平台独有供给具有研究价值，跨平台迁移需要逐平台验证。'],
      ['可靠时间页面中 2025 和 2026 占 '+pct(currentPages/o.dated_pages),'这两个年份占全部规范页面 '+pct(currentPages/o.pages)+'；另有 '+n(o.unknown_date_pages)+' 个页面时间未知、'+n(o.conflicting_date_pages)+' 个页面日期冲突。'],
      ['标题含年份页面 '+n(featureYear.page_count),'年份信号是最常见的显式标题特征，适合进入页面更新与一致性审计。'],
      ['四个产品双端全量信源重合度为 '+pct(pairLow.source_jaccard)+' 至 '+pct(pairHigh.source_jaccard),'最低为 '+pairLow.product_family+'，最高为 '+pairHigh.product_family+'。按筛选门槛计算后，重合度为 '+pct(Math.min(...qualifiedJaccards))+' 至 '+pct(Math.max(...qualifiedJaccards))+'，长尾信源是两种口径差距的主要来源。'],
      ['核心锚点在八端 Top 20 的继承率为 '+pct(anchorLow.source_count/REPORT.preferenceMeta.anchor_pool_size)+' 至 '+pct(anchorHigh.source_count/REPORT.preferenceMeta.anchor_pool_size),anchorLabel(anchorLow)+'继承率较低，'+anchorHighLabels+'并列较高。该差异来自 '+n(REPORT.preferenceMeta.common_scope_question_count)+' 个共同问题，可用于安排跨平台迁移验证。'],
      ['机会队列同步展示覆盖率与空白率','信源扩展保留筛选分排序，并补充问题覆盖率、平台渗透率和平台空白率，便于人工复核。']
    ];
    const root=document.getElementById('findingList');root.textContent='';findings.forEach((item,i)=>{
      const li=document.createElement('li');const index=document.createElement('span');index.className='finding-n';index.textContent=String(i+1).padStart(2,'0');
      const body=document.createElement('span');body.className='finding-body';const title=document.createElement('b');title.textContent=item[0];const detail=document.createElement('span');detail.textContent=item[1];
      body.append(title,detail);li.append(index,body);root.appendChild(li);
    });
  }

  function initNavigation(){
    const sections=[...document.querySelectorAll('.report-section')];const links=[...document.querySelectorAll('.nav a')];const linkMap=new Map(links.map(a=>[a.getAttribute('href').slice(1),a]));
    const observer=new IntersectionObserver(entries=>{entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio).slice(0,1).forEach(e=>{links.forEach(a=>a.removeAttribute('aria-current'));const link=linkMap.get(e.target.id);if(link){link.setAttribute('aria-current','true');link.scrollIntoView({block:'nearest',inline:'center'});}});},{rootMargin:'-20% 0px -65% 0px',threshold:[0,.1,.3]});sections.forEach(s=>observer.observe(s));
    const progress=document.getElementById('progress');let ticking=false;window.addEventListener('scroll',()=>{if(ticking)return;ticking=true;requestAnimationFrame(()=>{const max=document.documentElement.scrollHeight-innerHeight;progress.style.width=(max>0?scrollY/max*100:0)+'%';if(scrollY+innerHeight*.2<sections[0].offsetTop)links.forEach(a=>a.removeAttribute('aria-current'));ticking=false;});},{passive:true});
  }
  function initResize(){ let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>charts.forEach(c=>c.resize()),120);},{passive:true}); }

  validatePreferenceLinkage();initMeta();initFilters();initNavigation();initResize();requestAnimationFrame(drawAll);
})();
