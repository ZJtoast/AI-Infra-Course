(() => {
  "use strict";
  const data = document.getElementById("aira-cards");
  const panel = document.querySelector("#reader-toc, #catalog-toc, .side nav");
  if (!data || !panel) return;
  const cards = JSON.parse(data.textContent);
  if (!Object.keys(cards).length) return;
  const script=document.currentScript, assets=new URL(".",script.src);
  const artVersion=script.dataset.artVersion;
  function resource(file) {
    const url=new URL(file,assets);
    if(artVersion && file.endsWith(".webp"))url.searchParams.set("v",artVersion);
    return url.href;
  }
  const pacing={poke:38,tinyheart:55,hearthands:52};
  const dailyActions=["seal","sip","spin","starhug","stretch","sweep","watch","water","wave","whisk","yawn","plane","polish","pour","bell","bow","bubbles","sketch","stack","study","think"];
  const dailyRate=0.6;
  const forward=new Set(["hearthands","tinyheart"]);
  const transitionMs=760,breathMs=24*158,support=313/320;
  let visualScale=1,centerEyesUntil=0;
  const calm = matchMedia("(prefers-reduced-motion: reduce)");
  const widget = document.createElement("div");
  widget.className = "aira";
  widget.setAttribute("role","region");
  widget.setAttribute("aria-label","Aira");
  widget.innerHTML = `<button class="aira-toggle" type="button" aria-label="Aira" aria-describedby="aira-controls">
      <span class="aira-art" aria-hidden="true"></span><span class="aira-eyes" aria-hidden="true" hidden></span>
      <span class="aira-label" hidden>Aira</span></button>
    <span class="aira-sr" id="aira-controls">轻点互动，双击表演，拖动移动。右键、Shift+F10 或触屏长按切换姿势。方向键移动，Home 回到右下角。K 查看知识，Q 查看问题，D 挥掸子，F 表演飞踢，S 或 L 到组件边缘探头。</span>
    <div class="aira-bubble" aria-label="Aira 小节提示" hidden>
      <p class="aira-message" aria-hidden="true"></p><span class="aira-sr" role="status" aria-live="polite" aria-atomic="true"></span>
      <div class="aira-actions"><button type="button" data-action="content"></button><button type="button" data-action="dismiss">收起</button></div>
    </div>`;
  document.body.append(widget);
  const find = selector => widget.querySelector(selector);
  const toggle = find(".aira-toggle"), art = find(".aira-art");
  const message = find(".aira-message"), status = find('[role="status"]');
  function petImage(name) {
    const image=new Image();image.className=name;image.alt="";image.draggable=false;
    return image;
  }
  let duster=petImage("aira-duster"),poster=petImage("aira-poster");
  const bubble=find(".aira-bubble"),eyes=find(".aira-eyes");
  const content = find('[data-action="content"]');
  const stored = key => {try {return localStorage.getItem(key);} catch {return null;}};
  function save(key,value) {try {localStorage.setItem(key,value);} catch {}}
  const bound = (v,lo,hi) => Math.max(lo, Math.min(v,hi));
  let resting = stored("aira-rest") === "yes", key = null, mode = "fact", automatic = false;
  let gaze = -1, active = null, tick, generation = 0, drag = null;
  let promptTimer, dismiss, writing, layoutFrame, gazeFrame, lastPoint, alternate = false;
  let anchor = {x:1,y:1};
  try {
    const saved = JSON.parse(stored("aira-position-v1"));
    if (Number.isFinite(saved?.x) && Number.isFinite(saved?.y))
      anchor = {x:bound(saved.x,0,1),y:bound(saved.y,0,1)};
  } catch {}
  function viewport() {
    const v = window.visualViewport;
    return {x:v?.offsetLeft || 0, y:v?.offsetTop || 0,
      w:Math.min(document.documentElement.clientWidth,v?.width || Infinity),
      h:v?.height || document.documentElement.clientHeight};
  }
  function range() {
    const view=viewport(),box=toggle.getBoundingClientRect(),w=box.width,h=box.height;
    if(resting)return {left:view.x+8,top:view.y+8,dx:Math.max(0,view.w-w-16),dy:Math.max(0,view.h-h-16)};
    const unit=w/(visualScale*288),reach=AiraFraming.reach;
    const left=view.x+8+reach.left*unit-w/2,top=view.y+8+reach.above*unit-h*support;
    return {left,top,dx:Math.max(0,view.w-16-(reach.left+reach.right)*unit),
      dy:Math.max(0,view.h-16-(reach.above+reach.below)*unit)};
  }
  function moveTo(x,y) {
    const r = range();
    anchor = {x:r.dx ? bound((x-r.left)/r.dx,0,1) : 0,y:r.dy ? bound((y-r.top)/r.dy,0,1) : 0};
    layout();
  }
  function layout() {
    if (!widget.getClientRects().length) return;
    const r = range();
    const staged=scene?.position||heldScene?.position;
    widget.style.left=(staged?.x??(r.left+anchor.x*r.dx))+"px";
    const petY=r.top+anchor.y*r.dy;
    widget.style.top=(staged?.y??petY)+"px";
    widget.style.bottom = widget.style.right = "auto";
    const floating=bubble;
    if (floating.hidden) return;
    const v = viewport(), b = toggle.getBoundingClientRect(), edge=8, gap=6;
    const left=v.x+edge, top=v.y+edge, right=v.x+v.w-edge, bottom=v.y+v.h-edge;
    floating.style.width = Math.min(320,Math.max(1,right-left))+"px";
    const width=floating.offsetWidth;
    const petTop=b.top;
    const above=petTop-gap-top, below=bottom-b.bottom-gap, room=Math.max(above,below);
    const rightSpace=right-b.right-gap, leftSpace=b.left-left-gap;
    const side=room<Math.min(floating.scrollHeight+2,270) && Math.max(rightSpace,leftSpace)>=width;
    floating.style.maxHeight=Math.max(1,Math.min(416,side ? bottom-top : room))+"px";
    const height=floating.offsetHeight;
    const x=side ? (rightSpace>=leftSpace?b.right+gap:b.left-gap-width) : b.right-width;
    const y=side ? (b.top+b.bottom-height)/2 : above>=below ? petTop-gap-height : b.bottom+gap;
    floating.style.left=Math.round(bound(x,left,right-width))+"px";
    floating.style.top=Math.round(bound(y,top,bottom-height))+"px";
  }
  function scheduleLayout() {
    sceneChange();centerPaper();
    if (!layoutFrame) layoutFrame=requestAnimationFrame(()=>{layoutFrame=0;layout();});
  }
  const textures=new Map(),resident=new Map();
  function load(file) {
    if(resident.has(file)){
      const image=resident.get(file);resident.delete(file);resident.set(file,image);return Promise.resolve();
    }
    if(!textures.has(file)){
      const task=(async()=>{
        const image=new Image();image.src=resource(file);
        if(image.decode)await image.decode();
        else if(!image.complete)await new Promise((ok,fail)=>{image.onload=ok;image.onerror=fail;});
        resident.set(file,image);
        if(resident.size>8){
          const current="poses/"+bodyPose+".webp";
          const oldest=[...resident.keys()].find(key=>key!==current && key!==art.dataset.file);
          resident.delete(oldest);
        }
      })().finally(()=>textures.delete(file));
      textures.set(file,task);
    }
    return textures.get(file);
  }
  function scaleVisual(factor){
    if(factor===visualScale)return;
    if(scene)scene.changed=true;
    visualScale=factor;widget.style.setProperty("--aira-pose-scale",String(factor));layout();
  }
  function showCanvas(){
    releaseHeldScene();
    if(!duster.hidden || !poster.hidden){duster.hidden=true;poster.hidden=true;duster.remove();poster.remove();duster.removeAttribute("src");poster.removeAttribute("src");}
    art.hidden=resting;
  }
  function tile(file,index,cols,rows) {
    showCanvas();
    const code=file.slice("poses/".length,-".webp".length);
    scaleVisual(AiraFraming.scale(code,index));
    if(art.dataset.file===file && art.dataset.frame===String(index))return;
    if (art.dataset.file!==file) {
      if(art.dataset.file==="poses/plane.webp"){art.style.clipPath="";art.style.transform="";}
      art.dataset.file=file;art.style.backgroundImage='url("'+resource(file)+'")';
      art.style.backgroundSize=cols*100+"% "+rows*100+"%";
    }
    art.style.backgroundPosition=(cols===1?0:index%cols*100/(cols-1))+"% "+
      (rows===1?0:Math.floor(index/cols)*100/(rows-1))+"%";
    art.dataset.frame=String(index);
  }
  const postureNames = new Map([
    ["breathe","站着"], ["idle_sit","盘坐"], ["idle_side","侧卧"],
    ["idle_prone","趴着"], ["idle_kneel","跪坐"], ["idle_sway","站着轻摇"],
    ["idle_tokens","盘坐拨 token 珠"], ["idle_pipeline","拨动流水灯"]
  ]);
  const postureEdges = [
    ["breathe","idle_kneel",""], ["idle_kneel","idle_sit",""],
    ["idle_sit","idle_side","sit_side"], ["idle_side","idle_prone","side_prone"],
    ["breathe","idle_sway",""], ["idle_sit","idle_tokens",""], ["breathe","idle_pipeline",""]
  ];
  const directions=["e","se","s","sw","w","nw","n","ne"];
  const faces={breathe:[51,19],idle_sway:[52,20],idle_sit:[52,38],idle_side:[24,67],
    idle_prone:[33,70],idle_kneel:[50,32],idle_tokens:[52,38],idle_pipeline:[51,20]};
  const standing=id=>["breathe","idle_sway","idle_pipeline"].includes(id);
  const reactionFor=id=>standing(id)?"poke":id==="idle_tokens"?"react_sit":id.replace("idle_","react_");
  let warmPose="",pendingEye="";
  let choice=stored("aira-posture"),bodyPose=postureNames.has(choice)?choice:"breathe";
  if(!postureNames.has(choice))choice="auto";
  let homePose=bodyPose,previousHome="",journey=null,goal=homePose,arrive=null;
  let bodyTimer=0,lastBodyTime=0,breathTime=0,heldTime=0,phase=0;
  function layer(element,file,index) {
    const pos=(index%5*25)+"% "+(Math.floor(index/5)*25)+"%";
    if(element.dataset.file!==file){
      element.style.backgroundImage='url("'+resource(file)+'")';
      element.style.backgroundSize="500% 500%";element.dataset.file=file;
    }
    if(element.style.backgroundPosition!==pos)element.style.backgroundPosition=pos;
  }
  function canBreathe(){
    return (!active || active.loading) && (!journey || !journey.ready);
  }
  function renderBody() {
    if(!canBreathe() || resting)return;
    const index=calm.matches?0:phase;
    widget.dataset.posture=bodyPose;if(!journey)delete widget.dataset.transition;
    tile("poses/"+bodyPose+".webp",index,5,5);
    eyes.style.clipPath="";
    const eyeFile=gaze<0 || performance.now()<centerEyesUntil?"":"poses/"+bodyPose+"-"+directions[gaze]+".webp";
    if(!eyeFile){eyes.hidden=true;pendingEye="";}
    else if(eyes.dataset.file===eyeFile || resident.has(eyeFile)){
      eyes.hidden=false;layer(eyes,eyeFile,index);pendingEye="";
    }else if(pendingEye!==eyeFile){
      pendingEye=eyeFile;
      if(!eyes.dataset.file?.startsWith("poses/"+bodyPose+"-"))eyes.hidden=true;
      load(eyeFile).then(()=>{if(pendingEye===eyeFile){pendingEye="";renderBody();}}).catch(()=>{});
    }
    if(goal===bodyPose && warmPose!==bodyPose){
      const id=warmPose=bodyPose;
      load("poses/"+id+".webp").then(()=>{
        const warm=()=>{
          if(bodyPose!==id || document.hidden)return;
          load("poses/"+reactionFor(id)+".webp").catch(()=>{});
          const next=poseOrder[(poseOrder.indexOf(id)+1)%poseOrder.length],edge=firstEdge(id,next);
          load("poses/"+next+".webp").catch(()=>{});
          if(edge.file)load("poses/"+edge.file+".webp").catch(()=>{});
        };
        if(window.requestIdleCallback)window.requestIdleCallback(warm,{timeout:500});else setTimeout(warm,40);
      }).catch(()=>{});
    }
  }
  function near(pose) {
    return postureEdges.filter(([a,b])=>a===pose || b===pose).map(([a,b,file])=>
      ({to:pose===a?b:a,file,reverse:pose!==a}));
  }
  function firstEdge(from,to){
    return near(from).find(edge=>edge.to===to)||{to,file:"",reverse:false};
  }
  let directOnly=false;
  function settle(edge,at){
    if(journey!==edge)return;
    journey=null;bodyPose=at;widget.dataset.posture=at;
    phase=0;breathTime=0;lastBodyTime=performance.now();
    delete widget.dataset.transition;
    renderBody();
    advance();
  }
  function steer(edge){
    if(!edge.ready || journey!==edge)return;
    cancelAnimationFrame(edge.raf);
    const first=firstEdge(edge.from,goal),toward=goal!==edge.from && first?.to===edge.to;
    const target=toward?1:0,initial=edge.progress,start=performance.now();
    const span=Math.abs(target-initial),filename="poses/"+edge.file+".webp";
    function step(){
      if(journey!==edge)return;
      const done=!span || performance.now()-start>=span*transitionMs;
      edge.progress=done?target:initial+Math.sign(target-initial)*(performance.now()-start)/transitionMs;
      const frame=Math.round(edge.progress*24);edge.frame=edge.reverse?24-frame:frame;
      eyes.hidden=true;
      if(art.dataset.file!==filename || Number(art.dataset.frame)!==edge.frame)tile(filename,edge.frame,5,5);
      if(done){settle(edge,toward?edge.to:edge.from);return;}
      edge.raf=requestAnimationFrame(step);
    }
    step();
  }
  function advance(){
    if(journey || active || resting || document.hidden)return;
    if(bodyPose===goal && !directOnly){const done=arrive;arrive=null;if(done)done();return;}
    const edge=directOnly?{to:goal,file:"",reverse:false}:firstEdge(bodyPose,goal);
    directOnly=false;
    journey=edge;edge.from=bodyPose;edge.progress=0;edge.frame=edge.reverse?24:0;
    widget.dataset.transition=edge.file||"same-posture";
    const retain=()=>{if(journey===edge){goal=edge.from;arrive=null;settle(edge,edge.from);}};
    if(!edge.file || calm.matches){
      load("poses/"+edge.to+".webp").then(()=>{
        if(journey===edge)settle(edge,edge.to);
      }).catch(retain);
      return;
    }
    Promise.all([load("poses/"+edge.file+".webp"),load("poses/"+edge.to+".webp")]).then(()=>{
      if(journey!==edge)return;
      edge.ready=true;steer(edge);
    }).catch(retain);
  }
  function go(pose,done=null,cut=false){
    clearTimeout(returnTimer);returnTimer=0;
    goal=pose;arrive=done;
    if(journey && !cut && [journey.from,journey.to].includes(pose)){
      if(journey.file && journey.ready)steer(journey);
      else if(journey.to!==pose){cancelAnimationFrame(journey.raf);journey=null;directOnly=true;advance();}
      return;
    }
    if(journey){cancelAnimationFrame(journey.raf);journey=null;cut=true;}
    directOnly=cut;advance();
  }
  function stopTravel(){
    clearTimeout(returnTimer);returnTimer=0;
    if(journey){cancelAnimationFrame(journey.raf);journey=null;}
    directOnly=false;
    goal=bodyPose;arrive=null;widget.dataset.posture=bodyPose;delete widget.dataset.transition;
  }

  function setPosture(value,persist){
    if(resting)rest(false,true);
    if(persist){choice=value;save("aira-posture",value);}
    previousHome=homePose;homePose=value==="auto"?bodyPose:value;heldTime=0;
    const interrupted=!!active;if(interrupted)stopMotion(true);go(homePose,null,interrupted);
  }
  // Keep a short natural blink inside a longer breath. Sitting has two closed-eye drawings.
  function closedPhase(id){return id==="idle_sit"?11:id==="idle_prone"?12:15;}
  const phaseEnds=new Map([...postureNames.keys()].map(id=>{
    const blink=id==="idle_sit"?{10:62,11:56,12:58,13:64}:id==="idle_prone"?{11:62,12:56,13:58,14:64}:{14:65,15:78,16:67};
    const spans=Object.values(blink),other=(breathMs-spans.reduce((a,b)=>a+b,0))/(24-spans.length);
    let end=0;return [id,Array.from({length:24},(_,i)=>end+=blink[i]??other)];
  }));
  function bodyClock(){
    clearTimeout(bodyTimer);lastBodyTime=0;
    if(resting || document.hidden || calm.matches)return;
    lastBodyTime=performance.now();
    function step(){
      const now=performance.now(),dt=Math.min(now-lastBodyTime,550);lastBodyTime=now;
      let wait=140;
      if(canBreathe()){
        breathTime+=dt;heldTime+=dt;
        const moment=breathTime%breathMs,ends=phaseEnds.get(bodyPose);
        const next=ends.findIndex(end=>end>moment);
        wait=Math.max(8,ends[next]-moment+1);
        if(next!==phase){phase=next;renderBody();}
        if(choice!=="auto" && heldTime>=36000 && bodyPose===choice)choice="auto";
        if(!active && !journey && choice==="auto" && heldTime>=17000 && !busy() && bubble.hidden){
          const adjacent=near(homePose).map(e=>e.to), alternatives=adjacent.filter(p=>p!==previousHome);
          const options=alternatives.length?alternatives:adjacent;
          setPosture(options[Math.floor(Math.random()*options.length)],false);
        }
      }
      bodyTimer=setTimeout(step,wait);
    }
    bodyTimer=setTimeout(step,120);
  }
  const performances={
    duster:{file:"maid-duster",duration:6880,label:"羽毛掸子"},
    flying_squirrel:{file:"maid-flying-squirrel",duration:7040,label:"挑衅飞踢"}
  };
  const movies=new Map(),playbackSlot=new Map();
  function movieBytes(url){
    if(!movies.has(url)){
      const bytes=fetch(url).then(response=>{
        if(!response.ok)throw Error("动画下载失败");
        return response.blob();
      }).catch(error=>{movies.delete(url);throw error;});
      movies.set(url,bytes);
    }
    return movies.get(url);
  }
  // Each envelope covers both neighbouring time intervals in the original transparent animation.
  const inkBounds={"duster":{"width":720,"height":672,"step":160,"boxes":[[272,220,453,602],[272,220,453,602],[272,220,454,602],[272,216,457,602],[266,213,458,602],[141,213,553,605],[141,213,553,606],[141,213,553,606],[147,213,553,606],[147,180,553,607],[142,142,558,612],[137,138,561,616],[135,138,563,617],[134,138,564,617],[130,154,568,620],[130,204,599,620],[130,210,599,620],[130,184,599,620],[130,184,576,619],[117,184,563,614],[117,197,565,614],[117,161,566,615],[119,161,566,615],[119,161,566,615],[119,125,565,614],[123,94,564,614],[116,94,562,614],[116,90,561,613],[116,84,560,612],[129,84,559,611],[130,84,559,611],[125,100,562,610],[113,107,563,610],[113,154,563,610],[113,154,585,610],[113,152,585,612],[125,152,585,612],[126,152,611,612],[125,137,611,612],[125,137,613,611],[125,137,613,611],[126,178,613,609],[127,178,613,608],[127,178,612,608],[127,212,562,608]],"still":[151,219,550,603]},"flying_squirrel":{"width":720,"height":672,"step":160,"boxes":[[303,236,451,555],[303,232,463,567],[307,226,481,571],[289,218,522,571],[278,204,529,571],[275,202,558,571],[275,202,565,571],[275,202,565,573],[275,195,565,573],[296,195,562,573],[336,195,524,573],[348,195,517,573],[343,201,506,572],[343,209,499,571],[343,212,490,570],[344,216,472,566],[340,216,470,561],[330,209,469,563],[317,209,467,563],[308,205,464,563],[295,201,454,563],[281,201,448,562],[277,201,434,559],[277,201,423,557],[277,203,416,551],[281,220,441,547],[283,220,441,543],[285,220,443,543],[304,217,450,543],[306,216,457,544],[306,212,471,544],[307,206,494,555],[307,190,506,561],[319,173,521,563],[310,160,521,573],[290,134,532,582],[289,134,543,594],[289,134,547,594],[289,134,552,594],[289,134,552,594],[279,134,552,594],[279,136,552,588],[279,151,537,578],[286,173,498,568],[290,200,449,557],[290,213,435,557]],"still":[278,205,528,566]}};
  function placePerformance(motion,image,still){
    const profile=inkBounds[motion.code];
    if(!profile || image.naturalWidth!==profile.width)return;
    const foot=AiraFraming.films[performances[motion.code].file].foot;
    const started=performance.now(),offsetX=144-foot[0],offsetY=313-foot[1];
    Object.assign(image.style,{left:offsetX/288*100+"%",top:offsetY/320*100+"%",right:"auto",bottom:"auto",
      width:profile.width/288*100+"%",height:profile.height/320*100+"%"});
    function place(){
      if(active!==motion)return;
      const t=performance.now()-started,part=Math.min(profile.boxes.length-2,Math.floor(t/profile.step));
      const f=Math.min(1,(t-part*profile.step)/profile.step);
      const bounds=still?profile.still:profile.boxes[part].map((v,i)=>v+(profile.boxes[part+1][i]-v)*f);
      const b=toggle.getBoundingClientRect(),v=viewport(),scale=b.width/288;
      const left=b.left+(offsetX+bounds[0])*scale,right=b.left+(offsetX+bounds[2])*scale;
      const top=b.top+(offsetY+bounds[1])*scale,bottom=b.top+(offsetY+bounds[3])*scale;
      const x=Math.max(0,v.x+2-left)+Math.min(0,v.x+v.w-2-right);
      const y=Math.max(0,v.y+2-top)+Math.min(0,v.y+v.h-2-bottom);
      const transform="translate("+Math.round(x)+"px, "+Math.round(y)+"px)";
      if(image.style.transform!==transform)image.style.transform=transform;
      motion.raf=requestAnimationFrame(place);
    }
    place();
  }
  let performanceTimer=0;
  function perform(code,requested=false){
    cancelExpression();
    if(resting || document.hidden || drag?.moving)return;
    if(requested)cancelTap();
    if(active)stopMotion(true);
    go("breathe",()=>begin({code,requested}));
  }
  function schedulePerformance(){
    clearTimeout(performanceTimer);
    if(calm.matches || resting || document.hidden)return;
    performanceTimer=setTimeout(()=>{
      if(!busy() && !active && !journey && !pendingExpression && bubble.hidden)
        begin({code:nextDailyAction(),returnPose:bodyPose});
      schedulePerformance();
    },80000+Math.random()*60000);
  }
  const poseOrder=["breathe","idle_sway","idle_pipeline","idle_kneel","idle_sit","idle_tokens","idle_side","idle_prone"];
  let expressionTimer=0,pendingExpression=null;
  function cancelExpression(){
    clearTimeout(expressionTimer);pendingExpression=null;
  }
  function cyclePose(){
    if(resting)rest(false,true);
    cancelTap();cancelExpression();
    const from=postureNames.has(goal)?goal:bodyPose;
    const target=poseOrder[(poseOrder.indexOf(from)+1)%poseOrder.length];
    const name=drawFromBag("pose-expressions",["wink","look-around","look-down"]);
    const views=name==="look-around"?["w","e"]:name==="look-down"?["s","se"]:[];
    const request=pendingExpression={target,name,arrived:false,deadline:performance.now()+3000+Math.random()*2000};
    function deliver(){
      if(pendingExpression!==request || !request.arrived)return;
      const remaining=request.deadline-performance.now();
      if(remaining>0){clearTimeout(expressionTimer);expressionTimer=setTimeout(deliver,remaining+1);return;}
      if(drag){expressionTimer=setTimeout(deliver,80);return;}
      if(active || journey || bodyPose!==target)return;
      pendingExpression=null;showExpression(name);
    }
    const interrupted=!!active;if(interrupted)stopMotion(true);
    choice=target;homePose=target;heldTime=0;save("aira-posture",target);
    expressionTimer=setTimeout(deliver,Math.max(0,request.deadline-performance.now())+1);
    go(target,()=>{
      request.arrived=true;
      for(const dir of views)load("poses/"+target+"-"+dir+".webp").catch(()=>{});
      deliver();
    },interrupted);
  }
  function showExpression(expression){
    const blink=closedPhase(bodyPose);
    const opening=blink+(bodyPose==="idle_sit" || bodyPose==="idle_prone"?2:1);
    const poses=expression==="wink"?[[blink-1,75],[blink,120],[opening,75]]:
      expression==="look-around"?[["w",240],["e",320],["w",120]]:
        [["s",330],[blink,110],["se",230]];
    let i=0;const [x,y]=faces[bodyPose];active={code:"expression",reaction:true};
    widget.dataset.expression=expression;
    function draw(){
      const [item,ms]=poses[i];eyes.hidden=false;
      if(typeof item==="number"){
        layer(eyes,"poses/"+bodyPose+".webp",item);eyes.style.clipPath="ellipse(20% 16% at "+x+"% "+y+"%)";
      }else{
        layer(eyes,"poses/"+bodyPose+"-"+item+".webp",phase);eyes.style.clipPath="";
      }
      i++;
      tick=setTimeout(i===poses.length || calm.matches?finish:draw,calm.matches?330:ms);
    }
    draw();
  }

  const standingPokes=["poke","tinyheart","hearthands"],localPokes=["blink","lean","rub"];
  const firstRecoilFrame={lean:2},repeats={rub:[14,18]};
  let pendingPoke="",lastImpact=0;
  function poke(group){
    if(typeof group!=="string")group="gentle";
    widget.dataset.touchPattern=group;
    if(active?.reaction){
      if(group!==active.group || performance.now()-lastImpact>320)pendingPoke=group;
      return;
    }
    if(journey){go(journey.progress<.5?journey.from:journey.to,()=>poke(group));return;}
    const returnPose=journey?goal:active?.returnPose||bodyPose;
    stopTravel();if(active)stopMotion(true);
    const prop=bodyPose==="idle_pipeline" || bodyPose==="idle_tokens";
    const seated=!standing(bodyPose) || prop;
    const variant=drawFromBag(seated?"local-poke":"standing-poke",seated?localPokes:standingPokes);
    const code=seated?reactionFor(bodyPose):variant;lastImpact=performance.now();
    if(prop && art.dataset.file==="poses/"+bodyPose+".webp" && !art.hidden)propPoke(group,variant);
    else begin({code,group,variant,reaction:true,requested:true,returnPose});
    if(!prop && !art.hidden && art.dataset.file==="poses/"+bodyPose+".webp" && !resident.has("poses/"+code+".webp")){
      const [x,y]=faces[bodyPose];
      layer(eyes,"poses/"+bodyPose+".webp",closedPhase(bodyPose));
      eyes.style.clipPath="ellipse(20% 16% at "+x+"% "+y+"%)";eyes.hidden=false;
    }
  }
  function propPoke(group,variant){
    const id=bodyPose,action=id==="idle_tokens"?"react_tokens":"react_pipeline";
    active={code:action,reaction:true,group};
    art.dataset.action=action;art.dataset.pokeGroup=group;art.dataset.pokeVariant=variant;
    const [x,y]=faces[id],file="poses/"+id+".webp";
    const phases=variant==="rub"?[14,15,16,14,15,16]:[14,15,16];
    const beat={blink:57,lean:82,rub:76}[variant];
    eyes.style.clipPath="ellipse(22% 16% at "+x+"% "+y+"%)";eyes.hidden=false;
    let step=0;
    function blink(){
      if(step===phases.length){finish();return;}
      layer(eyes,file,calm.matches?15:phases[step++]);
      if(calm.matches){tick=setTimeout(finish,350);return;}
      tick=setTimeout(blink,beat);
    }
    blink();
  }

  function neutral() {
    art.hidden=resting; duster.hidden=true; poster.hidden=true;duster.remove();poster.remove();
    duster.removeAttribute("src");poster.removeAttribute("src");
    art.style.transition="";art.style.opacity="";duster.style.opacity="";
    delete art.dataset.action;
    art.style.clipPath="";art.style.transform="";
    eyes.hidden=true;renderBody();
    toggle.title=resting?"唤醒 Aira":"Aira · 轻点互动，右键换姿势";
  }
  function stopMotion(holdFrame=false) {
    if(scene && holdFrame){
      cancelAnimationFrame(scene.raf);clearTimeout(scene.timer);heldScene=scene;scene=null;
    }else if(scene)leaveScene();
    pendingPoke="";delete art.dataset.pokeGroup;delete art.dataset.pokeVariant;
    clearTimeout(tick);if(active?.raf)cancelAnimationFrame(active.raf);
    if(active?.objectURL)URL.revokeObjectURL(active.objectURL);
    generation++;active=null;delete widget.dataset.expression;
    if(!holdFrame)neutral();
  }
  function finish() {
    const ended=active,epoch=generation,pose=ended?.code==="flying_squirrel"?"breathe":ended?.returnPose||bodyPose;
    function land(){
      if(epoch!==generation || active!==ended)return;
      const further=pendingPoke;
      if(ended?.code==="flying_squirrel"){
        centerEyesUntil=performance.now()+240;homePose="breathe";choice="auto";heldTime=0;save("aira-posture","auto");
      }else if(!ended?.reaction && choice==="auto" && homePose!==pose){
        previousHome=homePose;homePose=pose;heldTime=0;
      }
      if(!ended?.reaction || pose!==bodyPose){phase=0;breathTime=0;lastBodyTime=performance.now();}
      bodyPose=pose;goal=pose;widget.dataset.posture=pose;stopMotion();
      if(further)poke(further);
      else if(choice!=="auto" && choice!==pose)scheduleReturn();
    }
    const file="poses/"+pose+".webp";
    if(resident.has(file))land();else load(file).then(land).catch(land);
  }
  let returnTimer=0;
  function scheduleReturn(){
    clearTimeout(returnTimer);
    returnTimer=setTimeout(function returnHome(){
      if(drag){returnTimer=setTimeout(returnHome,350);return;}
      returnTimer=0;
      if(!active && !journey && !resting && !document.hidden && choice!=="auto")go(choice);
    },7200);
  }
  function begin(intent) {
    const {code}=intent;
    if (resting || document.hidden) return;
    clearTimeout(tick);const epoch=++generation;
    const motion=active={code,loading:true,reduced:calm.matches,frame:0,direction:1,looped:false,reaction:!!intent.reaction,group:intent.group,returnPose:intent.returnPose||bodyPose};
    const file="poses/"+code+".webp";
    const daily=dailyActions.includes(code);
    const pace=(pacing[code]||40)/(daily?dailyRate:1);
    const local=code.startsWith("react_"),entry=local?(firstRecoilFrame[intent.variant]||0):0;
    const ready=()=>epoch===generation && active===motion;
    const finishIfCurrent=()=>{if(ready()) finish();};
    const show=(frame)=>{
      motion.loading=false;eyes.hidden=true;eyes.style.clipPath="";
      if(intent.reaction){art.dataset.pokeGroup=intent.group;art.dataset.pokeVariant=intent.variant;}
      motion.frame=frame;art.hidden=false;art.dataset.action=code;
      tile(file,frame,5,5);toggle.title="Aira · 轻点互动，右键换姿势";
      if(code==="plane"){
        if(frame===0){
          const b=toggle.getBoundingClientRect(),v=viewport();
          motion.facing=b.left+b.width/2>v.x+v.w/2?-1:1;
        }
        art.style.transform="scaleX("+motion.facing+")";
        art.style.clipPath=[15,17,20,22,23,24].includes(frame)?"polygon(0 0,75% 0,75% 23%,100% 23%,100% 100%,0 100%)":"";
        if(frame===15)throwPlane(motion);
      }
    };
    const native=performances[code];
    if(native){
      load("poses/"+(code==="flying_squirrel"?"breathe":motion.returnPose)+".webp").catch(()=>{});
      const still=calm.matches,image=petImage(still?"aira-poster":"aira-duster"),filename=native.file+(still?"-still":"")+".webp";
      const url=new URL(resource(filename));
      async function prepare(){
        if(!still && /^https?:$/.test(url.protocol)){
          const bytes=await movieBytes(url.href);
          if(!ready())return;
          motion.objectURL=URL.createObjectURL(bytes);image.src=motion.objectURL;
        }else{
          if(!still){
            const slot=1-(playbackSlot.get(code)??1);
            playbackSlot.set(code,slot);url.searchParams.set("play",String(slot));
          }
          image.src=url.href;
        }
        await image.decode();
        if(!ready())return;
        motion.loading=false;
        releaseHeldScene();eyes.hidden=true;scaleVisual(AiraFraming.scale(native.file));
        duster.remove();poster.remove();duster.removeAttribute("src");poster.removeAttribute("src");
        if(still)poster=image;else duster=image;
        image.dataset.performance=code;
        toggle.append(image);image.hidden=false;art.hidden=true;placePerformance(motion,image,still);layout();
        art.dataset.action=code;art.dataset.frame="0";toggle.title=native.label;
        tick=setTimeout(finishIfCurrent,still?850:native.duration);
      }
      prepare().catch(finishIfCurrent);
      return;
    }
    (code==="plane"?Promise.all([load(file),load("maid-plane.webp")]):load(file)).then(()=>{
      if(!ready())return;
      if (calm.matches) {
        show(12);tick=setTimeout(finishIfCurrent,850);return;
      }
      function step() {
        if (!ready()) return;
        const f=motion.frame,repeat=local && repeats[intent.variant];
        if(repeat && !motion.looped && f===repeat[1]){
          motion.direction=-1;motion.looped=true;
        }else if(repeat && motion.direction<0 && f<=repeat[0])motion.direction=1;
        if (motion.direction>0 && f===24) {
          if(daily || local || forward.has(code)){finish();return;}
          motion.direction=-1;
        }
        if(motion.direction<0 && f<=entry){finish();return;}
        const next=f+motion.direction;show(next);
        let beat=pace;
        if(intent.reaction && next<6 && motion.direction>0)beat=24;
        if(intent.reaction && next>=15)beat+=5;
        if(next===24 && !daily)beat=pace*3;
        tick=setTimeout(step,beat);
      }
      motion.step=step;show(entry);
      tick=setTimeout(step,intent.reaction?20:pace);
    }).catch(finishIfCurrent);
  }
  const bags=new Map();
  function drawFromBag(id,lines){
    if(!Array.isArray(lines) || !lines.length)return "";
    let bag=bags.get(id);
    if(!bag?.length){
      const last=bag?.last;bag=shuffle(lines);
      if(bag.length>1 && bag.at(-1)===last)[bag[0],bag[bag.length-1]]=[bag.at(-1),bag[0]];
      bags.set(id,bag);
    }
    const line=bag.pop();bag.last=line;return line;
  }
  // Keep the rest of this round when the reader opens the next lesson in the same tab.
  function nextDailyAction(){
    const id="aira-daily-round-v1";
    if(!bags.has(id)){
      try{
        const saved=JSON.parse(sessionStorage.getItem(id));
        if(saved && dailyActions.includes(saved.last) && Array.isArray(saved.left) &&
           new Set(saved.left).size===saved.left.length && saved.left.every(code=>dailyActions.includes(code) && code!==saved.last)){
          const bag=saved.left;bag.last=saved.last;bags.set(id,bag);
        }
      }catch{}
    }
    const code=drawFromBag(id,dailyActions);
    try{sessionStorage.setItem(id,JSON.stringify({last:code,left:bags.get(id)}));}catch{}
    return code;
  }
  function close(){
    clearTimeout(writing);clearTimeout(dismiss);automatic=false;bubble.hidden=true;
  }
  function speak(kind,auto=false) {
    if (resting || !key || !cards[key]) return;
    mode=kind;automatic=auto;clearTimeout(writing);clearTimeout(dismiss);
    const text=cards[key][kind], letters=Array.from(text);
    content.textContent=kind==="fact"?"考考我":kind==="question"?"看答案":"看知识";
    status.textContent=text;message.textContent="";
    bubble.hidden=false;layout();
    let end=calm.matches?letters.length:0;
    function write() {
      end=Math.min(end+2,letters.length);
      message.textContent=letters.slice(0,end).join("");scheduleLayout();
      if(end<letters.length) writing=setTimeout(write,38);
      else if(auto) dismiss=setTimeout(()=>{
        if (!widget.matches(":hover") && !widget.contains(document.activeElement)) close();
      },17000);
    }
    write();
  }
  const drawnQuestions=new Set();
  let lastQuestion="",paper=null;
  function lessonGroup(id){
    const lesson=id.match(/^(?:l(\d+)-s\d+|ch-(\d+))$/);
    if(lesson)return "lesson-"+Number(lesson[1]||lesson[2]);
    return document.getElementById(id)?.closest("section[data-part]")?.dataset.part || "";
  }
  function chooseQuestion(){
    const ids=Object.keys(cards);
    let eligible=ids.filter(id=>!drawnQuestions.has(cards[id].question));
    if(!eligible.length){
      drawnQuestions.clear();
      eligible=ids.filter(id=>cards[id].question!==lastQuestion);
      if(!eligible.length)eligible=ids;
    }
    const group=lessonGroup(key||"");
    const related=group?eligible.filter(id=>lessonGroup(id)===group):eligible;
    const pool=related.length?related:eligible;
    const id=pool[Math.floor(Math.random()*pool.length)];
    const card=cards[id],section=document.getElementById(id);
    const heading=section?.matches("h1,h2,h3,h4")?section:section?.querySelector("h1,h2,h3,h4");
    return {id,question:card.question,answer:card.answer,title:heading?.textContent.trim()||""};
  }
  function clearPaper(restore=false){
    if(!paper)return;
    const p=paper;paper=null;cancelAnimationFrame(p.raf);
    clearTimeout(p.expiry);clearTimeout(p.reveal);
    const hadFocus=p.card.contains(document.activeElement);
    p.flight.remove();p.card.remove();
    if(restore && hadFocus)toggle.focus({preventScroll:true});
  }
  function centerPaper(){
    if(!paper)return;
    const v=viewport(), card=paper.card;
    card.style.left=(v.x+v.w/2)+"px";card.style.top=(v.y+v.h/2)+"px";
    card.style.width=Math.min(350,v.w-24)+"px";card.style.maxHeight=Math.max(60,v.h-24)+"px";
  }
  function throwPlane(motion) {
    if(paper || document.hidden || busy())return;
    const q=chooseQuestion(),flight=petImage("aira-flight"),card=document.createElement("div");
    flight.src=resource("maid-plane.webp");flight.setAttribute("aria-hidden","true");
    card.className="aira-plane-card";card.hidden=true;
    card.setAttribute("role","dialog");card.setAttribute("aria-labelledby","aira-paper-question");
    card.innerHTML=`<div class="aira-plane-head"><span></span><button type="button" data-paper="close" aria-label="收起纸飞机题目">关闭</button></div>
      <h3 id="aira-paper-question"></h3><p class="aira-plane-answer" aria-live="polite" hidden></p>
      <button type="button" data-paper="answer" aria-expanded="false">看答案</button>`;
    card.querySelector("h3").textContent=q.question;
    card.querySelector(".aira-plane-head span").textContent=q.title || "Aira 的纸飞机";
    const answerButton=card.querySelector('[data-paper="answer"]'),answer=card.querySelector(".aira-plane-answer");
    const p=paper={flight,card,q,raf:0,expiry:0,reveal:0};
    card.dataset.questionKey=q.id;
    card.querySelector('[data-paper="close"]').onclick=()=>clearPaper(true);
    answerButton.onclick=()=>{
      clearTimeout(p.expiry);
      answer.textContent=q.answer;answer.hidden=false;
      answerButton.setAttribute("aria-expanded","true");answerButton.hidden=true;
      if(card.contains(document.activeElement))card.querySelector('[data-paper="close"]').focus({preventScroll:true});
      centerPaper();
    };
    const manual=motion.requested || widget.contains(document.activeElement);
    widget.append(flight,card);
    const b=toggle.getBoundingClientRect(),scale=b.width/144,dir=motion.facing;
    // Frame 15 is the first released sheet. Attach the tail to the extended fingertip.
    const fingerX=b.left+(dir===1?109/144:1-109/144)*b.width;
    const fingerY=b.top+39/160*b.height;
    const sx=fingerX+dir*12*scale,sy=fingerY-5*scale;
    const started=performance.now(),duration=calm.matches?0:810;
    flight.style.width=(27*scale)+"px";flight.style.height=(15*scale)+"px";
    function expire() {
      if(paper!==p)return;
      if(card.matches(":hover") || card.contains(document.activeElement)){
        p.expiry=setTimeout(expire,2500);return;
      }
      clearPaper();
    }
    function unfold(){
      // Only questions that actually opened count toward a reading session's draw.
      drawnQuestions.add(q.question);lastQuestion=q.question;
      flight.remove();centerPaper();card.hidden=false;
      card.dataset.state="unfolding";
      p.reveal=setTimeout(()=>{
        if(paper!==p)return;
        card.dataset.state="open";
        if(manual)answerButton.focus({preventScroll:true});
      },calm.matches?0:260);
      p.expiry=setTimeout(expire,manual?22000:12000);
    }
    function fly(now) {
      if(paper!==p)return;
      const v=viewport(),tx=v.x+v.w/2,ty=v.y+v.h/2;
      const t=duration?Math.min(1,(now-started)/duration):1, u=1-t;
      const cx=sx+(tx-sx)*.44, cy=Math.max(v.y+12,Math.min(sy,ty)-Math.min(96,Math.hypot(tx-sx,ty-sy)*.24));
      const x=u*u*sx+2*u*t*cx+t*t*tx,y=u*u*sy+2*u*t*cy+t*t*ty;
      const tangent=Math.atan2(u*(cy-sy)+t*(ty-cy),u*(cx-sx)+t*(tx-cx))*180/Math.PI;
      const angle=tangent-(dir===1?-25:-155);
      flight.style.left=x+"px";flight.style.top=y+"px";
      flight.style.transform="translate(-50%,-50%) rotate("+angle+"deg) scaleX("+dir+")";
      if(t===1){unfold();return;}
      p.raf=requestAnimationFrame(fly);
    }
    close();fly(started);
  }


  let scene=null,heldScene=null,sceneTimer=0;
  function releaseHeldScene(){
    if(!heldScene)return;
    heldScene=null;toggle.style.clipPath="";art.style.transform="";
    if(!scene){delete widget.dataset.stage;delete widget.dataset.edge;delete widget.dataset.exposed;}
    layout();
  }
  const edgeInk = {
    ledgehide:[[40.5,155.5],[42.0,155.5],[45.5,155.5],[46.5,155.5],[57.5,155.5],[58.5,155.5],[59.0,155.5],[71.5,155.5],[72.5,155.5],[80.5,155.5],[81.5,155.5],[91.0,155.5],[92.0,155.5],[96.0,155.5],[96.0,155.5],[96.0,155.5],[94.5,155.5],[91.5,155.5],[90.5,155.0],[90.0,155.0],[77.5,155.0],[76.5,155.0],[64.5,155.0],[63.0,155.0],[51.0,155.0]],
    sidehide:[[28.5,108.0],[27.0,106.5],[26.0,105.5],[24.5,104.5],[23.5,102.5],[24.5,103.0],[25.5,104.0],[21.5,97.5],[22.5,97.5],[26.0,93.5],[27.0,93.0],[34.0,95.5],[34.0,95.5],[40.0,88.5],[40.5,89.5],[32.0,88.5],[32.5,88.5],[26.0,95.5],[27.0,95.0],[27.5,95.5],[28.5,101.5],[27.0,102.0],[26.5,97.0],[26.0,97.0],[28.5,90.0]]
  };
  const sceneActions = new Set(["sidehide","ledgehide"]);
  const surface = "aside, #catalog-toc, .articlehero, .homehero, .intro, .metrics > *, .cataloghead, " +
    ".tablewrap, .table-wrap, .table-scroll, pre, table, th, figure, .equation, .formula, .lab, .practice-route, " +
    "main h1, main h2, main h3, main summary, main button, main a, main section.topic, article a[class], main code, header nav, header button, header a, header select";
  function visibleRect(el) {
    if(!el?.isConnected || !el.getClientRects().length || widget.contains(el)) return null;
    const sidebar=el.closest("aside, #catalog-toc");
    if(sidebar && (sidebar.closest("[hidden]") || sidebar.querySelector(":scope > details:not([open])") ||
      sidebar.getAttribute("aria-hidden")==="true")) return null;
    if(el.closest("details:not([open])") && !el.matches("summary")) return null;
    const style=getComputedStyle(el);
    if(style.visibility!=="visible" || style.opacity==="0") return null;
    const b=el.getBoundingClientRect(),v=viewport();
    let left=Math.max(v.x,b.left),top=Math.max(v.y,b.top),right=Math.min(v.x+v.w,b.right),bottom=Math.min(v.y+v.h,b.bottom);
    if(right-left<30 || bottom-top<19)return null;
    for(let p=el.parentElement;p && p!==document.body;p=p.parentElement) {
      const s=getComputedStyle(p);
      if(s.overflowY==="visible" && s.overflowX==="visible")continue;
      const a=p.getBoundingClientRect();
      if(s.overflowY!=="visible"){top=Math.max(top,a.top);bottom=Math.min(bottom,a.bottom);}
      if(s.overflowX!=="visible"){left=Math.max(left,a.left);right=Math.min(right,a.right);}
    }
    if(right-left<30 || bottom-top<19) return null;
    return {b,left,right,top,bottom};
  }
  function geometry(el,code) {
    const c=visibleRect(el);
    if(!c)return null;
    const v=viewport(),w=toggle.offsetWidth,h=toggle.offsetHeight, pad=8;
    if(code==="sidehide"){
      if(c.bottom-c.top<Math.min(h*.3,32))return null;
      const rightFits=Math.abs(c.right-c.b.right)<2 && c.right+Math.max(17,w*.14) <= v.x+v.w-3 &&
        c.right-w*.67>=v.x+pad;
      const leftFits=Math.abs(c.left-c.b.left)<2 && c.left-Math.max(17,w*.14) >=v.x+3 &&
        c.left+w*.67<=v.x+v.w-pad;
      if(!rightFits&&!leftFits)return null;
      const direction=rightFits?1:-1;
      const cy=bound(c.top+(c.bottom-c.top)*.5,v.y+pad,v.y+v.h-pad);
      const free=direction===1?v.x+v.w-3-c.right:c.left-v.x-3;
      return {edge:direction===1?c.right:c.left,along:cy-.27*h,direction,cap:Math.min(.4,free/w),clipTop:c.top,clipBottom:c.bottom};
    }
    if(Math.abs(c.top-c.b.top)>2 || c.right-c.left<w*.52 ||
        c.top-v.y<Math.max(20,h*.16) || c.top>v.y+v.h-pad) return null;
    return {edge:c.top,along:bound((c.left+c.right-w)/2,v.x+pad,v.x+v.w-pad-w),direction:0,cap:Math.min(.4,(c.top-v.y-3)/h)};
  }
  function findSurface(code) {
    const v=viewport(),set=new Set();
    if(code==="sidehide")for(const s of document.querySelectorAll("aside, #catalog-toc"))set.add(s);
    for(const yy of [.10,.17,.33,.51,.68,.84,.95,.985])for(const xx of [.12,.34,.56,.78,.92]){
      const layers=document.elementsFromPoint(v.x+v.w*xx,v.y+v.h*yy);
      for(const el of layers){
        if(widget.contains(el))continue;
        for(let p=el.closest(surface);p;p=p.parentElement?.closest(surface))set.add(p);
      }
    }
    for(const e of document.querySelectorAll(".articlehero, .homehero, .metrics > *, .intro, .articlehero .btn, .homehero a, header button, header select, header a, header nav, .side summary"))set.add(e);
    const midSection=document.elementFromPoint(v.x+v.w*.5,v.y+v.h*.56)?.closest("section");
    if(midSection) {
      const nearby=midSection.querySelectorAll("h2,h3,pre,figure,.tablewrap,.table-wrap,table,.equation,.formula,summary,code,.btn");
      for(let i=0;i<Math.min(nearby.length,230);i++)set.add(nearby[i]);
    }
    const here=toggle.getBoundingClientRect(), middle=here.top+here.height/2;
    let best=null,bestScore=-Infinity;
    for(const el of set){
      const g=geometry(el,code);if(!g)continue;
      const b=el.getBoundingClientRect();
      const visibleTop=Math.max(v.y,b.top),visibleBottom=Math.min(v.y+v.h,b.bottom);
      const center=(visibleTop+visibleBottom)/2;
      const atViewportEdge=center<v.y+v.h*.09 || center>v.y+v.h*.9;
      const score=(el.matches("aside,#catalog-toc") && code==="sidehide"?1000:0)+
        (el.matches("pre,table,.tablewrap,.table-wrap,.table-scroll,figure,.articlehero,.homehero,.intro,.equation,.lab")?400:0)+
        (code==="sidehide"?(g.cap-.4)*1050:0)+Math.min(160,visibleBottom-visibleTop)+Math.min(100,b.width*.15)-
        Math.abs(center-middle)*.15-(atViewportEdge?420:0);
      if(score>bestScore){bestScore=score;best=el;}
    }
    return best;
  }
  function leaveScene() {
    if(!scene) return;
    const previous=scene;
    cancelAnimationFrame(previous.raf);clearTimeout(previous.timer);
    scene=null;delete widget.dataset.stage;delete widget.dataset.edge;
    delete widget.dataset.exposed;delete art.dataset.action;toggle.style.clipPath="";art.style.transform="";
    layout();
  }
  function stageScene(intent) {
    const node=findSurface(intent.code);
    if(!node){finish();return;}
    const code=intent.code,file="poses/"+code+".webp",epoch=generation;
    const start=toggle.getBoundingClientRect();
    const s=scene={code,node,startX:start.left,startY:start.top,position:null,changed:true,raf:0};
    active={code,loading:true,returnPose:intent.returnPose||bodyPose};
    widget.dataset.stage=code;
    let walkReady=false;
    if(!calm.matches)load("poses/run.webp").then(()=>{walkReady=true;}).catch(()=>{});
    load(file).then(()=>{
      if(scene!==s || epoch!==generation) return;
      art.dataset.action=code;toggle.title=code==="sidehide"?"Aira 在侧边探头":"Aira 在上沿探头";
      let previousFrame=-1;
      const started=performance.now();let lastGeom=0;
      function draw(now){
        if(scene!==s) return;
        active.loading=false;releaseHeldScene();eyes.hidden=true;
        const elapsed=now-started;
        const walk=calm.matches?1:Math.min(1,elapsed/540);
        const phase=calm.matches?1:Math.max(0,(elapsed-540)/630);
        const breathe=calm.matches?0:Math.sin(Math.max(0,phase-1)*1.15);
        const frame=walk<1?Math.min(6,Math.floor(walk*6)):phase<1?6+Math.floor(phase*17):
          calm.matches?18:Math.round(18+6*breathe);
        const walking=walkReady && walk<1,picked=walking?6+Math.floor(walk*18):frame;
        const image=walking?"poses/run.webp":file;
        if(picked!==previousFrame || art.dataset.file!==image){tile(image,picked,5,5);previousFrame=picked;}
        if(s.changed || !s.geometry || now-lastGeom>250){
          let g=geometry(s.node,code);
          if(!g){s.node=findSurface(code);g=geometry(s.node,code);}
          if(!g){leaveScene();finish();return;}
          s.geometry=g;lastGeom=now;s.changed=false;
        }
        const g=s.geometry,w=toggle.offsetWidth,h=toggle.offsetHeight;
        const tuck=Math.min(1,phase);
        const peak=Math.min(.398,g.cap??.398);
        const excursion=Math.min(.043,Math.max(0,(peak-.14)/2));
        const show=1-tuck*(1-peak+excursion-excursion*breathe);
        const margin=100*(1-show);
        const dx=code==="ledgehide"?g.along:g.direction===1?g.edge-w*(1-show):g.edge-w*show;
        const dy=code==="ledgehide"?g.edge-h*show:g.along;
        const ease=walk*walk*(3-2*walk),vp=viewport();
        s.position={x:bound(s.startX+(dx-s.startX)*ease,vp.x+2,vp.x+vp.w-w-2),
                    y:bound(s.startY+(dy-s.startY)*ease,code==="ledgehide"?vp.y+2:vp.y-h*.27,
                      vp.y+vp.h-(code==="ledgehide"?h*show:h*.27)-2)};
        widget.dataset.edge=g.direction===0?"top":g.direction===1?"right":"left";
        widget.dataset.exposed=String(Math.round(show*100));
        toggle.style.clipPath=code==="ledgehide"?"inset(0 0 "+margin+"% 0)":
          g.direction===1?"inset(0 0 0 "+margin+"%)":"inset(0 "+margin+"% 0 0)";
        if(code==="sidehide" && walk===1) {
          const t=Math.max(0,g.clipTop-s.position.y),b=Math.max(0,s.position.y+h-g.clipBottom);
          toggle.style.clipPath=g.direction===1?"inset("+t+"px 0 "+b+"px "+margin+"%)":
            "inset("+t+"px "+margin+"% "+b+"px 0)";
        }

        const [low,high]=edgeInk[code][picked];
        if(!walking && code==="ledgehide"){
          const faceDepth=bound(.68+(show-.355)*1.8,.46,.81);
          const shift=(show*160-low-faceDepth*(high-low))*tuck/160*100;
          art.style.transform="translateY("+shift+"%)";
        } else if(!walking){
          const cheek=Math.min(.45,Math.max(0,show*144-3)/(high-low));
          const shift=((1-show)*144-high+cheek*(high-low))*tuck/144*100;
          art.style.transform="translateX("+g.direction*shift+"%) scaleX("+g.direction+")";
        } else art.style.transform="";
        layout();
        if(elapsed>12500){
          leaveScene();finish();return;
        }
        if(calm.matches){s.timer=setTimeout(()=>{s.raf=requestAnimationFrame(draw);},240);}
        else s.raf=requestAnimationFrame(draw);
      }
      s.raf=requestAnimationFrame(draw);
    }).catch(()=>{if(scene===s){leaveScene();finish();}});
  }
  function sceneChange() {
    if(scene) scene.changed=true;
  }
  document.addEventListener("scroll",sceneChange,{capture:true,passive:true});
  document.addEventListener("toggle",sceneChange,true);

  function hideAt(code){
    if(resting || document.hidden || drag?.moving)return;
    cancelTap();cancelExpression();
    if(active)stopMotion(true);
    go("breathe",()=>stageScene({code}));
  }
  function scheduleScene(){
    clearTimeout(sceneTimer);
    if(calm.matches || resting || document.hidden)return;
    sceneTimer=setTimeout(()=>{
      if(!active && !journey && !pendingExpression && !busy() && bubble.hidden)
        hideAt(drawFromBag("component-scenes",["sidehide","ledgehide"]));
      scheduleScene();
    },160000+Math.random()*85000);
  }
  function sync() {
    const selected=panel.querySelector('a[aria-current="location"]');
    let id=selected?.hash.slice(1);
    if(id) {try {id=decodeURIComponent(id);} catch {}}
    const next=Object.hasOwn(cards,id)?id:null;
    if(next===key) return;
    key=next;
    if(!key) close();
    else if(!bubble.hidden) speak(automatic?"fact":mode==="answer"?"question":mode,automatic);
    schedulePrompt();
  }
  function busy() {return !!drag || !!firstTap || !!paper || !!document.querySelector("dialog[open], html.toc-overlay");}
  function schedulePrompt() {
    clearTimeout(promptTimer);
    if(calm.matches || resting || document.hidden) return;
    promptTimer=setTimeout(()=>{
      if(!active && !journey && bubble.hidden && key && !busy() && !widget.contains(document.activeElement) &&
         !document.activeElement?.matches("input,textarea,select,[contenteditable=true]")) {
        speak(alternate?"question":"fact",true);alternate=!alternate;
      }
      schedulePrompt();
    },52000+Math.random()*30000);
  }
  function shuffle(items) {
    const bag=[...items];
    for(let i=bag.length-1;i>0;i--){
      const j=Math.floor(Math.random()*(i+1));[bag[i],bag[j]]=[bag[j],bag[i]];
    }
    return bag;
  }
  function rest(value,persist) {
    clearPaper();resting=value;widget.dataset.rest=value?"yes":"no";
    toggle.setAttribute("aria-label",value?"唤醒 Aira":"Aira");
    find(".aira-label").hidden=!value;stopTravel();stopMotion();close();layout();
    if(!value)go(homePose);bodyClock();schedulePrompt();schedulePerformance();scheduleScene();
    if(persist)save("aira-rest",value?"yes":"no");
  }
  function rememberPosition() {save("aira-position-v1",JSON.stringify(anchor));}
  let swallow=false,firstTap=null,tapTimer=0,burstUntil=0,touchMenuAt=-Infinity;
  function cancelTap(){clearTimeout(tapTimer);firstTap=null;}
  function touchGesture(x,y){
    const now=performance.now();
    if(now<burstUntil){burstUntil=now+320;return;}
    if(firstTap && now-firstTap.time<=320 && Math.hypot(x-firstTap.x,y-firstTap.y)<=24){
      cancelTap();burstUntil=now+320;
      perform(drawFromBag("double-performances",Object.keys(performances)),true);return;
    }
    cancelTap();firstTap={time:now,x,y};
    tapTimer=setTimeout(()=>{if(!firstTap)return;firstTap=null;if(!drag){cancelExpression();poke("gentle");}},322);
  }
  document.addEventListener("pointerdown",e=>{if(!toggle.contains(e.target))cancelTap();},{capture:true,passive:true});
  toggle.addEventListener("pointerdown",e=>{
    if(!e.isPrimary || e.button!==0 || drag) return;
    const r=toggle.getBoundingClientRect();
    const g=drag={id:e.pointerId,x:e.clientX,y:e.clientY,dx:e.clientX-r.left,dy:e.clientY-r.top,
      moving:false,long:false,canMove:!resting,threshold:e.pointerType==="touch"?9:6};
    swallow=false;
    if(e.pointerType==="touch" || e.pointerType==="pen")g.timer=setTimeout(()=>{
      if(drag!==g || g.moving) return;
      g.long=true;touchMenuAt=performance.now();cancelTap();cyclePose();
    },530);
    toggle.setPointerCapture(e.pointerId);
  });
  toggle.addEventListener("pointermove",e=>{
    if(!drag || drag.id!==e.pointerId || drag.long || !drag.canMove) return;
    if(!drag.moving && Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<drag.threshold) return;
    if(!drag.moving) {
      clearTimeout(drag.timer);drag.moving=true;cancelTap();widget.dataset.dragging="yes";
    }
    if(scene){leaveScene();stopMotion();}
    moveTo(e.clientX-drag.dx,e.clientY-drag.dy);
  });
  function release(e,cancelled) {
    if(!drag || drag.id!==e.pointerId) return;
    const g=drag;drag=null;clearTimeout(g.timer);delete widget.dataset.dragging;
    if(g.moving)rememberPosition();
    swallow=g.moving || g.long || cancelled;
    if(toggle.hasPointerCapture(e.pointerId))toggle.releasePointerCapture(e.pointerId);

  }
  toggle.addEventListener("pointerup",e=>release(e,false));
  toggle.addEventListener("pointercancel",e=>release(e,true));
  toggle.addEventListener("lostpointercapture",e=>release(e,true));
  toggle.addEventListener("contextmenu",e=>{
    e.preventDefault();if(drag?.moving)return;
    if(drag?.long || (e.pointerType==="touch" || e.sourceCapabilities?.firesTouchEvents) && performance.now()-touchMenuAt<700)return;
    if(drag){clearTimeout(drag.timer);drag.long=true;}
    cyclePose();
  });
  toggle.addEventListener("click",e=>{
    if(e.button!==0 || (e.detail && swallow)){swallow=false;return;}
    if(resting)rest(false,true);
    if(scene){cancelTap();stopMotion();return;}
    touchGesture(e.clientX,e.clientY);
  });
  const shifts={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
  toggle.addEventListener("keydown",e=>{
    if((e.shiftKey && e.key==="F10") || e.key==="ContextMenu"){e.preventDefault();cyclePose();return;}
    const k=e.key.toLowerCase();
    if(!e.ctrlKey && !e.altKey && !e.metaKey && ["d","f","k","q","s","l"].includes(k)){
      e.preventDefault();if(e.repeat)return;
      if(resting)rest(false,true);
      if(k==="k" || k==="q")readLesson(k==="k"?"fact":"question");
      else if(k==="s" || k==="l")hideAt(k==="s"?"sidehide":"ledgehide");
      else perform(k==="d"?"duster":"flying_squirrel",true);
      return;
    }
    if(e.key!=="Home" && !Object.hasOwn(shifts,e.key)) return;
    e.preventDefault();
    if(scene)stopMotion();
    if(e.key==="Home") {anchor={x:1,y:1};layout();}
    else {
      const step=e.shiftKey?40:12,b=toggle.getBoundingClientRect(),[x,y]=shifts[e.key];
      moveTo(b.left+x*step,b.top+y*step);
    }
    rememberPosition();
  });
  widget.addEventListener("keydown",e=>{
    if(e.key==="Escape" && (!bubble.hidden)){close();toggle.focus({preventScroll:true});}
  });
  function readLesson(kind){
    cancelTap();cancelExpression();
    if(scene)stopMotion();
    if(resting)rest(false,true);sync();speak(kind);
    if(!bubble.hidden)content.focus({preventScroll:true});
  }
  find('[data-action="dismiss"]').onclick=()=>{close();toggle.focus({preventScroll:true});};
  content.onclick=()=>{
    speak(mode==="fact"?"question":mode==="question"?"answer":"fact");
  };
  document.addEventListener("pointermove",e=>{
    if(e.pointerType!=="mouse" || resting || calm.matches || drag?.moving) return;
    lastPoint=[e.clientX,e.clientY];
    if(gazeFrame) return;
    gazeFrame=requestAnimationFrame(()=>{
      gazeFrame=0;if(!lastPoint) return;
      const b=toggle.getBoundingClientRect(),face=faces[bodyPose],
            x=lastPoint[0]-b.left-b.width*face[0]/100,y=lastPoint[1]-b.top-b.height*face[1]/100;
      const next=Math.hypot(x,y)<b.width*.15?-1:(Math.round(Math.atan2(y,x)/(Math.PI/4))+8)%8;
      if(next!==gaze){gaze=next;renderBody();}
    });
  },{passive:true});
  document.addEventListener("pointerleave",()=>{lastPoint=null;gaze=-1;renderBody();});
  new MutationObserver(sync).observe(panel,{attributes:true,subtree:true,attributeFilter:["aria-current"]});
  if(window.ResizeObserver){const size=new ResizeObserver(scheduleLayout);size.observe(bubble);}
  window.addEventListener("resize",scheduleLayout,{passive:true});
  window.visualViewport?.addEventListener("resize",scheduleLayout,{passive:true});
  window.visualViewport?.addEventListener("scroll",scheduleLayout,{passive:true});
  calm.addEventListener("change",()=>{
    const motion=active;
    if(motion && performances[motion.code]){
      if(motion.reduced!==calm.matches){
        stopMotion(true);
        begin({code:motion.code,returnPose:motion.returnPose});
      }
    }else if(!journey){
      stopTravel();stopMotion();go(homePose);
    }
    bodyClock();
    if(!bubble.hidden)speak(mode,automatic);schedulePrompt();schedulePerformance();scheduleScene();
  });
  document.addEventListener("pointerdown",e=>{
    if(!widget.contains(e.target) && (!bubble.hidden))close();
  },{capture:true,passive:true});
  document.addEventListener("keydown",e=>{
    if(e.key!=="Escape")return;
    if(paper)clearPaper(true);
    if(scene)stopMotion();
    if(!bubble.hidden)close();
  });
  document.addEventListener("visibilitychange",()=>{
    if(document.hidden){
      clearPaper();cancelTap();cancelExpression();
        if(drag)release({pointerId:drag.id},true);
      stopTravel();stopMotion();if(automatic)close();
    } else go(homePose);
    bodyClock();schedulePrompt();schedulePerformance();scheduleScene();
  });
  rest(resting,false);sync();
})();
