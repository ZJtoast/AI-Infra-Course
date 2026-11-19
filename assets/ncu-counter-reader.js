"use strict";
(() => {
  const root = document.querySelector("[data-ncu-catalog]");
  if (!root) return;
  const metrics = window.ncuGA100;
  const search = root.querySelector("input");
  const select = root.querySelector("select");
  const list = root.querySelector("[data-counter-results]");
  const detail = root.querySelector("[data-counter-detail]");
  const status = root.querySelector("[role=status]");
  const prev = root.querySelector("[data-counter-prev]");
  const next = root.querySelector("[data-counter-next]");
  const pageSize = 8;
  const device = {dram:"显存接口",fbpa:"显存分区",lts:"L2 缓存切片",l1tex:"L1 / 纹理 / shared 路径",sm:"流多处理器",smsp:"SM 内的调度分区",gpu:"整块设备",pcie:"PCIe 接口",nvlrx:"NVLink 接收端",nvltx:"NVLink 发送端",tpc:"纹理处理簇",gpc:"图形处理簇",gcc:"常量缓存",idc:"指令数据缓存",sys:"系统时钟域",gr:"图形与计算引擎",fe:"前端"};
  const units = {sector:"sector（缓存事务以 32 B 为一段）",request:"请求",inst:"指令",cycle:"周期",byte:"字节",warp:"warp",thread:"线程",block:"线程块",ns:"纳秒",ops:"操作","%":"百分比","inst/warp":"指令/warp","thread/warp":"线程/warp","inst/cycle":"指令/周期","byte/sector":"字节/sector","cycle/warp":"周期/warp",tag:"缓存标签"};
  const suffixes = {
    sum:"所有相应硬件实例的计数相加", avg:"对相应硬件实例取平均",
    max:"取硬件实例中的最大值", min:"取硬件实例中的最小值",
    pct:"比值以百分数显示", ratio:"按比值显示", per_second:"折算为每秒的量",
    per_cycle_active:"除以该单元活跃的周期数", per_cycle_elapsed:"除以经过的周期数",
    pct_of_peak_sustained_active:"以活跃周期内可持续达到的峰值为分母，结果用百分数显示",
    pct_of_peak_sustained_elapsed:"以整个测量时间内可持续达到的峰值为分母，结果用百分数显示"
  };
  function fullMetric() {
    return search.value.trim().toLowerCase().match(/^([a-z0-9]+__[a-z0-9_]+)((?:\.[a-z0-9_]+)+)$/);
  }
  const stall = {
    long_scoreboard:"等待经 L1TEX 路径返回的依赖。选消费寄存器的 PC，向前追生产它的加载，再看 sector 及有效 lane。",
    short_scoreboard:"这个名称指向 MIO 操作的依赖。先从等待读数的指令找到产生结果的指令，再分开查 shared 加载、特殊数学运算或动态跳转。",
    not_selected:"warp 已可发射，调度器本周期选了其他 warp；用 issue active 判断发射槽是否已得到利用。",
    barrier:"warp 已到线程块屏障，需要同块其他 warp 到齐。比较屏障前工作量和动态同步次数。",
    membar:"warp 在等待内存顺序约束完成。定位屏障作用域与跨线程数据依赖。",
    lg_throttle:"本地或全局内存指令队列阻塞。看热点 load/store 的频次、宽度和事务效率。",
    mio_throttle:"MIO 路径的指令队列阻塞。用 SASS 区分 shared、特殊数学和其他 MIO 操作。",
    tex_throttle:"纹理指令路径出现队列压力；定位相关访问并检查请求数和并行发射。",
    math_pipe_throttle:"所需数学执行管线无法接收本条指令。展开 Compute Workload，先找出占用该管线的操作。",
    wait:"固定延迟的执行依赖使下一指令暂不可发射。在 Source 追相邻的生产与消费寄存器。",
    no_instruction:"调度器暂时没有可用的下一条指令。查看取指、代码体积及热点展开循环。",
    branch_resolving:"等待分支目标或程序计数器更新。查看分支前的谓词、目标及各路径参与线程。",
    drain:"warp 已进入退出阶段，仍有待完成的内存操作。查看结束前的写回和网格尾部。",
    sleeping:"warp 正在休眠。定位主动等待循环及其唤醒条件。"
  };
  const guidance = {
    lts:"在 Memory Workload 查同一来源、地址范围和操作的总量。求命中率时用匹配条件下的命中数除以命中与未命中之和。一个请求可跨多个 sector，应在相同计量单位内相除。",
    l1tex:"在 Memory Workload 按 global、local、shared 和操作方向分列。全局加载用 sectors/request 验算地址合并；shared 指令先对齐访问宽度，再比较实际与理想 wavefront。",
    dram:"把本次显存读写字节除以相同输出量，并与持续时间一起读。缓存命中会使逻辑访问量与穿过显存接口的流量不同。",
    smsp:"在 Scheduler 与 Source 检查同一调度分区的 active、eligible、issue，再看动态指令、谓词和依赖。整次启动和单条 PC 是两个统计范围。",
    sm:"将执行管线与实际 SASS 对齐。按动态操作数、有效 lane 和执行时间区分计算量增加、管线繁忙和发射不足。",
    nvlrx:"对齐参与 rank 和传输方向；将用户负载、协议字节和链路时间分开。检查同一消息在发送侧的配对活动。",
    nvltx:"按同一消息对照接收侧与请求进度；协议头、请求和响应可分别计数。先核对互连是否位于消费依赖上。",
    pcie:"把方向、传输大小及经过时间与 Systems 的 H2D/D2H 或对端传输对齐；多个进程共用链路时还要辨明来源。"
  };
  function para(text) { const el=document.createElement("p"); el.textContent=text; return el; }
  function show(row) {
    const [name,unit,type,meaning]=row;
    const [unitName,body]=name.split("__");
    const title=document.createElement("code"); title.textContent=name;
    const reason=Object.keys(stall).find(s=>body.includes("stalled_"+s));
    const text=reason ? stall[reason] : guidance[unitName];
    const kind={Counter:"计数器",Ratio:"比值",Throughput:"综合吞吐"}[type];
    const pieces=[title,para([device[unitName],kind,"基础单位：" + (units[unit]||unit||"见派生指标")].join(" · "))];
    pieces.push(para(meaning));
    const full = fullMetric();
    if (full && full[1] === name) {
      const parts = full[2].slice(1).split(".");
      pieces.push(para(full[0]));
      for (const part of parts) {
        if (suffixes[part]) pieces.push(para("." + part + "　" + suffixes[part]));
      }
    }
    if(text)pieces.push(para(text));
    const cli=document.createElement("code");
    cli.textContent="ncu --query-metrics --chips ga100 --query-metrics-mode suffix --metrics "+name;
    pieces.push(para("可在本机查询该指标的可用后缀"),cli);
    detail.replaceChildren(...pieces);
    for(const button of list.querySelectorAll("button")) button.setAttribute("aria-pressed",String(button.textContent===name));
  }
  let page=0;
  function render() {
    const full = fullMetric();
    const words = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const rows=metrics.filter(([id, , , meaning])=>{
      const unit=id.split("__")[0];
      if (select.value && select.value !== unit) return false;
      if (full) return id === full[1];
      const haystack = (id + " " + device[unit] + " " + meaning).toLowerCase();
      return words.every(word => haystack.includes(word));
    });
    const slice=rows.slice(page*pageSize,(page+1)*pageSize);
    list.replaceChildren();
    for(const record of slice) {
      const li=document.createElement("li"), btn=document.createElement("button");btn.type="button";
      btn.textContent=record[0];btn.addEventListener("click",()=>show(record));li.append(btn);list.append(li);
    }
    prev.disabled=page===0;next.disabled=(page+1)*pageSize>=rows.length;
    status.textContent=rows.length ? `共 ${rows.length} 项 · 第 ${page+1} / ${Math.ceil(rows.length/pageSize)} 页` : "未找到匹配指标";
    if(slice.length)show(slice[0]);else detail.replaceChildren();
  }
  for(const [id,label] of Object.entries(device)){
    const opt=document.createElement("option");opt.value=id;opt.textContent=id+" · "+label;select.append(opt);
  }
  for(const control of [select,search])control.addEventListener("input",()=>{page=0;render()});
  prev.addEventListener("click",()=>{page--;render()});next.addEventListener("click",()=>{page++;render()});
  render();root.hidden=false;
})();
