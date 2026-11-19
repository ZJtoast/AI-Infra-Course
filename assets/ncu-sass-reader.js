"use strict";
(() => {
  const lab = document.querySelector("[data-sass-lab]");
  if (!lab) return;
  const [locations, programs] = window.ncuSM80;
  const select = lab.querySelector("select");
  const filter = lab.querySelector("input");
  const grid = lab.querySelector("tbody");
  const status = lab.querySelector("[role=status]");
  const prev = lab.querySelector("[data-sass-prev]"), next = lab.querySelector("[data-sass-next]");
  const details = lab.querySelector("[data-sass-detail]");
  const families={
    LDG:"从全局地址加载，目的寄存器在前，地址在方括号中。追等待时沿目的寄存器寻找后续消费者。",
    LDL:"从当前线程的 local 地址读回寄存器。若编译器报告有 spill loads，可与同一栈位置的写入及循环次数对照。",
    STL:"把寄存器写入当前线程的 local 地址。检查写入的值在后续何处读回，并对照编译器的 spill stores。",
    STG:"将寄存器写回全局地址。先按 lane 计算地址间隔，再比较 global store 的 sectors/request。",
    LDS:"从 shared 加载到寄存器。地址分布与访问宽度共同决定理想 wavefront 数，超出的批次可提示 bank 冲突。",
    STS:"将线程的源值写入 shared。检查其他 warp 在读取该位置前是否已经经过必要的块同步。",
    HMMA:"Tensor Core 半精度矩阵乘加。16816 对应该机器指令的矩阵形状，F32 表示此处的浮点累加形式；操作数是按架构约定分布的寄存器组。",
    MOVM:"在寄存器中变换矩阵片段的排布。检查前后的矩阵加载和 HMMA，核对操作数的行列布局。",
    CS2R:"读取特殊寄存器。结合操作数判断它读取的是哪一项硬件状态。",
    HADD2:"对两个打包的半精度通道执行加法；读取操作数可区分算术路径和编译器生成的数据处理。",
    S2UR:"将特殊寄存器的内容读到 uniform 寄存器，供整个 warp 的一致控制或地址计算使用。",
    UIMAD:"uniform 路径的整数乘加，用于一致的下标或地址计算。",
    UISETP:"比较 uniform 整数操作数并设置谓词，可控制整 warp 一致的分支。",
    ULEA:"在 uniform 路径形成有效地址。",
    ULOP3:"对 uniform 操作数执行按位逻辑。",
    UMOV:"将值写入 uniform 寄存器。",
    USHF:"在 uniform 路径执行组合移位，常用于索引计算。",
    FFMA:"以两个输入相乘再加第三个输入，一次舍入，结果写到第一个寄存器。若目的寄存器也在输入中，继续追该累加链。",
    FADD:"浮点加法。对齐两个输入寄存器的生产指令，确认慢的一侧是内存返回还是前次算术。",
    FMUL:"浮点乘法。查看源、目的寄存器及其前后依赖；相邻的指数路径也可能用乘法转换对数底。",
    FMNMX:"按指令的选择条件取浮点最小值或最大值。在 softmax 中沿这些结果查行最大值如何汇合。",
    SHFL:"在同一 warp 内交换寄存器值。观察 DOWN、BFLY 或 IDX 及其位移参数，分别对应向下读取、异或交换或按 lane 索引读取。",
    MUFU:"特殊函数操作。后缀标出所用操作，如 EX2 求以 2 为底的指数、RCP 求倒数。",
    BAR:"执行块级屏障。屏障前工作的快慢差会让部分 warp 在这里等待。",
    WARPSYNC:"按参与掩码在 warp 内同步；掩码覆盖的线程需要按要求到达。",
    MATCH:"匹配 warp 中的值。直方图用匹配结果将键相同的 lane 划成一组。",
    POPC:"计算输入中值为 1 的位数。对匹配掩码计数，就得到该组替同一桶累加的键数。",
    ATOMS:"更新 shared 地址中的原子值。比较谓词有效的 lane 和目标桶分布，区分单一热点与多个独立桶。",
    RED:"对内存目标做归约更新，调用方无需取得更新前的旧值。后缀与目标地址说明数据类型和访问范围。",
    IADD3:"整数加法，可用于步长、下标或地址计算；不要把整数寻址计入浮点 FLOP。",
    IMAD:"整数乘加。WIDE 形式常用于生成地址：按索引计算偏移，再加基址。",
    LEA:"形成有效地址，组合寄存器与移位等操作。沿结果可找到随后访问的 LDG、STG 或其他内存指令。",
    S2R:"读取特殊寄存器。例如 CTAID 和 TID 分别与块编号和线程编号有关，可从这里向后推导 lane 地址。",
    ISETP:"用整数比较设置谓词。后续带 @P 或 @!P 的指令以该条件选择参与线程。",
    FSETP:"浮点比较并设置谓词。追踪使用谓词的下一条条件指令。",
    BRA:"跳转到目标标签。若受谓词控制，检查循环条件、边界或分支路径。",
    BSSY:"设置分支路径的汇合状态，配合相应的 BSYNC 阅读。",
    BSYNC:"在标出的汇合点同步此前分开的控制流。",
    CALL:"调用子程序。对数学库慢路径，先看进入该调用的谓词或分支是否常被实际输入触发。",
    RET:"从子程序返回。静态存在的数学库路径是否执行，要通过动态计数验证。",
    EXIT:"结束当前线程的执行。检查前面是否有边界谓词使一部分线程提前退出。",
    SEL:"按选择条件将不同输入写入目的寄存器。可在没有显式跳转时实现条件选择。",
    MOV:"将值送入目的寄存器。常量、立即数或另一个寄存器可作为源。",
    NOP:"空操作，常出现在布局或填充位置。静态列表中的行数无法直接给出运行时间。",
    LOP3:"对三路输入执行由真值表指定的按位逻辑，经常用于掩码与地址计算。",
    PLOP3:"对谓词执行逻辑组合，结果控制后面的条件操作。",
    SHF:"对组合的输入进行移位，常见于整数索引和位域运算。",
    I2F:"将整数转换为浮点数，例如将一组 lane 的整数成员数变为 FP32 增量。",
    IMNMX:"按条件取整数最小或最大值，常用于范围或循环控制。",
    UIADD3:"在 uniform 数据路径上执行整数加法，操作数可由 warp 内的线程共享。",
    ULDC:"把常量空间中的值加载到 uniform 寄存器，随后可供一致的地址或控制计算使用。",
    FCHK:"检查浮点运算的特殊情况，编译生成的数学路径可根据检查结果进入后备实现。",
    BREV:"反转输入的位次序，可与查找高位的指令组合定位掩码中最低的置位。",
    FLO:"寻找输入的前导有效位；在选择组长的路径中，可和位反转指令配对阅读。",
    HFMA2:"执行成对的半精度乘加。还要核对操作数；编译器也会借用它构造或搬运某些位模式。",
    YIELD:"为 warp 调度提供让出当前执行机会的提示。"
  };
  const hex=n=>"0x"+n.toString(16).padStart(4,"0");
  let page=0;
  for(const name of Object.keys(programs).sort())select.add(new Option(name,name));
  select.value="stride_add";filter.value="FADD";
  function show(row) {
    const [pc,instruction,index]=row, [file,line,statement]=locations[index];
    const opcode=instruction.replace(/^@\S+\s+/,"").split(/[.\s]/)[0];
    const title=document.createElement("strong");title.textContent=hex(pc)+"　"+instruction;
    const src=document.createElement("code");
    src.textContent=(file?file+" · "+line+" 行":"无独立源行映射")+(statement?"　"+statement:"");
    const note=document.createElement("p");note.textContent=families[opcode]||"按操作数和指令后缀对照源语句。";
    details.replaceChildren(title,src,note);
    for(const b of grid.querySelectorAll("button"))b.setAttribute("aria-pressed",String(b.textContent===hex(pc)));
  }
  function render(){
    const words=filter.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const rows=programs[select.value].filter(([pc,ins,loc])=>{
      const [file,line]=locations[loc];
      const hay=(hex(pc)+" "+ins+" "+file+":"+line).toLowerCase();
      return words.every(w=>/^r\d+$/.test(w) ? new RegExp("\\b"+w+"\\b","i").test(ins) : hay.includes(w));
    });
    const count=12;
    grid.replaceChildren();
    const chunk=rows.slice(page*count,(page+1)*count);
    for(const row of chunk){
      const tr=document.createElement("tr"),pc=tr.insertCell(),code=tr.insertCell();
      const button=document.createElement("button");button.type="button";button.textContent=hex(row[0]);
      button.addEventListener("click",()=>show(row));pc.append(button);
      const value=document.createElement("code");value.textContent=row[1];code.append(value);grid.append(tr);
    }
    prev.disabled=page===0;next.disabled=(page+1)*count>=rows.length;
    status.textContent=rows.length ? `共 ${rows.length} 条 · 第 ${page+1} / ${Math.ceil(rows.length/count)} 页` : "未找到匹配指令";
    if(chunk.length)show(chunk[0]);else details.replaceChildren();
  }
  for(const el of [filter,select])el.addEventListener("input",()=>{page=0;render()});
  prev.addEventListener("click",()=>{page--;render()});next.addEventListener("click",()=>{page++;render()});
  document.addEventListener("click", event => {
    const link = event.target.closest("a[data-sass-kernel][data-sass-pc]");
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const {sassKernel, sassPc} = link.dataset;
    if (!programs[sassKernel]?.some(row => hex(row[0]) === sassPc)) return;
    select.value = sassKernel;
    filter.value = sassPc;
    page = 0;
    render();
  });
  render();lab.hidden=false;
})();
