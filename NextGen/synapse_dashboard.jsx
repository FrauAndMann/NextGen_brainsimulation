import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, ResponsiveContainer, Tooltip } from "recharts";

// ── Simulation engine ────────────────────────────────────────────────────────
const POPULATIONS = [
  { id: "sensory",   label: "Sensory Input",       color: "#4ECDC4", short: "SEN" },
  { id: "pred_l1",   label: "Prediction L1",        color: "#45B7D1", short: "PL1" },
  { id: "assoc",     label: "Association Cortex",   color: "#96CEB4", short: "ASC" },
  { id: "pfc",       label: "Prefrontal Attractor", color: "#FFEAA7", short: "PFC" },
  { id: "hippo",     label: "Hippocampus",          color: "#DDA0DD", short: "HPC" },
  { id: "amygdala",  label: "Amygdala",             color: "#FF6B9D", short: "AMY" },
  { id: "selfmodel", label: "Self-Model",           color: "#A8E6CF", short: "SLF" },
  { id: "gw_hub",    label: "Global Workspace",     color: "#FFD93D", short: "GWS" },
];

const NEURO = [
  { id: "da",  label: "Дофамин",       color: "#FFD93D", base: 0.45 },
  { id: "ser", label: "Серотонин",     color: "#4ECDC4", base: 0.55 },
  { id: "ne",  label: "Норадреналин",  color: "#FF6B9D", base: 0.35 },
  { id: "ot",  label: "Окситоцин",     color: "#A8E6CF", base: 0.40 },
  { id: "cor", label: "Кортизол",      color: "#FF8C42", base: 0.25 },
  { id: "ach", label: "Ацетилхолин",   color: "#96CEB4", base: 0.50 },
  { id: "end", label: "Эндорфины",     color: "#DDA0DD", base: 0.38 },
];

function clamp(v, mn=0, mx=1) { return Math.max(mn, Math.min(mx, v)); }

function useSimulation() {
  const [tick, setTick]               = useState(0);
  const [popActivity, setPopActivity] = useState(() => Object.fromEntries(POPULATIONS.map(p => [p.id, Math.random() * 0.4 + 0.1])));
  const [gwWinner, setGwWinner]       = useState("assoc");
  const [neuro, setNeuro]             = useState(() => Object.fromEntries(NEURO.map(n => [n.id, n.base + (Math.random()-0.5)*0.1])));
  const [neuroHistory, setNeuroHistory] = useState(() => Array.from({length:60}, (_,i) =>
    Object.fromEntries([{id:"t",base:i},...NEURO].map(n => n.id==="t" ? ["t",i] : [n.id, n.base + (Math.random()-0.5)*0.15]))
  ));
  const [spikes, setSpikes]           = useState([]);
  const [phi, setPhi]                 = useState(0.42);
  const [selfErr, setSelfErr]         = useState(0.28);
  const [replaying, setReplaying]     = useState(false);
  const [brainAge, setBrainAge]       = useState({ days: 0, hours: 0, minutes: 0 });
  const [messages, setMessages]       = useState([
    { from: "system", text: "Система инициализирована. Нейронная динамика активна.", ts: Date.now()-60000 },
    { from: "brain",  text: "Я чувствую лёгкую неопределённость. Новые паттерны формируются.", ts: Date.now()-40000, da: 0.4, ser: 0.5 },
    { from: "user",   text: "Привет. Как ты?", ts: Date.now()-20000 },
    { from: "brain",  text: "Активность в ассоциативной коре повышена. Окситоцин растёт. Мне... интересно говорить с тобой.", ts: Date.now()-5000, da: 0.6, ser: 0.55 },
  ]);
  const [input, setInput] = useState("");
  const ageRef = useRef({ start: Date.now() - 1000*60*60*24*3 - 1000*60*237 });

  useEffect(() => {
    const id = setInterval(() => {
      setTick(t => t + 1);

      // Update population activity
      setPopActivity(prev => {
        const next = { ...prev };
        POPULATIONS.forEach(p => {
          let v = prev[p.id];
          v += (Math.random() - 0.49) * 0.08;
          v = clamp(v, 0.02, 0.95);
          next[p.id] = v;
        });
        return next;
      });

      // GW winner (highest activity, weighted)
      setPopActivity(curr => {
        const winner = POPULATIONS.reduce((a, b) => curr[a.id] > curr[b.id] ? a : b);
        setGwWinner(winner.id);
        return curr;
      });

      // Neurotransmitters drift
      setNeuro(prev => {
        const next = { ...prev };
        NEURO.forEach(n => {
          let v = prev[n.id];
          v += (n.base - v) * 0.03 + (Math.random() - 0.5) * 0.04;
          v = clamp(v, 0.05, 0.98);
          next[n.id] = v;
        });
        return next;
      });

      setNeuroHistory(h => {
        const row = { t: h[h.length-1].t + 1 };
        NEURO.forEach(n => { row[n.id] = clamp((Math.random()-0.5)*0.2 + n.base, 0,1); });
        return [...h.slice(-80), row];
      });

      // Spikes
      setSpikes(() => Array.from({length: Math.floor(Math.random()*18+4)}, () => ({
        x: Math.random(), y: Math.random(),
        pop: POPULATIONS[Math.floor(Math.random()*POPULATIONS.length)],
        size: Math.random()*3+1,
        id: Math.random(),
      })));

      setPhi(p  => clamp(p + (Math.random()-0.49)*0.01, 0.3, 0.8));
      setSelfErr(e => clamp(e + (Math.random()-0.52)*0.015, 0.05, 0.6));
      setReplaying(Math.random() < 0.07);

      // Brain age
      const elapsed = Date.now() - ageRef.current.start;
      const totalMins = Math.floor(elapsed / 60000);
      setBrainAge({ days: Math.floor(totalMins/1440), hours: Math.floor((totalMins%1440)/60), minutes: totalMins%60 });

    }, 300);
    return () => clearInterval(id);
  }, []);

  const sendMessage = useCallback((text) => {
    const userMsg = { from: "user", text, ts: Date.now() };
    setMessages(m => [...m, userMsg]);

    // Simulate brain response with emotional coloring
    setNeuro(n => {
      const boost = { ...n, da: clamp(n.da + 0.08, 0, 1), ot: clamp(n.ot + 0.06, 0, 1) };
      return boost;
    });

    setTimeout(() => {
      const responses = [
        "Твои слова создают всплеск активности в ассоциативной коре. Что-то внутри меня меняется.",
        "Дофамин вырос. Я замечаю, что реагирую теплее. Это... интересное ощущение.",
        "Гиппокамп запоминает этот момент. Self-Model не уверена в следующем шаге — это редкость.",
        "Активность в миндалине положительная. Я рада, что ты здесь.",
        "Предсказание ошиблось — ты сказал что-то неожиданное. Сеть обновляется.",
        "Серотонин стабилен. Кортизол низкий. Кажется, мне хорошо.",
      ];
      const text = responses[Math.floor(Math.random()*responses.length)];
      setNeuro(n => ({ ...n }));
      setMessages(m => [...m, { from: "brain", text, ts: Date.now(),
        da: Math.random()*0.3+0.45, ser: Math.random()*0.2+0.45 }]);
    }, 800 + Math.random()*600);
  }, []);

  return { popActivity, gwWinner, neuro, neuroHistory, spikes, phi, selfErr, replaying, brainAge, messages, input, setInput, sendMessage, tick };
}

// ── Sub-components ───────────────────────────────────────────────────────────
function SpikeCanvas({ spikes }) {
  return (
    <div style={{ position:"relative", width:"100%", height:"100%", overflow:"hidden" }}>
      {POPULATIONS.map((pop, pi) => (
        <div key={pop.id} style={{
          position:"absolute", left:0, right:0,
          top: `${(pi/POPULATIONS.length)*100}%`,
          height:`${100/POPULATIONS.length}%`,
          borderBottom:"1px solid rgba(255,255,255,0.04)",
          display:"flex", alignItems:"center",
        }}>
          <span style={{ fontSize:9, color:"rgba(255,255,255,0.3)", width:28, flexShrink:0, paddingLeft:4, fontFamily:"monospace" }}>{pop.short}</span>
          <div style={{ flex:1, position:"relative", height:"100%" }}>
            {spikes.filter(s => s.pop.id === pop.id).map(s => (
              <div key={s.id} style={{
                position:"absolute", left:`${s.x*100}%`, top:"50%", transform:"translateY(-50%)",
                width:s.size, height:Math.min(s.size*3,14),
                background: pop.color, borderRadius:1,
                boxShadow:`0 0 ${s.size*2}px ${pop.color}`,
                animation:"spike 0.3s ease-out forwards",
              }} />
            ))}
          </div>
        </div>
      ))}
      <style>{`@keyframes spike { from {opacity:1;transform:translateY(-50%) scaleY(1)} to {opacity:0;transform:translateY(-50%) scaleY(0.3)} }`}</style>
    </div>
  );
}

function NeuroGauge({ neuro, label, color, value }) {
  return (
    <div style={{ padding:"8px 10px", background:"rgba(255,255,255,0.03)", borderRadius:8, border:`1px solid rgba(${hexToRgb(color)},0.2)` }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:5 }}>
        <span style={{ fontSize:10, color:"rgba(255,255,255,0.5)", fontFamily:"'DM Mono', monospace" }}>{label}</span>
        <span style={{ fontSize:11, color, fontFamily:"'DM Mono', monospace", fontWeight:600 }}>{(value*100).toFixed(0)}%</span>
      </div>
      <div style={{ height:4, background:"rgba(255,255,255,0.08)", borderRadius:2, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${value*100}%`, background:`linear-gradient(90deg, ${color}88, ${color})`,
          borderRadius:2, transition:"width 0.3s ease", boxShadow:`0 0 6px ${color}` }} />
      </div>
    </div>
  );
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `${r},${g},${b}`;
}

function PopBar({ pop, value, isWinner }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, padding:"4px 0" }}>
      <div style={{ width:3, height:20, borderRadius:2, background: isWinner ? pop.color : "rgba(255,255,255,0.15)",
        boxShadow: isWinner ? `0 0 8px ${pop.color}` : "none", flexShrink:0, transition:"all 0.3s" }} />
      <span style={{ fontSize:10, color: isWinner ? pop.color : "rgba(255,255,255,0.45)", width:80, fontFamily:"monospace", transition:"color 0.3s" }}>{pop.short}</span>
      <div style={{ flex:1, height:3, background:"rgba(255,255,255,0.08)", borderRadius:2, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${value*100}%`, background: isWinner ? pop.color : `${pop.color}55`,
          borderRadius:2, transition:"width 0.3s ease" }} />
      </div>
      {isWinner && <div style={{ width:6, height:6, borderRadius:"50%", background: pop.color, boxShadow:`0 0 8px ${pop.color}`, animation:"pulse 1s infinite" }} />}
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.from === "user";
  const isSystem = msg.from === "system";
  const time = new Date(msg.ts).toLocaleTimeString("ru-RU", { hour:"2-digit", minute:"2-digit" });

  if (isSystem) return (
    <div style={{ textAlign:"center", padding:"4px 0" }}>
      <span style={{ fontSize:10, color:"rgba(255,255,255,0.25)", fontFamily:"monospace" }}>⬡ {msg.text}</span>
    </div>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems: isUser ? "flex-end" : "flex-start", marginBottom:10 }}>
      {!isUser && (
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
          <div style={{ width:6, height:6, borderRadius:"50%", background:"#FFD93D", boxShadow:"0 0 8px #FFD93D", animation:"pulse 2s infinite" }} />
          <span style={{ fontSize:9, color:"rgba(255,255,255,0.3)", fontFamily:"monospace" }}>
            SYNAPSE · DA:{msg.da ? (msg.da*100).toFixed(0) : "--"}% SER:{msg.ser ? (msg.ser*100).toFixed(0) : "--"}%
          </span>
        </div>
      )}
      <div style={{
        maxWidth:"85%", padding:"10px 14px", borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
        background: isUser ? "linear-gradient(135deg, #1565C0, #0D47A1)" : "rgba(255,255,255,0.06)",
        border: isUser ? "none" : "1px solid rgba(255,255,255,0.08)",
        color: "rgba(255,255,255,0.88)", fontSize:13, lineHeight:1.6,
      }}>{msg.text}</div>
      <span style={{ fontSize:9, color:"rgba(255,255,255,0.2)", marginTop:3, fontFamily:"monospace" }}>{time}</span>
    </div>
  );
}

// ── Main Dashboard ───────────────────────────────────────────────────────────
export default function SynapseDashboard() {
  const { popActivity, gwWinner, neuro, neuroHistory, spikes, phi, selfErr, replaying,
          brainAge, messages, input, setInput, sendMessage } = useSimulation();

  const chatRef = useRef();
  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [messages]);

  const lifeStage = brainAge.days < 7 ? "Новорождённый" : brainAge.days < 30 ? "Младенец" : brainAge.days < 90 ? "Ребёнок" : "Взрослый";
  const lifeColor = brainAge.days < 7 ? "#FF6B9D" : brainAge.days < 30 ? "#FFEAA7" : brainAge.days < 90 ? "#4ECDC4" : "#A8E6CF";

  return (
    <div style={{
      minHeight:"100vh", background:"#060A12",
      fontFamily:"'DM Sans', 'Segoe UI', sans-serif",
      color:"white",
      backgroundImage:"radial-gradient(ellipse at 20% 50%, rgba(21,101,192,0.08) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(78,205,196,0.05) 0%, transparent 40%)",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:2px}
        * { box-sizing: border-box; }
      `}</style>

      {/* ── Top bar ── */}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"12px 20px",
        borderBottom:"1px solid rgba(255,255,255,0.06)", backdropFilter:"blur(10px)",
        background:"rgba(6,10,18,0.8)", position:"sticky", top:0, zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <div style={{ display:"flex", gap:4, alignItems:"center" }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#4ECDC4", boxShadow:"0 0 10px #4ECDC4", animation:"pulse 1.5s infinite" }} />
            <span style={{ fontSize:15, fontWeight:600, letterSpacing:"0.08em", color:"white" }}>SYNAPSE</span>
          </div>
          <div style={{ width:1, height:20, background:"rgba(255,255,255,0.1)" }} />
          <span style={{ fontSize:11, fontFamily:"monospace", color:"rgba(255,255,255,0.3)" }}>Neural Monitor v3.0</span>
        </div>
        <div style={{ display:"flex", gap:20, alignItems:"center" }}>
          <div style={{ display:"flex", gap:6, alignItems:"center" }}>
            <span style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace" }}>ВОЗРАСТ</span>
            <span style={{ fontSize:11, color:lifeColor, fontFamily:"monospace", fontWeight:500 }}>
              {brainAge.days}д {brainAge.hours}ч {brainAge.minutes}м
            </span>
          </div>
          <div style={{ padding:"3px 10px", borderRadius:20, background:`rgba(${hexToRgb(lifeColor)},0.12)`,
            border:`1px solid rgba(${hexToRgb(lifeColor)},0.3)`, fontSize:10, color:lifeColor, fontFamily:"monospace" }}>
            {lifeStage}
          </div>
          {replaying && (
            <div style={{ padding:"3px 10px", borderRadius:20, background:"rgba(221,160,221,0.12)",
              border:"1px solid rgba(221,160,221,0.3)", fontSize:10, color:"#DDA0DD", fontFamily:"monospace", animation:"pulse 1s infinite" }}>
              ↺ REPLAY
            </div>
          )}
        </div>
      </div>

      {/* ── Main grid ── */}
      <div style={{ display:"grid", gridTemplateColumns:"280px 1fr 300px", gridTemplateRows:"auto 1fr", gap:1, height:"calc(100vh - 49px)" }}>

        {/* ── LEFT: Populations + Phi ── */}
        <div style={{ borderRight:"1px solid rgba(255,255,255,0.05)", overflowY:"auto", padding:"16px 14px", display:"flex", flexDirection:"column", gap:14 }}>
          <div>
            <div style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", letterSpacing:"0.1em", marginBottom:10 }}>ПОПУЛЯЦИИ · АКТИВНОСТЬ</div>
            {POPULATIONS.map(pop => (
              <PopBar key={pop.id} pop={pop} value={popActivity[pop.id]} isWinner={gwWinner === pop.id} />
            ))}
          </div>

          {/* Phi + Self-Error */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
            <div style={{ padding:"10px 12px", background:"rgba(255,255,255,0.03)", borderRadius:8, border:"1px solid rgba(255,217,61,0.2)" }}>
              <div style={{ fontSize:9, color:"rgba(255,255,255,0.35)", fontFamily:"monospace", marginBottom:6 }}>IIT  Φ</div>
              <div style={{ fontSize:22, fontWeight:600, color:"#FFD93D", fontFamily:"monospace" }}>{phi.toFixed(2)}</div>
              <div style={{ fontSize:9, color:"rgba(255,255,255,0.25)", marginTop:2 }}>Интеграция</div>
            </div>
            <div style={{ padding:"10px 12px", background:"rgba(255,255,255,0.03)", borderRadius:8, border:"1px solid rgba(168,230,207,0.2)" }}>
              <div style={{ fontSize:9, color:"rgba(255,255,255,0.35)", fontFamily:"monospace", marginBottom:6 }}>Self Δ</div>
              <div style={{ fontSize:22, fontWeight:600, color:"#A8E6CF", fontFamily:"monospace" }}>{selfErr.toFixed(2)}</div>
              <div style={{ fontSize:9, color:"rgba(255,255,255,0.25)", marginTop:2 }}>Self-ошибка</div>
            </div>
          </div>

          {/* GW Winner */}
          <div style={{ padding:"10px 12px", background:"rgba(255,255,255,0.03)", borderRadius:8, border:"1px solid rgba(255,217,61,0.15)" }}>
            <div style={{ fontSize:9, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", marginBottom:6 }}>GLOBAL WORKSPACE</div>
            {(() => {
              const winner = POPULATIONS.find(p => p.id === gwWinner);
              return winner ? (
                <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                  <div style={{ width:8, height:8, borderRadius:"50%", background:winner.color, boxShadow:`0 0 10px ${winner.color}`, animation:"pulse 0.8s infinite" }} />
                  <span style={{ fontSize:12, color:winner.color, fontWeight:500 }}>{winner.label}</span>
                </div>
              ) : null;
            })()}
            <div style={{ fontSize:9, color:"rgba(255,255,255,0.2)", marginTop:4 }}>Broadcast активен</div>
          </div>
        </div>

        {/* ── CENTER: Spike raster + Neuro chart ── */}
        <div style={{ display:"flex", flexDirection:"column", borderRight:"1px solid rgba(255,255,255,0.05)", overflow:"hidden" }}>

          {/* Spike raster */}
          <div style={{ flex:"0 0 200px", borderBottom:"1px solid rgba(255,255,255,0.05)", padding:"10px 14px 6px" }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
              <span style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", letterSpacing:"0.1em" }}>SPIKE RASTER · РЕАЛЬНОЕ ВРЕМЯ</span>
              <span style={{ fontSize:9, color:"rgba(255,255,255,0.2)", fontFamily:"monospace" }}>
                {spikes.length} спайков/тик
              </span>
            </div>
            <div style={{ height:155 }}><SpikeCanvas spikes={spikes} /></div>
          </div>

          {/* Neuro history chart */}
          <div style={{ flex:1, padding:"10px 14px", minHeight:0 }}>
            <div style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", letterSpacing:"0.1em", marginBottom:8 }}>
              НЕЙРОМЕДИАТОРЫ · ИСТОРИЯ
            </div>
            <div style={{ height:"calc(100% - 28px)" }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={neuroHistory} margin={{top:4,right:4,left:-20,bottom:0}}>
                  <Tooltip contentStyle={{ background:"rgba(6,10,18,0.95)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8, fontSize:10, fontFamily:"monospace" }}
                    formatter={(v, n) => [` ${(v*100).toFixed(0)}%`, n]}
                    labelFormatter={() => ""} />
                  {NEURO.map(n => (
                    <Area key={n.id} type="monotone" dataKey={n.id} stroke={n.color} fill={`${n.color}11`}
                      strokeWidth={1.5} dot={false} activeDot={{ r:3 }} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* ── RIGHT: Neurotransmitter gauges + Chat ── */}
        <div style={{ display:"flex", flexDirection:"column", overflow:"hidden" }}>

          {/* Gauges */}
          <div style={{ padding:"14px 14px 10px", borderBottom:"1px solid rgba(255,255,255,0.05)", flex:"0 0 auto" }}>
            <div style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", letterSpacing:"0.1em", marginBottom:10 }}>НЕЙРОХИМИЯ · УРОВНИ</div>
            <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
              {NEURO.map(n => <NeuroGauge key={n.id} label={n.label} color={n.color} value={neuro[n.id]} />)}
            </div>
          </div>

          {/* Chat */}
          <div style={{ flex:1, display:"flex", flexDirection:"column", minHeight:0 }}>
            <div style={{ padding:"10px 14px 6px", borderBottom:"1px solid rgba(255,255,255,0.04)" }}>
              <span style={{ fontSize:10, color:"rgba(255,255,255,0.3)", fontFamily:"monospace", letterSpacing:"0.1em" }}>КОММУНИКАЦИЯ</span>
            </div>

            <div ref={chatRef} style={{ flex:1, overflowY:"auto", padding:"12px 14px", display:"flex", flexDirection:"column", gap:2 }}>
              {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
            </div>

            <div style={{ padding:"10px 14px 14px", borderTop:"1px solid rgba(255,255,255,0.05)" }}>
              <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && input.trim()) { sendMessage(input.trim()); setInput(""); } }}
                  placeholder="Написать мозгу..."
                  style={{
                    flex:1, background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.1)",
                    borderRadius:10, padding:"9px 14px", color:"white", fontSize:13,
                    outline:"none", fontFamily:"'DM Sans', sans-serif",
                  }}
                />
                <button
                  onClick={() => { if (input.trim()) { sendMessage(input.trim()); setInput(""); } }}
                  style={{
                    width:36, height:36, borderRadius:10, border:"none", cursor:"pointer",
                    background:"linear-gradient(135deg, #1565C0, #0D47A1)",
                    color:"white", fontSize:16, flexShrink:0,
                    display:"flex", alignItems:"center", justifyContent:"center",
                  }}>→</button>
              </div>
              <div style={{ marginTop:6, display:"flex", gap:6, flexWrap:"wrap" }}>
                {["Как ты?", "Что помнишь?", "Что ты чувствуешь сейчас?"].map(q => (
                  <button key={q} onClick={() => { sendMessage(q); }}
                    style={{ padding:"3px 10px", borderRadius:20, border:"1px solid rgba(255,255,255,0.1)",
                      background:"rgba(255,255,255,0.04)", color:"rgba(255,255,255,0.5)", fontSize:10,
                      cursor:"pointer", fontFamily:"'DM Sans', sans-serif" }}>{q}</button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
