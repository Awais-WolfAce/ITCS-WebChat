(() => {
  /* ── DOM ──────────────────────────────────── */
  const messagesEl  = document.getElementById("messages");
  const welcomeEl   = document.getElementById("welcome");
  const inputEl     = document.getElementById("user-input");
  const sendBtn     = document.getElementById("send-btn");
  const modeSwitch  = document.getElementById("mode-switch");
  const lblText     = document.getElementById("lbl-text");
  const lblVoice    = document.getElementById("lbl-voice");
  const textBar     = document.getElementById("text-input-bar");
  const voiceBarEl  = document.getElementById("voice-bar");
  const stateDot    = document.getElementById("state-dot");
  const voiceText   = document.getElementById("voice-bar-text");
  const waveformEl  = document.getElementById("waveform");

  const NUM_BARS = 20;
  for (let i = 0; i < NUM_BARS; i++) {
    const b = document.createElement("div");
    b.className = "bar";
    b.style.height = "4px";
    waveformEl.appendChild(b);
  }
  const bars = waveformEl.querySelectorAll(".bar");

  let history = [];
  let busy    = false;

  /* ── Conversation id ──────────────────────────
     The server is authoritative: we send whatever id we have stored
     (may be null on first turn) and adopt whatever id comes back. If
     our stored id has already expired server-side (idle timeout), the
     server will mint a new one and the next response will return that
     new id, which we save. Per-tab storage means a fresh tab / new
     window starts a new conversation; reload of the same tab keeps it.
     -------------------------------------------- */
  const USER_ID_KEY = "itcs_chat_user_id";
  function getUserId() {
    try { return sessionStorage.getItem(USER_ID_KEY); }
    catch { return null; }
  }
  function setUserId(id) {
    if (!id) return;
    try { sessionStorage.setItem(USER_ID_KEY, id); } catch {}
  }
  function clearUserId() {
    try { sessionStorage.removeItem(USER_ID_KEY); } catch {}
  }

  /* ════════════════════════════════════════════
     SHARED HELPERS
     ════════════════════════════════════════════ */
  function renderMarkdown(raw) {
    let html = raw
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    html = html.replace(/```([\s\S]*?)```/g, (_, code) =>
      `<pre><code>${code.trim()}</code></pre>`
    );
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    const lines = html.split("\n");
    let out = "", inUl = false, inOl = false;

    function closeLists() {
      if (inUl) { out += "</ul>"; inUl = false; }
      if (inOl) { out += "</ol>"; inOl = false; }
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      if (/^#{1,4}\s+/.test(line)) {
        closeLists();
        const lvl = line.match(/^(#{1,4})/)[1].length;
        const txt = line.replace(/^#{1,4}\s+/, "");
        out += `<h${lvl}>${inlineFormat(txt)}</h${lvl}>`;
        continue;
      }

      const ulMatch = line.match(/^\s*[-+]\s+(.*)/);
      if (ulMatch) {
        if (inOl) { out += "</ol>"; inOl = false; }
        if (!inUl) { out += "<ul>"; inUl = true; }
        out += `<li>${inlineFormat(ulMatch[1])}</li>`;
        continue;
      }

      const olMatch = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (olMatch) {
        if (inUl) { out += "</ul>"; inUl = false; }
        if (!inOl) { out += "<ol>"; inOl = true; }
        out += `<li>${inlineFormat(olMatch[1])}</li>`;
        continue;
      }

      closeLists();
      if (line.trim() === "") { out += "<br>"; continue; }
      out += `<p>${inlineFormat(line)}</p>`;
    }
    closeLists();
    return out;
  }

  function inlineFormat(s) {
    // Stash linkified spans in placeholders so later markdown passes
    // (bold/italic, etc.) cannot break the generated <a> tags.
    const stash = [];
    const ph = () => `\u0000L${stash.length - 1}\u0000`;
    const a = (href, text) => {
      const isExternal = /^https?:/i.test(href);
      const attrs = isExternal ? ' target="_blank" rel="noopener noreferrer"' : "";
      stash.push(`<a href="${href}"${attrs}>${text}</a>`);
      return ph();
    };

    s = s.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+|tel:[^\s)]+)\)/g,
      (_, text, url) => a(url, text)
    );

    s = s.replace(
      /&lt;(https?:\/\/[^\s&]+)&gt;/g,
      (_, url) => a(url, url)
    );

    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, (_, lead, url) => {
      let trail = "";
      const m = url.match(/[).,;:!?'"]+$/);
      if (m) { trail = m[0]; url = url.slice(0, -trail.length); }
      return `${lead}${a(url, url)}${trail}`;
    });

    s = s.replace(
      /([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/g,
      (m) => a(`mailto:${m}`, m)
    );

    s = s.replace(/(\+\d[\d\s().-]{6,}\d)/g, (m) => {
      const tel = m.replace(/[^\d+]/g, "");
      return a(`tel:${tel}`, m.trim());
    });

    s = s
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(.+?)__/g, "<strong>$1</strong>")
      .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>")
      .replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, "<em>$1</em>");

    s = s.replace(/\u0000L(\d+)\u0000/g, (_, i) => stash[+i]);
    return s;
  }

  function addMessage(role, text) {
    if (welcomeEl && welcomeEl.parentNode) welcomeEl.remove();
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (role === "bot") {
      div.innerHTML = renderMarkdown(text);
    } else {
      div.textContent = text;
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function showTyping() {
    const div = document.createElement("div");
    div.className = "typing"; div.id = "typing-indicator";
    div.innerHTML = "<span></span><span></span><span></span>";
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  async function chatRequest(msgs) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: msgs, user_id: getUserId() }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", botText = "", lang = "en";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") break;
        try {
          const d = JSON.parse(payload);
          if (d.user_id) setUserId(d.user_id);
          if (d.error) throw new Error(d.error);
          if (d.content) botText = d.content;
          if (d.lang) lang = d.lang;
        } catch (e) { if (e.message && !e.message.startsWith("Unexpected")) throw e; }
      }
    }
    return { text: botText, lang };
  }

  /* ── End-of-session beacon ─────────────────────
     Best-effort notify the server when the tab is closing or hidden
     for good, so the session-end email fires immediately instead of
     waiting for the idle timeout. Uses sendBeacon (queued by the
     browser even after the page is unloading); falls back to a
     keepalive fetch if sendBeacon isn't available.
     -------------------------------------------- */
  function sendEndBeacon() {
    const id = getUserId();
    if (!id) return;
    const url = "/api/chat/end";
    const body = JSON.stringify({ user_id: id });
    try {
      if (navigator.sendBeacon) {
        const blob = new Blob([body], { type: "application/json" });
        navigator.sendBeacon(url, blob);
        return;
      }
    } catch {}
    try {
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: true,
      });
    } catch {}
  }
  // pagehide fires on tab close, window close, and full-page navigation
  // away. It does NOT fire on a quick tab switch (the user clicking
  // back to another tab and returning), so the session stays alive
  // across brief context switches and only the 60s idle sweeper can
  // expire it in those cases.
  window.addEventListener("pagehide", sendEndBeacon);

  /* ════════════════════════════════════════════
     TEXT CHAT
     ════════════════════════════════════════════ */
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
  });
  sendBtn.addEventListener("click", sendText);

  async function sendText() {
    const text = inputEl.value.trim();
    if (!text || busy) return;
    addMessage("user", text);
    history.push({ role: "user", content: text });
    inputEl.value = ""; inputEl.style.height = "auto";
    busy = true; sendBtn.disabled = true; inputEl.disabled = true;
    showTyping();
    try {
      const { text: botText } = await chatRequest(history);
      removeTyping();
      addMessage("bot", botText || "No response received.");
      history.push({ role: "assistant", content: botText });
    } catch {
      removeTyping();
      addMessage("bot", "An error occurred. Please try again.");
    }
    busy = false; sendBtn.disabled = false; inputEl.disabled = false;
    inputEl.focus();
  }

  /* ════════════════════════════════════════════
     VOICE SESSION — continuous hands-free
     ════════════════════════════════════════════ */
  const VAD_INTERVAL_MS  = 80;
  const SPEECH_CONFIRM   = 3;
  const SILENCE_END_MS   = 1200;
  const PRE_BUFFER_COUNT = 5;
  const RMS_THRESHOLD    = 0.012;

  let voiceActive  = false;
  let voiceState   = "IDLE";
  let audioCtx     = null;
  let micStream    = null;
  let analyser     = null;
  let processor    = null;
  let vadTimer     = null;
  let chunks       = [];
  let preBuffer    = [];
  let speechFrames = 0;
  let silenceStart = null;
  let currentAudio = null;

  /* ── state management ─────────────────────── */
  function setVoiceState(state) {
    voiceState = state;
    stateDot.className = "state-dot " + state.toLowerCase();
    waveformEl.className = "waveform" + (
      state === "RECORDING" ? " recording" :
      state === "SPEAKING"  ? " speaking" :
      state === "LISTENING" ? " active" : ""
    );
    const labels = {
      LISTENING: "Listening...",
      RECORDING: "Recording...",
      PROCESSING: "Processing...",
      SPEAKING: "Speaking...",
    };
    voiceText.textContent = labels[state] || "";
  }

  /* ── waveform visualisation ───────────────── */
  function updateWaveform() {
    if (!analyser) return;
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    const step = Math.floor(data.length / NUM_BARS);
    for (let i = 0; i < NUM_BARS; i++) {
      let sum = 0;
      for (let j = 0; j < step; j++) sum += Math.abs(data[i * step + j]);
      const avg = sum / step;
      const h = Math.max(4, Math.min(20, avg * 500));
      bars[i].style.height = h + "px";
    }
  }

  /* ── WAV encoding ─────────────────────────── */
  function mergeChunks(arr) {
    const len = arr.reduce((s, c) => s + c.length, 0);
    const out = new Float32Array(len);
    let off = 0;
    for (const c of arr) { out.set(c, off); off += c.length; }
    return out;
  }

  function downsample(samples, from, to) {
    if (from === to) return samples;
    const ratio = from / to;
    const n = Math.round(samples.length / ratio);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = samples[Math.round(i * ratio)];
    return out;
  }

  function encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0,"RIFF"); v.setUint32(4,36+samples.length*2,true);
    w(8,"WAVE"); w(12,"fmt "); v.setUint32(16,16,true);
    v.setUint16(20,1,true); v.setUint16(22,1,true);
    v.setUint32(24,sr,true); v.setUint32(28,sr*2,true);
    v.setUint16(32,2,true); v.setUint16(34,16,true);
    w(36,"data"); v.setUint32(40,samples.length*2,true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44+i*2, s < 0 ? s*0x8000 : s*0x7FFF, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  /* ── start / stop session ─────────────────── */
  async function startVoice() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch {
      voiceText.textContent = "Microphone access denied";
      return false;
    }
    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(micStream);

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      const data = new Float32Array(e.inputBuffer.getChannelData(0));
      if (voiceState === "RECORDING") {
        chunks.push(data);
      } else {
        preBuffer.push(data);
        if (preBuffer.length > PRE_BUFFER_COUNT) preBuffer.shift();
      }
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);

    voiceActive = true;
    speechFrames = 0;
    silenceStart = null;
    setVoiceState("LISTENING");
    startVAD();
    return true;
  }

  function stopVoice() {
    voiceActive = false;
    stopVAD();
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    if (processor) { processor.disconnect(); processor = null; }
    if (analyser) { analyser.disconnect(); analyser = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    chunks = []; preBuffer = [];
    setVoiceState("IDLE");
    bars.forEach(b => b.style.height = "4px");
  }

  /* ── VAD loop ─────────────────────────────── */
  function startVAD() {
    const buf = new Float32Array(analyser.fftSize);
    vadTimer = setInterval(() => {
      if (!voiceActive) return;
      analyser.getFloatTimeDomainData(buf);
      const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
      const isSpeech = rms > RMS_THRESHOLD;
      updateWaveform();

      if (voiceState === "LISTENING") {
        if (isSpeech) {
          speechFrames++;
          if (speechFrames >= SPEECH_CONFIRM) {
            chunks = [...preBuffer];
            preBuffer = [];
            speechFrames = 0;
            silenceStart = null;
            setVoiceState("RECORDING");
          }
        } else {
          speechFrames = 0;
        }
      } else if (voiceState === "RECORDING") {
        if (!isSpeech) {
          if (!silenceStart) silenceStart = Date.now();
          if (Date.now() - silenceStart >= SILENCE_END_MS) {
            finishRecording();
          }
        } else {
          silenceStart = null;
        }
      } else if (voiceState === "SPEAKING") {
        if (isSpeech) {
          speechFrames++;
          if (speechFrames >= SPEECH_CONFIRM + 2) {
            if (currentAudio) { currentAudio.pause(); currentAudio = null; }
            chunks = [...preBuffer];
            preBuffer = [];
            speechFrames = 0;
            silenceStart = null;
            setVoiceState("RECORDING");
          }
        } else {
          speechFrames = 0;
        }
      }
    }, VAD_INTERVAL_MS);
  }

  function stopVAD() {
    if (vadTimer) { clearInterval(vadTimer); vadTimer = null; }
  }

  /* ── process a finished recording ─────────── */
  async function finishRecording() {
    if (!voiceActive) return;
    setVoiceState("PROCESSING");
    const rate = audioCtx ? audioCtx.sampleRate : 16000;
    const pcm = mergeChunks(chunks);
    chunks = []; silenceStart = null;
    const final = rate !== 16000 ? downsample(pcm, rate, 16000) : pcm;
    const wav = encodeWav(final, 16000);

    try {
      const fd = new FormData();
      fd.append("audio", wav, "r.wav");
      const sttRes = await fetch("/api/stt", { method: "POST", body: fd });
      const sttData = await sttRes.json();

      if (!sttData.text) {
        if (voiceActive) setVoiceState("LISTENING");
        return;
      }

      addMessage("user", sttData.text);
      history.push({ role: "user", content: sttData.text });
      showTyping();

      const { text: botText, lang } = await chatRequest(history);
      removeTyping();
      addMessage("bot", botText || "No response received.");
      history.push({ role: "assistant", content: botText });

      if (botText && voiceActive) {
        const ttsRes = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: botText, lang }),
        });
        if (ttsRes.ok && voiceActive) {
          const ab = await ttsRes.arrayBuffer();
          await playTTS(ab);
        }
      }
    } catch {
      removeTyping();
      addMessage("bot", "An error occurred during voice processing.");
    }

    if (voiceActive && voiceState !== "RECORDING") {
      setVoiceState("LISTENING");
    }
  }

  function playTTS(audioBuffer) {
    return new Promise((resolve) => {
      if (!voiceActive) { resolve(); return; }
      setVoiceState("SPEAKING");
      const blob = new Blob([audioBuffer], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      currentAudio = new Audio(url);
      currentAudio.onended = () => {
        URL.revokeObjectURL(url);
        currentAudio = null;
        resolve();
      };
      currentAudio.onerror = () => { currentAudio = null; resolve(); };
      currentAudio.play().catch(() => { currentAudio = null; resolve(); });
    });
  }

  /* ════════════════════════════════════════════
     MODE TOGGLE
     ════════════════════════════════════════════ */
  modeSwitch.addEventListener("click", async () => {
    if (busy) return;
    const goingVoice = !voiceActive && !modeSwitch.classList.contains("on");

    if (goingVoice) {
      modeSwitch.classList.add("on");
      modeSwitch.setAttribute("aria-checked", "true");
      lblText.classList.remove("active");
      lblVoice.classList.add("active");
      textBar.style.display = "none";
      voiceBarEl.classList.add("visible");
      await startVoice();
    } else {
      stopVoice();
      modeSwitch.classList.remove("on");
      modeSwitch.setAttribute("aria-checked", "false");
      lblText.classList.add("active");
      lblVoice.classList.remove("active");
      textBar.style.display = "flex";
      voiceBarEl.classList.remove("visible");
    }
  });
})();
