const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const cameraOff = document.getElementById("camera-off");

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const enrollBtn = document.getElementById("enroll-btn");
const enrollStopBtn = document.getElementById("enroll-stop-btn");
const trainBtn = document.getElementById("train-btn");
const reloadBtn = document.getElementById("reload-btn");

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const facesText = document.getElementById("faces-text");
const fpsText = document.getElementById("fps-text");
const modelText = document.getElementById("model-text");
const enrollStatus = document.getElementById("enroll-status");
const peopleList = document.getElementById("people-list");

const thresholdInput = document.getElementById("threshold");
const thresholdVal = document.getElementById("threshold-val");
const minFaceInput = document.getElementById("min-face");
const minFaceVal = document.getElementById("min-face-val");
const enrollNameInput = document.getElementById("enroll-name");
const samplesInput = document.getElementById("samples");
const samplesVal = document.getElementById("samples-val");

const API = "";

let stream = null;
let running = false;
let detecting = false;
let enrolling = false;
let enrollSaved = 0;
let enrollTarget = 60;
let lastEnrollAt = 0;
let sessionId = "";
let frameTimes = [];
let detectTimer = null;

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

function setStatus(text, mode = "idle") {
    statusText.textContent = text;
    statusDot.className = "status-dot";
    if (mode === "live") statusDot.classList.add("live");
    if (mode === "error") statusDot.classList.add("error");
    if (mode === "enrolling") statusDot.classList.add("enrolling");
}

function syncOverlaySize() {
    if (!video.videoWidth) return;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
}

async function api(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || data.reason || `Request failed (${response.status})`);
    }
    return data;
}

async function captureBlob() {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0);
    return new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.85));
}

function updateFps() {
    const now = performance.now();
    frameTimes.push(now);
    if (frameTimes.length > 30) frameTimes.shift();
    if (frameTimes.length >= 2) {
        const elapsed = frameTimes[frameTimes.length - 1] - frameTimes[0];
        const fps = ((frameTimes.length - 1) / elapsed) * 1000;
        fpsText.textContent = `${fps.toFixed(0)} fps`;
    }
}

function drawTracks(tracks) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    for (const track of tracks) {
        let [x, y, w, h] = track.box;
        x = overlay.width - x - w;
        const known = track.name !== "Unknown";
        ctx.strokeStyle = known ? "#28be78" : "#00aaff";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        const conf = track.confidence == null ? "" : ` ${Math.round(track.confidence)}`;
        const label = `ID ${track.track_id}: ${track.name}${conf}`;
        ctx.font = "16px Segoe UI, Arial, sans-serif";
        const textWidth = ctx.measureText(label).width;
        const labelY = Math.max(22, y - 8);
        ctx.fillStyle = known ? "#28be78" : "#00aaff";
        ctx.fillRect(x, labelY - 20, textWidth + 10, 22);
        ctx.fillStyle = "#101820";
        ctx.fillText(label, x + 5, labelY - 4);
    }
}

async function detectLoop() {
    if (!running || !video.videoWidth) {
        detecting = false;
        return;
    }

    detecting = true;
    try {
        const blob = await captureBlob();
        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        form.append("session_id", sessionId);

        const data = await api(
            `/api/detect?threshold=${thresholdInput.value}&min_face=${minFaceInput.value}`,
            { method: "POST", body: form }
        );

        sessionId = data.session_id || sessionId;
        drawTracks(data.tracks || []);
        facesText.textContent = `${data.faces} faces · ${(data.tracks || []).length} tracks`;
        updateFps();
        setStatus("Live", enrolling ? "enrolling" : "live");
    } catch (err) {
        console.error(err);
        setStatus(err.message, "error");
    }

    detectTimer = setTimeout(detectLoop, 120);
}

async function enrollLoop() {
    if (!enrolling || !running) return;

    const now = performance.now();
    if (now - lastEnrollAt < 180) {
        requestAnimationFrame(enrollLoop);
        return;
    }

    try {
        const name = enrollNameInput.value.trim();
        const blob = await captureBlob();
        const form = new FormData();
        form.append("file", blob, "frame.jpg");

        const data = await api(
            `/api/enroll/${encodeURIComponent(name)}?min_face=${minFaceInput.value}`,
            { method: "POST", body: form }
        );

        if (data.saved) {
            enrollSaved += 1;
            lastEnrollAt = now;
            enrollStatus.textContent = `${data.person}: ${enrollSaved}/${enrollTarget}`;
            await refreshPeople();
        } else {
            enrollStatus.textContent = data.reason || "Waiting for one face";
        }

        if (enrollSaved >= enrollTarget) {
            stopEnroll(true);
            setStatus("Enrollment complete — train the model", "live");
        }
    } catch (err) {
        enrollStatus.textContent = err.message;
    }

    if (enrolling) requestAnimationFrame(enrollLoop);
}

async function startCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraOff.style.display = "none";
    syncOverlaySize();
    video.onloadedmetadata = syncOverlaySize;
}

async function startTracking() {
    if (running) return;
    try {
        if (!stream) await startCamera();
        running = true;
        sessionId = "";
        frameTimes = [];
        startBtn.disabled = true;
        stopBtn.disabled = false;
        enrollBtn.disabled = false;
        setStatus("Live", "live");
        detectLoop();
    } catch (err) {
        setStatus(`Camera error: ${err.message}`, "error");
    }
}

async function stopTracking() {
    running = false;
    if (detectTimer) clearTimeout(detectTimer);
    stopEnroll(false);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    enrollBtn.disabled = true;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    facesText.textContent = "0 faces · 0 tracks";
    fpsText.textContent = "";
    setStatus("Stopped", "idle");

    if (sessionId) {
        const form = new FormData();
        form.append("session_id", sessionId);
        fetch(`${API}/api/reset-tracker`, { method: "POST", body: form }).catch(() => {});
        sessionId = "";
    }
}

function startEnroll() {
    const name = enrollNameInput.value.trim();
    if (!name) {
        enrollStatus.textContent = "Enter a name first";
        return;
    }
    if (!running) {
        enrollStatus.textContent = "Start the camera first";
        return;
    }

    enrolling = true;
    enrollSaved = 0;
    enrollTarget = Number(samplesInput.value);
    lastEnrollAt = 0;
    enrollBtn.disabled = true;
    enrollStopBtn.disabled = false;
    enrollStatus.textContent = `Enrolling ${name}: 0/${enrollTarget}`;
    setStatus("Enrolling", "enrolling");
    enrollLoop();
}

function stopEnroll(completed) {
    enrolling = false;
    enrollBtn.disabled = !running;
    enrollStopBtn.disabled = true;
    if (!completed) {
        enrollStatus.textContent = enrollSaved ? `Stopped at ${enrollSaved} samples` : "Idle";
    }
}

async function trainModel() {
    trainBtn.disabled = true;
    setStatus("Training…", "live");
    try {
        const data = await api("/api/train", { method: "POST" });
        modelText.textContent = data.loaded
            ? `loaded · ${data.people} person(s)`
            : "not loaded";
        setStatus("Training complete", "live");
        await refreshPeople();
    } catch (err) {
        setStatus(err.message, "error");
    } finally {
        trainBtn.disabled = false;
    }
}

async function refreshModel() {
    try {
        const data = await api("/api/model");
        modelText.textContent = data.loaded
            ? `loaded · ${data.people} person(s)`
            : "not loaded";
    } catch (err) {
        modelText.textContent = "server offline";
    }
}

async function refreshPeople() {
    try {
        const data = await api("/api/people");
        peopleList.innerHTML = "";
        if (!data.people.length) {
            peopleList.innerHTML = '<li class="muted">No enrolled people</li>';
            return;
        }
        for (const person of data.people) {
            const li = document.createElement("li");
            li.innerHTML = `<span>${person.name}</span><span>${person.count}</span>`;
            peopleList.appendChild(li);
        }
    } catch (err) {
        peopleList.innerHTML = '<li class="muted">Could not load people</li>';
    }
}

thresholdInput.addEventListener("input", () => {
    thresholdVal.textContent = thresholdInput.value;
});
minFaceInput.addEventListener("input", () => {
    minFaceVal.textContent = minFaceInput.value;
});
samplesInput.addEventListener("input", () => {
    samplesVal.textContent = samplesInput.value;
});

startBtn.addEventListener("click", startTracking);
stopBtn.addEventListener("click", stopTracking);
enrollBtn.addEventListener("click", startEnroll);
enrollStopBtn.addEventListener("click", () => stopEnroll(false));
trainBtn.addEventListener("click", trainModel);
reloadBtn.addEventListener("click", async () => {
    await refreshModel();
    await refreshPeople();
    setStatus("Reloaded", running ? "live" : "idle");
});

window.addEventListener("beforeunload", () => {
    stopTracking();
    if (stream) stream.getTracks().forEach((track) => track.stop());
});

refreshModel();
refreshPeople();
