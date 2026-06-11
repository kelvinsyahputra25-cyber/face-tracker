const API = "";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const faceList = document.getElementById("faceList");
const peopleList = document.getElementById("peopleList");

let running = false;
let enrolling = false;

let fpsCounter = 0;
let lastTime = performance.now();

document.getElementById("startBtn").onclick = startCamera;
document.getElementById("trainBtn").onclick = trainModel;
document.getElementById("reloadBtn").onclick = loadPeople;
document.getElementById("enrollBtn").onclick = toggleEnroll;

async function startCamera() {  

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

    video.srcObject = stream;

    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!running) {
        running = true;
        detectLoop();
    }
}

async function toggleEnroll() {

    const name =
        document.getElementById("personName").value.trim();

    if (!name) {
        alert("Enter a name first");
        return;
    }

    enrolling = !enrolling;

    document.getElementById("enrollBtn").textContent =
        enrolling ? "Stop Enroll" : "Start Enroll";

    document.getElementById("enrollStatus").textContent =
        enrolling ? `Enrolling ${name}` : "Idle";
}

async function detectLoop() {

    if (!running) return;

    const temp = document.createElement("canvas");

    temp.width = video.videoWidth;
    temp.height = video.videoHeight;

    const tctx = temp.getContext("2d");
    tctx.drawImage(video, 0, 0);

    const blob =
        await new Promise(resolve =>
            temp.toBlob(resolve, "image/jpeg", 0.85)
        );

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {

        const response =
            await fetch(API + "/detect", {
                method: "POST",
                body: formData
            });

        const data = await response.json();

        drawFaces(data.faces);

        if (
            enrolling &&
            data.faces &&
            data.faces.length === 1
        ) {

            const name =
                document.getElementById("personName").value.trim();

            await sendEnrollFrame(blob, name);
        }

    } catch (err) {

        console.error(err);

    }

    setTimeout(detectLoop, 150);
}

function drawFaces(faces) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    faceList.innerHTML = "";

    faces.forEach(face => {

        const [x, y, w, h] = face.box;

        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 3;

        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "#22c55e";
        ctx.font = "18px Segoe UI";

        ctx.fillText(
            face.name,
            x,
            y - 10
        );

        const li = document.createElement("li");

        li.textContent =
            `${face.name} (${face.confidence.toFixed(1)})`;

        faceList.appendChild(li);
    });

    fpsCounter++;

    const now = performance.now();

    if (now - lastTime > 1000) {

        document.getElementById("fps").textContent =
            `FPS: ${fpsCounter}`;

        fpsCounter = 0;
        lastTime = now;
    }
}

async function sendEnrollFrame(blob, name) {

    const formData = new FormData();

    formData.append(
        "file",
        blob,
        "face.jpg"
    );

    try {

        await fetch(
            `${API}/enroll/${encodeURIComponent(name)}`,
            {
                method: "POST",
                body: formData
            }
        );

    } catch (err) {

        console.error(err);

    }
}

async function trainModel() {

    try {

        const response =
            await fetch(API + "/train", {
                method: "POST"
            });

        const data =
            await response.json();

        if (data.ok) {

            alert(
                `Training Complete\nPeople: ${data.people}`
            );

        } else {

            alert(
                data.error || "Training Failed"
            );
        }

        loadPeople();

    } catch (err) {

        console.error(err);

    }
}

async function loadPeople() {

    try {

        const response =
            await fetch(API + "/people");

        const data =
            await response.json();

        peopleList.innerHTML = "";

        data.people.forEach(person => {

            const li =
                document.createElement("li");

            li.textContent =
                `${person.name} (${person.count})`;

            peopleList.appendChild(li);
        });

    } catch (err) {

        console.error(err);

    }
}

loadPeople();
