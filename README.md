# PiVis

Raspberry Pi camera stream with local people detection and Claude-powered voice greetings.

## Requirements

- Raspberry Pi 5 (4GB+) with Pi Camera Module
- Python 3.11+
- `piper` TTS binary — [install guide](https://github.com/rhasspy/piper#installation)
- `alsa-utils` for local speaker output (`sudo apt install alsa-utils`)
- `libcamera` / `picamera2` (`sudo apt install python3-picamera2`)

## Setup

```bash
git clone <repo> && cd pivis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download YOLO + Piper voice models (~120MB total)
bash scripts/download_models.sh

# Configure
cp .env.example .env
nano .env  # set ANTHROPIC_API_KEY at minimum
```

## Run

```bash
source .venv/bin/activate
python -m pivis
```

Open `http://<pi-ip>:8000` in a browser.

**Mic input ("Talk to PiVis") needs HTTPS.** Browsers only expose
`navigator.mediaDevices.getUserMedia` in a secure context (HTTPS, or literal
`localhost`) — plain `http://<pi-ip>:8000` fails with `Cannot read
properties of undefined (reading 'getUserMedia')`. A reverse proxy
(the shared nginx instance) fronts the Pi at `https://pivis.apps.arpa` with
a self-signed cert for this reason; use that URL instead if you want the mic
button to work. Everything else (video stream, greetings, lighting) works
fine over plain HTTP too.

## Audio output

Set `AUDIO_OUTPUT` in `.env`:

| Value | Behaviour |
|---|---|
| `browser` | Plays through the open browser tab |
| `local` | Plays through Pi speaker via `aplay` |
| `both` | Both simultaneously (default) |

For local output, set `AUDIO_DEVICE` to your `aplay` device (e.g. `plughw:0,0`). Run `aplay -L` to list devices.

## Run as a service (auto-start on boot)

```bash
# Edit WorkingDirectory and User in pivis.service to match your setup
sudo cp pivis.service /etc/systemd/system/
sudo systemctl enable --now pivis
sudo journalctl -fu pivis  # tail logs
```

## Dev / off-Pi testing

Tests mock `picamera2`, `piper`, `aplay`, and the Claude API — no hardware needed:

```bash
ANTHROPIC_API_KEY=test pytest
```

## Environment variables

See `.env.example` for the full list with defaults.

## Audio module — container / k8s deployment

The standalone audio module (`pivis/audio_app.py`, browser mic → WebSocket →
Whisper STT) has no camera dependency and is built as its own image,
separate from the main Pi-only app above.

- **Build**: `.github/workflows/docker-build.yml` builds `Dockerfile.audio`
  and pushes to GHCR (`ghcr.io/<owner>/pivis/audio`) on every push to `main`
  that touches the audio module, or via manual dispatch. Docker layer
  caching uses the GitHub Actions cache (`type=gha`), so unchanged layers
  (deps, the baked-in Whisper model) are reused on subsequent builds instead
  of rebuilt.
- **Deploy**: manifests in `k8s/audio/` (`namespace`, `configmap`,
  `deployment`, `service`, `ingress`) pull the image straight from GHCR:
  ```bash
  kubectl apply -f k8s/audio/namespace.yaml
  kubectl apply -f k8s/audio/
  ```
  If the `ghcr.io/<owner>/pivis/audio` package is private, create the pull
  secret referenced in `k8s/audio/deployment.yaml` first (see the comment
  there) — or set the package's visibility to public in GitHub and drop the
  `imagePullSecrets` block.
- **Expose**: `k8s/audio/ingress.yaml` routes `pivis-aud.apps.arpa` to the
  service via the cluster's Traefik ingress controller. The shared nginx
  reverse proxy terminates TLS for `https://pivis-aud.apps.arpa` (self-signed
  cert, shared with `pivis.apps.arpa` below) and forwards to the Traefik
  LoadBalancer on the cluster nodes' port 80. TLS here is required, not just
  nice-to-have: once the main app is HTTPS (see above), browsers block a
  plain `ws://` connection from an `https://` page as mixed content, so this
  has to be `wss://`.
- **Main app wiring**: set `AUDIO_SERVER_WS_URL=wss://pivis-aud.apps.arpa`
  (see `.env.example`) on the Pi-hosted main app. The main app's `/config`
  endpoint exposes it to the browser, which then connects
  `pivis/web/static/audio-client.js` to
  `<AUDIO_SERVER_WS_URL>/ws/audio/<stream_id>` for live mic transcription
  (the "Talk to PiVis" button, only visible when the setting is non-empty).
- **Local build**: `docker build -f Dockerfile.audio -t pivis-audio .`
