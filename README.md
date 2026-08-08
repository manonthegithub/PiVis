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
