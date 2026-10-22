# Running the analysis service on a server

The service is a small FastAPI app (`server.py`) around the YOLO model. It
listens on **port 4445** and is meant to sit behind a Cloudflare tunnel.

Everything it needs is in this folder (`07-app`): the code, `weights/best.pt`,
`items/` (diagram pictures) and `references/` (the connectors to choose from).

## 1. Clone and set up

Python **3.12** is what the pinned versions were built with.

Linux / macOS:

```bash
git clone <repo-url>
cd <repo>/07-app
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone <repo-url>
cd <repo>\07-app
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

**Server without a GPU?** On Linux, `pip install torch` downloads the CUDA build
(several GB). The CPU build is much smaller and is all this nano model needs
(about 1 second per photo). Install torch first, then the rest:

```bash
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Run

```bash
export ANALYSE_API_KEY='a-long-random-secret'   # see "Security" below
./run_server.sh                                  # Linux / macOS
```

```powershell
$env:ANALYSE_API_KEY = 'a-long-random-secret'
.\run_server.ps1                                 # Windows (or run_server.cmd)
```

or, on any system, directly: `.venv/bin/python server.py`
(`.venv\Scripts\python server.py` on Windows).

Wait for `Uvicorn running on http://127.0.0.1:4445`. Loading the model takes
about 15 seconds. The first start also reads each photo in `references/` that has
no `.json` yet.

| Setting | Variable | Default |
| ------- | -------- | ------- |
| Port | `PORT` | `4445` |
| Address | `HOST` | `127.0.0.1` (this machine only; right for a tunnel) |
| API key | `ANALYSE_API_KEY` | none (open) |
| Match cut-off (0 to 1) | `MATCH_THRESHOLD` | `0.70` |
| Reference photos folder | `REFERENCES_DIR` | `references/` |
| Browser origins allowed | `ALLOWED_ORIGIN_REGEX` | localhost / 127.0.0.1 only |

Check it: `curl http://127.0.0.1:4445/health` should print `{"status":"ok"}`.

## 3. Expose it with Cloudflare

Keep `HOST` at `127.0.0.1`: `cloudflared` runs on the same machine and connects
to `http://localhost:4445`, so the service is only reachable through the tunnel.

Quick test (temporary random address):

```bash
cloudflared tunnel --url http://localhost:4445
```

Permanent (named tunnel), in the tunnel's `config.yml`:

```yaml
ingress:
  - hostname: connector.example.com
    service: http://localhost:4445
  - service: http_status:404
```

Then check `https://connector.example.com/health`.

Notes:

- Cloudflare closes a request that takes longer than about 100 seconds. A photo
  takes a few seconds, so this only matters if the server is very slow or busy.
- The service handles one photo at a time; others wait their turn.
- Run it as a service so it survives reboots and restarts on a crash
  (a `systemd` unit on Linux, or NSSM / Task Scheduler on Windows) with the same
  environment variables and working directory `07-app`.

## 4. Security

**Set `ANALYSE_API_KEY` before exposing the service.** Without it anyone who
finds the address can use your CPU and read the reference photos. Every endpoint
except `/health` then requires the header `X-API-Key: <the key>`; the service
logs a warning at start-up if no key is set.

Point the Flutter app at the tunnel when you build or run it:

```bash
flutter run \
  --dart-define=ANALYSIS_SERVER_URL=https://connector.example.com \
  --dart-define=ANALYSIS_API_KEY=a-long-random-secret
```

A key built into an app can be extracted from it, so treat it as a shield
against casual use, not as real user authentication. A Cloudflare Access policy
or service token in front of the hostname is a stronger option.

The Chrome (web) build is called from a browser, so the service must allow its
origin: set `ALLOWED_ORIGIN_REGEX`, e.g. `https://app\.example\.com`.

## 5. Updating

```bash
git pull
.venv/bin/python -m pip install -r requirements.txt   # only if requirements.txt changed
```

then restart the service. New or replaced photos in `references/` are picked up
without a restart (see the main README, "Adding or correcting a connector").
