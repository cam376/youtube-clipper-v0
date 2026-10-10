# Deployment — private beta on one Linux server (v0.2.1)

This puts the existing app online for 5-10 trusted testers. Nothing else:
no accounts, no billing, no database, no queue service. The whole thing is
two containers (the app and Ollama) plus an HTTPS reverse proxy on the host.

## 1. Server specs

Processing is CPU-bound. Measured on the Windows test machine with Whisper
`small`: a 27-minute interview takes about 20 minutes end to end; with
`medium` a 23m30 video took about 50 minutes for a slight caption gain, so
`small` is the production default. One job runs at a time by default.

| | Minimum (5-10 testers, a few videos a day) | Recommended |
|---|---|---|
| CPU | 4 dedicated vCPU (x86-64, AVX2) | 8 vCPU |
| RAM | 8 GB | 16 GB |
| Disk | 60 GB SSD | 160 GB NVMe |
| OS | Ubuntu 24.04 LTS or Debian 12/13, 64-bit | same |
| Network | 100 Mbit/s | 1 Gbit/s |
| GPU | none | none needed for this version |

Memory budget at peak: Whisper `small` int8 ~1 GB, Qwen2.5 3B ~2.5 GB,
ffmpeg + OpenCV ~1 GB, OS + Docker ~1 GB. `medium` adds ~1.5 GB. Disk per
job: source video (300 MB to 1 GB for 20-30 min at 1080p) + audio + clips,
so budget about 1 GB per job plus 4 GB for models. Shared-CPU "burstable"
VPS plans will be noticeably slower than the timings above.

## 2. Install Docker

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker --version && docker compose version
```

## 3. Get the code and build

```bash
git clone https://github.com/cam376/youtube-clipper-v0.git
cd youtube-clipper-v0
git checkout v0.2.1          # or the branch you deploy from
docker compose build
```

The image is Debian trixie + Python 3.12 + FFmpeg 7.1 (with libass) +
Liberation fonts (used for the burned "Arial" captions) + the pinned Python
dependencies from `requirements-lock.txt`.

## 4. Start, pull the Qwen model, check health

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:3b     # once, ~2 GB, kept in a volume
curl -s http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","version":"0.2.1","whisper_model":"small","ollama":"ok",...}`.
If `ollama` says `unreachable`, the ranking falls back to the heuristic and
the page says so; fix Ollama rather than shipping clips ranked that way.

The first job downloads the Whisper `small` model (about 480 MB) into the
`clipper_data` volume; later jobs start immediately.

## 5. Environment variables

Set them in `docker-compose.yml` under `app.environment`, or in a `.env`
file next to it for the `${...}` ones.

| Variable | Default in the image | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | bind address inside the container (local dev uses 127.0.0.1) |
| `PORT` | `8000` | port inside the container |
| `OUTPUT_DIR` | `/data/output` | where jobs and clips are written |
| `HF_HOME` | `/data/models` | Whisper model cache |
| `WHISPER_MODEL` | `small` | production default. `medium` = optional high-accuracy mode, ~2.5x slower |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | `cpu` / `int8` | leave as is on a CPU server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | must be pulled in the Ollama container |
| `OLLAMA_URL` | `http://ollama:11434` | the compose service name |
| `MAX_CONCURRENT_JOBS` | `1` | jobs processed at once; others show "Waiting for a free slot..." |
| `DEBUG_FACES` | `false` | `true` writes face-tracking contact sheets per clip (development only) |
| `ROOT_PATH` | empty | URL prefix when hosted under a sub-path (section 8b), e.g. `/vezly.ai` |

To try the high-accuracy mode for everyone:
```bash
WHISPER_MODEL=medium docker compose up -d
```
and back:
```bash
WHISPER_MODEL=small docker compose up -d
```

## 6. Persistent storage

Two named volumes survive restarts, rebuilds and `docker compose down`:

| Volume | Contents | Host path |
|---|---|---|
| `clipper_data` | `/data/output/<job_id>/` (source, audio, transcript, clips) and `/data/models` (Whisper) | `docker volume inspect youtube-clipper-v0_clipper_data` |
| `ollama_models` | Qwen weights | `docker volume inspect youtube-clipper-v0_ollama_models` |

Generated videos: `/data/output/<job_id>/clip_N.mp4` inside the container.
To copy one out: `docker cp clipper:/data/output/<job_id>/clip_1.mp4 .`

To use a host directory instead of a named volume, replace the volume line
with `- /srv/clipper-data:/data` and run
`sudo mkdir -p /srv/clipper-data && sudo chown -R 1000:1000 /srv/clipper-data`
first (the app runs as uid 1000).

Disk cleanup is manual in this version. Jobs are never deleted by the app.
A weekly cron that removes job folders older than 14 days:
```bash
docker compose exec app sh -c 'find /data/output -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +'
```

Only `clip_N.mp4` files are served over HTTP. Source videos, transcripts,
filter scripts and debug sheets stay on disk for the operator.

## 7. Ollama in production

The compose file runs the official `ollama/ollama` image next to the app,
CPU only, not published on any port. This is the simplest reliable setup:
one `ollama pull` once, the weights live in the `ollama_models` volume, and
the app reaches it as `http://ollama:11434`.

If you already run Ollama on the host instead, delete the `ollama` service
and set `OLLAMA_URL=http://host.docker.internal:11434` plus
`extra_hosts: ["host.docker.internal:host-gateway"]` on the app service.

Check it answers: `docker compose exec ollama ollama list`.

## 8. HTTPS, domain and keeping the beta private

The app has no login. For a private beta the reverse proxy is the gate:
Caddy terminates HTTPS with an automatic Let's Encrypt certificate and asks
for a single shared password.

1. Point a DNS A record (for example `clips.example.com`) at the server.
2. Open ports 80 and 443 in the firewall; keep 8000 closed (the app only
   listens on 127.0.0.1 anyway).
3. Install Caddy on the host: https://caddyserver.com/docs/install#debian-ubuntu-raspbian
4. Create the password hash: `caddy hash-password` (type the beta password).
5. `/etc/caddy/Caddyfile`:
   ```
   clips.example.com {
       basic_auth {
           beta <paste the hash here>
       }
       reverse_proxy 127.0.0.1:8000
       encode gzip
   }
   ```
6. `sudo systemctl reload caddy`, then open https://clips.example.com and
   log in with user `beta`.

Give each tester the password privately. Rotate it by rerunning step 4-6.

## 8b. Serving under a sub-path (settermonster.com/vezly.ai)

The app can live under a path prefix on an existing domain instead of its
own host. Every link it emits is relative to the page, so the only
requirements are: the reverse proxy strips the prefix before forwarding,
the page is reached with a trailing slash (`/vezly.ai/`, not `/vezly.ai`),
and `ROOT_PATH` is set so FastAPI's `/docs` links are right.

Set it in `.env` next to `docker-compose.yml`:
```
ROOT_PATH=/vezly.ai
```

Caddy, added inside the existing `settermonster.com { ... }` site block:
```
settermonster.com {
    # ... whatever already serves the main site ...

    redir /vezly.ai /vezly.ai/ permanent
    handle_path /vezly.ai/* {
        basic_auth {
            beta <hash from: caddy hash-password>
        }
        reverse_proxy 127.0.0.1:8000
    }
}
```
`handle_path` strips `/vezly.ai` so the app sees `/`, `/api/...`, `/output/...`.

Nginx equivalent:
```
location = /vezly.ai { return 301 /vezly.ai/; }
location /vezly.ai/ {
    auth_basic "Kivro beta";
    auth_basic_user_file /etc/nginx/kivro.htpasswd;
    proxy_pass http://127.0.0.1:8000/;          # trailing slash = strip the prefix
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 600;
    client_max_body_size 50m;
}
```

Check: `https://settermonster.com/vezly.ai/health` must answer with
`"root_path": "/vezly.ai"`, and the page must load its styles and list jobs.

A dot inside a path (`vezly.ai`) is legal, but if you own `vezly.ai` as a
domain, a host is cleaner and avoids the trailing-slash rule entirely:
`app.vezly.ai` or `vezly.settermonster.com` with the plain block from
section 8 and `ROOT_PATH` left empty. Both setups work with the same build.

## 9. Operate

| Task | Command (from the repo directory) |
|---|---|
| Logs | `docker compose logs -f app` |
| Restart app | `docker compose restart app` |
| Stop everything | `docker compose down` (volumes are kept) |
| Update to a new version | `git pull && docker compose build && docker compose up -d` |
| Update yt-dlp only | bump `yt-dlp==` in `requirements-lock.txt`, then the update command |
| Health | `curl -s http://127.0.0.1:8000/health` |
| Run the test suite in the image | `docker compose run --rm -v "$PWD/tests:/srv/clipper/tests:ro" app python -m unittest discover -s tests -v` |

A restart ends any job in progress; the tester has to resubmit. Finished
clips stay on disk, but the in-memory job list is empty after a restart, so
the page cannot show old jobs. The clip URLs
`/output/<job_id>/clip_N.mp4` keep working if the tester still has them.

## 10. Limitations of this V0 deployment

- **YouTube downloads from a datacenter IP are the biggest risk.** YouTube
  often answers VPS IPs with "Sign in to confirm you're not a bot". If that
  happens the job fails at "Importing video..." with that message. This
  version does not work around it. Options, in order: update yt-dlp (it is
  pinned in `requirements-lock.txt`), host at a provider whose IPs are not
  flagged, or run the app on a residential connection for the beta.
- No accounts: everyone behind the Caddy password shares one app, sees only
  their own job id, and nothing stops a tester from starting several jobs.
- One job at a time by default; a second tester waits with the status
  "Waiting for a free slot...". Raising `MAX_CONCURRENT_JOBS` only helps on
  a machine with the cores and RAM to match.
- Job state lives in memory; a restart drops it.
- No automatic cleanup of `/data/output`.
- CPU only. A 27-minute video takes about 20 minutes on a desktop-class CPU
  and longer on a shared VPS.
- yt-dlp needs regular updates to keep working with YouTube.
- The engine is frozen at v0.2.1; see CHANGELOG.md.
