# Windsurf Anweisungen (Autonom ausführbar)

## Grundsatz
- Verwende **keine** `set -euo pipefail` (oder nur sehr sparsam), damit ein einzelner 404/500 den Lauf nicht abbricht.
- Nutze `curl -sS` und bei jedem Abschnitt **immer** einen sichtbaren Header/Status-Print.
- Keine ES-Index-Änderungen. Erlaubt sind:
  - GET /
  - GET _cluster/health
  - GET/POST rag_files_v1/_count, rag_files_v1/_search
  - Agent: GET /health, GET /v1/models, POST /v1/chat/completions, GET /open?path=...

## Ausführen (empfohlen)
```bash
bash scripts/smoke_small.sh
# bei Release:
bash scripts/smoke_release_train.sh
```

## Interpretation
- Small Suite ist die Default-Gate für jeden Schritt.
- Release Train Suite nur vor Tags/Releases oder nach Reboot/Upgrade.

## /open Signatur (wichtig)
- `GET /open?path=<urlencoded absolute path>`
- Kein POST. Kein `q`/`k` Parameter.
Beispiel:
`http://localhost:11436/open?path=/media/felix/RAG/1/<...>/file.pdf`
