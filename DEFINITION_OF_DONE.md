# Definition of Done (DoD)

## DoD: Schritt-Implementierung (Small Suite = Pflicht)
Ein Umsetzungsschritt gilt als "done", wenn:
1) `scripts/smoke_small.sh` läuft ohne Abbruch durch und endet mit Exit Code 0
2) Kernservices liefern erwartete HTTP Codes:
   - ES: 200
   - Agent /health: 200
   - Ollama /api/tags: 200
   - WebUI: 200
3) ES liefert Count für `rag_files_v1` und eine Sample-Search ist möglich (read-only)
4) Agent Chat funktioniert (LLM-only), und eine RAG-Frage liefert nachvollziehbare Quellen-Links (`/open?...`), oder dokumentiert sauber "keine Quellen gefunden".

## DoD: Release Train (Large Suite = vor Tag/Release)
Ein Release gilt als "ready", wenn zusätzlich:
1) `scripts/smoke_release_train.sh` läuft komplett durch (Exit 0)
2) ES-Aggregationen (ext/mime) funktionieren (read-only)
3) Content-Sanity-Checks bestehen (Anteil leerer Inhalte unter Schwellwert)
4) Stichproben-RAG: mehrere Queries liefern:
   - mind. N Antworten mit Quellen (Konfigurationswert)
   - keine offensichtlichen "Halluzinations-Quellen" (Quellen sind plausibel zur Query)
5) `/open` endpoint ist verifizierbar (GET mit `path=`); mindestens 1 Datei wird erfolgreich served (HTTP 200/206)

## Leitplanke
Keine ES-Index-Änderungen als Teil der Tests (kein DELETE/PUT auf Indizes).
