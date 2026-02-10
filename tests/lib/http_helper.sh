#!/bin/bash
# HTTP Helper Library - No hangs, no jq dependency

# Get HTTP code only
http_code() {
    local url="$1"
    curl --connect-timeout 2 --max-time 10 -sS -o /dev/null -w "%{http_code}" --retry 0 "$url" 2>/dev/null || echo "000"
}

# GET request to file
http_get() {
    local url="$1"
    local outfile="$2"
    curl --connect-timeout 2 --max-time 10 -sS -o "$outfile" -w "%{http_code}" --retry 0 "$url" 2>/dev/null || echo "000"
}

# POST JSON request to file
http_post_json() {
    local url="$1"
    local json="$2"
    local outfile="$3"
    curl --connect-timeout 2 --max-time 10 -sS -o "$outfile" -w "%{http_code}" \
         -H "Content-Type: application/json" -d "$json" --retry 0 "$url" 2>/dev/null || echo "000"
}
