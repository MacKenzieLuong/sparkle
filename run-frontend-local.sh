#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/frontend"
npm install
# Arguments are forwarded to vite, so `./run-frontend-local.sh --host` serves
# the dashboard on the LAN instead of localhost only. Note that the browser
# only grants microphone access on a secure origin: reached over plain http at
# a LAN address the dashboard loads and the camera still streams, but
# push-to-talk stops working. Keep it on localhost, or tunnel to it, if you
# need the microphone.
exec npm run dev -- "$@"
