#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/frontend"
npm install
exec npm run dev
