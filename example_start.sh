#!/bin/bash
trap 'kill $(jobs -p)' EXIT
source .venv/bin/activate
python app.py &
sleep 3
ngrok http --domain=your-custom-domain 8888