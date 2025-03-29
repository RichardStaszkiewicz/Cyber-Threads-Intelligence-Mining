podman build -t cti-threat-detection .
podman run -p 8000:8000 --name cti-system cti-threat-detection
