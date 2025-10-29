#!/bin/bash

# Quick CPU burner script using stress-ng
# Burns CPU with 4 workers for 60 seconds
# Usage: ./cpu_burner.sh

echo "Starting CPU burn with stress-ng..."
stress-ng --cpu 4 --timeout 60s
echo "CPU burn complete."