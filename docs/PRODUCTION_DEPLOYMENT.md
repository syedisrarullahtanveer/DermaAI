# DermaAI High-Availability Production Deployment Guide

## Architecture
- Kubernetes ingress with TLS termination and HMAC client validation.
- Horizontal autoscaling based on CPU and GPU inference queue depths.
- Multi-tier caching layer (Redis / In-Memory LRU) for instant duplicate hash lookups.
- Automated Prometheus / Grafana observability dashboards.
