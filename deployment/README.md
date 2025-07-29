# Kubernetes Pattern Detector Deployment

This directory contains Kubernetes manifests and deployment scripts for the Mjolnir Pattern Detector service.

## Overview

The K8s Pattern Detector is a containerized service that:
- Connects to Redis to retrieve the list of stocks to monitor
- Subscribes to WebSocket tick data for each stock
- Runs pattern detection algorithms on incoming data
- Publishes detected patterns back to the WebSocket server

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│     Redis       │◄───┤  Pattern        │───►│   WebSocket     │
│  (Stock List)   │    │  Detector       │    │    Server       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Files

### Kubernetes Manifests
- `deployment.yaml` - Main application deployment
- `service.yaml` - ClusterIP service for health checks

### Container Build
- `../Dockerfile` - Container image definition for Argo builds

### Application
- `../k8s_pattern_detector.py` - Main application code

## Configuration

The service is configured through environment variables from existing ConfigMaps/Secrets and hardcoded values:

### External ConfigMap (`stock-poller-config`)
- `REDIS_HOST` - Redis server hostname
- `REDIS_PORT` - Redis server port

### External Secret (`redis`)
- `redis-password` - Redis authentication password

### Environment Variables
- `ENVIRONMENT` - Runtime environment (production/development) - controls WebSocket URL selection
- `STOCK_LIST_KEY` - Redis key pattern for stock lists (`scanner:latest`)
- `WEBSOCKET_URL` - Optional override for WebSocket server URL (auto-derived if not set)

### Code Constants (in k8s_pattern_detector.py)
- `PATTERN_MIN_CONFIDENCE = 0.65` - Minimum confidence threshold for pattern detection
- `PATTERN_TOLERANCE = 0.01` - Price tolerance for pattern matching
- `PATTERN_LOOKBACK_WINDOW = 50` - Number of candles to keep in history
- `LOG_LEVEL = 'INFO'` - Application logging level
- `stock_refresh_interval = 15` - Redis stock list refresh interval (seconds)

## Redis Data Format

The service expects stock lists in Redis with the following key pattern:
```
scanner:latest:AAPL
scanner:latest:MSFT
scanner:latest:NVDA
...
```

Each key contains a payload with stock data:
```json
{
  "ticker": "AAPL",
  "price": 150.25,
  "prev_price": 149.80,
  "volume": 1000,
  "mav10": 148.50,
  "float": 15000000000,
  "delta": 0.45,
  "multiplier": 1.0,
  "timestamp": 1234567890
}
```

## WebSocket Protocol

### Connection
No explicit subscription required - the WebSocket server automatically sends data for all stocks upon connection.

### Incoming Tick Data
```json
{
  "topic": "stock:AAPL",
  "data": {
    "ticker": "AAPL",
    "price": 150.25,
    "prev_price": 149.80,
    "volume": 1000,
    "mav10": 148.50,
    "float": 15000000000,
    "delta": 0.45,
    "multiplier": 1.0,
    "timestamp": 1234567890
  },
  "sync": {
    "candle_state": {...},
    "server_time": 1234567890,
    "next_candle_boundary": 1234567920
  }
}
```

### Outgoing Pattern Detection
```json
{
  "type": "pattern_detected",
  "data": {
    "symbol": "AAPL",
    "pattern_type": "BULLISH_REVERSAL",
    "confidence": 0.7500,
    "trigger_price": 150.25,
    "timestamp": "2024-01-01T12:00:00.000000",
    "candle_data": {
      "open": 149.80,
      "high": 150.50,
      "low": 149.50,
      "close": 150.25,
      "volume": 1000
    },
    "source": "k8s_pattern_detector"
  }
}
```

## Deployment

### Prerequisites
1. Kubernetes cluster with `stock` namespace
2. Existing Redis deployment with `stock-poller-config` ConfigMap
3. Existing WebSocket server
4. Image pull secret `regcred` configured
5. Argo CD configured for automated deployments

### Environment-Based WebSocket Configuration

The service automatically selects the correct WebSocket URL based on the `ENVIRONMENT` variable:

- **Production** (`ENVIRONMENT=production`): `ws://hermes.tekuro.io`
- **Development** (`ENVIRONMENT=development`): `wss://hermes.tekuro.io` (TLS terminated at k8s)
- **Override**: Set `WEBSOCKET_URL` environment variable to use a custom URL

### Deployment via Argo

Argo CD handles:
- Building the container image from `Dockerfile`
- Applying Kubernetes manifests from `deployment/`
- Managing environment-specific configurations

To deploy to different environments, update the `ENVIRONMENT` value in `deployment.yaml`:

```yaml
- name: ENVIRONMENT
  value: "production"  # or "development"
```

### Health Checks

The service exposes health check endpoints on port 8080:
- `/health` - Overall health status
- `/ready` - Readiness for traffic
- `/startup` - Startup completion status
- `/metrics` - Service metrics and statistics

## Environment-Specific Configuration

### Production
- Uses production WebSocket endpoints
- Higher resource limits
- Production logging levels

### Development
- Uses development WebSocket endpoints
- Debug logging enabled
- Lower resource requests

Set the environment during build:
```bash
./build.sh development v1.0.0-dev
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis host/port configuration
   - Verify Redis password is correct
   - Ensure network connectivity

2. **WebSocket Connection Failed**
   - Verify WebSocket URL is reachable
   - Check firewall/network policies
   - Confirm WebSocket server is running

3. **No Stocks Monitored**
   - Verify Redis contains stock keys in expected format
   - Check Redis permissions
   - Monitor logs for stock list refresh messages

4. **Pattern Detection Not Working**
   - Verify tick data is being received
   - Check pattern confidence thresholds
   - Review candle history accumulation

### Debugging Commands

```bash
# Check pod status
kubectl get pods -l app=mjolnir-pattern-detector -n stock

# View logs
kubectl logs deployment/mjolnir-pattern-detector -n stock

# Check configuration
kubectl describe configmap mjolnir-config -n stock

# Port forward for local testing
kubectl port-forward service/mjolnir-pattern-detector-service 8080:8080 -n stock

# Check metrics
curl http://localhost:8080/metrics
```

## Scaling

The service is designed to run as a single instance per stock symbol set. To scale:

1. **Horizontal Scaling**: Deploy multiple instances with different stock symbol subsets
2. **Resource Scaling**: Adjust CPU/memory limits in `deployment.yaml`

## Monitoring

Key metrics to monitor:
- `patterns_detected` - Number of patterns found
- `messages_processed` - Tick messages processed
- `errors` - Error count
- `stocks_monitored` - Active stock count
- WebSocket connection status
- Redis connection status

## Security

- Service runs as non-root user
- Secrets are stored in Kubernetes Secrets
- Network policies can be applied to restrict traffic
- RBAC can be configured for service account permissions