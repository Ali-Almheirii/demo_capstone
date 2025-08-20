# Taxi-Out Prediction Backend

## Overview
Backend API for taxi-out time prediction at airports. Frontend collects core inputs, backend computes engineered features and returns predictions.

## API Endpoints

### 1. POST `/predict`
Main prediction endpoint.

**Request Body:**
```json
{
  "scenario_mode": "historical_preset_week" | "manual_core_only",
  "airport": "DXB",
  "timestamp": "2024-01-15T14:30:00+04:00",
  "runway": "05L",
  "terminal": "T1", 
  "stand": "S1",
  "aircraft_type": "A320"
}
```

**Response:**
```json
{
  "prediction_min": 12.5,
  "percentile_vs_week": "75th",
  "engineered_features": {
    "hour_of_day": 14,
    "day_of_week": 1,
    "is_peak_hour": true,
    "runway_queue_length": 3
  },
  "bin_info": {
    "state": "High",
    "departures_count": 8
  }
}
```

### 2. GET `/traffic/day`
Returns 10-minute binned traffic data for a single day.

**Query Params:**
- `date`: Single date (YYYY-MM-DD)

**Example:**
```
GET /traffic/day?date=2024-01-15
```

**Response:**
```json
{
  "timezone": "Asia/Dubai",
  "bins": [
    {
      "bin_start": "2024-01-15T00:00:00+04:00",
      "bin_end": "2024-01-15T00:10:00+04:00", 
      "state": "Low",
      "departures_count": 2,
      "departures_count_prev_bin": 1,
      "core_features_example": {
        "runway": "05L",
        "terminal": "T1",
        "stand": "S1", 
        "aircraft_type": "A320"
      }
    }
    // ... 144 bins total for a full day (24 hours × 6 bins per hour)
  ]
}
```

### 3. GET `/distribution`
Returns taxi-out time distribution statistics and histogram data.

**Response:**
```json
{
  "distribution_stats": {
    "mean": 15.2,
    "std": 8.1,
    "sample_size": 50000,
    "percentiles": {
      "25th": 10.5,
      "50th": 14.2,
      "75th": 18.9
    }
  },
  "histogram_data": [
    {
      "bin_start": 0,
      "bin_end": 5,
      "count": 1250
    },
    {
      "bin_start": 5,
      "bin_end": 10,
      "count": 3200
    }
    // ... additional histogram bins
  ]
}
```

## Core Inputs
- backend should decide whats core input based on the preprocessing pipeline what input that pipeline expects
- based on that UI will change the req payload

## Engineered Features (Backend Computed)
- backend should inform about the engineered features that the model is actually gonna use as input 
- so UI can expects and use the response from the backend properly

## Development Notes
- Frontend only sends core inputs
- Backend must compute all engineered features
- Ensure feature pipeline matches training data
- Handle timezone conversion properly
- Return predictions in minutes
- Include confidence metrics if available
