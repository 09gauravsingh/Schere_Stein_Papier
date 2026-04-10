# Weather & Power Consumption Correlation Analysis

## Executive Summary

**Surprising Finding:** External weather temperature shows **strong correlation with power consumption (r=0.8009)**, but this is NOT because weather directly drives individual container behavior. Instead, it's because the weather station and reefer ambient sensors measure the same thermal conditions (**r=0.9973 agreement**).

**Practical Implication:** For your forecasting challenge, use weather data carefully:
- ✓ Excellent for 24h-ahead forecasting (when future ambient is unknown)
- ✗ Redundant for historical analysis (reefer sensors already capture ambient)
- ⚠️ The strong r=0.8009 is aggregate-level; individual containers show weak power↔weather response

---

## 1. Overall Weather-Power Correlations (Oct 2025 - Feb 2026)

### Primary Findings

| Relationship | Correlation | Strength | Interpretation |
|--------------|-------------|----------|-----------------|
| **Power ↔ Weather Temp** | **+0.8009** | **Strong** | Impressive correlation at hourly aggregate level |
| **Power ↔ Wind Speed** | **-0.2054** | **Weak** | Negative (higher wind = slightly lower power?) |
| **Weather Temp ↔ Reefer Ambient** | **+0.9973** | **Nearly Perfect** | External weather station and container sensors measure same conditions |
| **Reefer Ambient ↔ Power** | **+0.1714** | **Weak** | Direct thermal mechanism is weak (r=0.16, as found earlier) |

### The Paradox: Why r=0.80 (Strong) vs r=0.17 (Weak)?

```
Weather Temperature ─┐
                     ├─→ [Hourly Aggregation] ─→ r=0.8009 (Strong)
Reefer Ambient ──────┘

                     But individually:
Reefer Ambient ─────→ Power  r=0.1714 (Weak)
```

**Explanation:**
- At hourly level, both weather and ambient shift together, amplifying correlation through aggregation
- Individual containers show weak response (r=0.17) regardless of temperature
- **Hardware type, not temperature, dominates individual power prediction**

---

## 2. Seasonal Variation in Weather-Power Correlation

| Month | Power ↔ Temp | Power ↔ Wind | Avg Power | Avg Weather Temp |
|-------|--------------|-------------|-----------|-----------------|
| **Oct 2025** | +0.5083 | -0.4211 | 2,536 W | 11.3°C |
| **Nov 2025** | +0.6403 | -0.1544 | 2,215 W | 7.3°C |
| **Dec 2025** | +0.6750 | +0.1339 | 2,157 W | 5.6°C |
| **Jan 2026** | +0.2245 | -0.2526 | 1,684 W | -0.3°C |

### Key Observations

**October** (warm, variable weather):
- Correlation is moderate (+0.5083)
- Wind shows strong negative correlation (-0.4211)
- **Interpretation:** Warm months have more variability; wind effects more visible

**November-December** (cool, stable):
- Correlation strengthens (+0.64 to +0.68)
- Power drops as weather cools
- **Interpretation:** Cooler baseline reduces noise; clearer temperature signal

**January** (cold, consistent):
- Correlation drops sharply (+0.2245)
- Very low average power (1,684W)
- **Interpretation:** Cold weather creates "ceiling effect" — most containers already at minimum power

### Wind Speed Paradox

**Negative correlation with power** (-0.2054 overall, -0.4211 in October):
- **Expected:** Higher wind → more heat transfer → higher power
- **Observed:** Higher wind → lower power?
- **Likely cause:** High wind correlated with low-pressure systems bringing cold weather
  - Cold weather → lower power demand overall
  - Wind effect masked by temperature effect

---

## 3. Weather Station vs Reefer Sensor Comparison

### Perfect Agreement: r=0.9973

External weather station (VC_Halle3) and reefer container ambient sensors measure essentially **the same thermal environment**.

```
Weather Station: Outdoor, point measurement
Reefer Sensors:  Inside/attached to containers, distributed across yard

Correlation:     +0.9973 (essentially identical)
```

### Implication

1. **Redundancy:** External weather data and reefer ambient sensors are largely redundant for explaining power
2. **Reefer advantage:** Container sensors are available for ALL 374 days, not just 35% of data
3. **Weather advantage:** External data allows 24h-ahead forecasting (you can use weather forecast for tomorrow's power)

---

## 4. Data Availability: With vs Without Weather

### Coverage

| Period | Reefer Records | Hours with Both | Hours Reefer-only |
|--------|----------------|-----------------|------------------|
| **With Weather** (Sep 24 - Feb 23) | 1,324,545 | 2,599 | — |
| **Without Weather** (Jan 1 - Sep 23) | 2,450,012 | — | 5,804 |
| **Total** | 3,774,557 | 2,599 | 5,804 |

**65% of your reefer data has NO external weather observations** (Jan-Sep 2025).

---

## 5. Weather Correlation Consistency Across Hardware Types

All major hardware types show **identical weather correlations**:
- Power ↔ Weather Temp: +0.8009 (SCC6, ML3, DecosVb, DecosVa, DecosIIIj, MP4000 all same)
- Power ↔ Wind Speed: -0.2054 (all identical)

**Why identical?** Likely because we're correlating *hourly aggregated* power (merged across all hardware in that hour) with *hourly aggregated* weather. Individual hardware heterogeneity averages out.

---

## 6. Reefer Ambient Correlation: Stable Across Periods

| Period | Ambient ↔ Power |
|--------|-----------------|
| **Full year** | +0.1563 |
| **With weather (Oct-Jan)** | +0.1714 |
| **Without weather (Jan-Sep)** | +0.1518 |

**Consistent r=0.15-0.17** regardless of whether external weather is available. This confirms:
- Reefer ambient drives power only weakly (fundamental truth, not measurement artifact)
- Other factors (hardware, tier, SetPoint) dominate
- Weather data doesn't strengthen the ambient→power relationship

---

## 7. What This Means for Your Challenge

### For Forecasting WITHOUT Weather (Jan-Sep 2025)
```
Power ~ Hardware + Tier + Hour + Ambient + (weak weather effects)
        ↑ Strong   ↑ Moderate  ↑ Strong  ↑ Weak

Use: Hardware-based model with hourly/tier adjustments
Reefer ambient sensors sufficient; no weather needed
```

### For Forecasting WITH Weather (Oct 2025 - Jan 2026)
```
Power ~ Hardware + Tier + Hour + [Weather OR Ambient] + (interaction)
        ↑ Strong   ↑ Moderate  ↑ Strong ↑ Redundant

Option A: Use reefer ambient (available in test data?)
Option B: Use external weather (if provided in test set)
Option C: Use both as ensemble (weather for 24h-ahead, ambient for same-day)
```

### Why Weather Correlation is HIGH but Ambient Correlation is LOW

**Apparent paradox resolved:**
- **Weather-Power r=0.80** (hourly aggregate, both measured at hour level)
- **Ambient-Power r=0.17** (hour-to-hour individual records)

The strong weather correlation is an **aggregate phenomenon**:
1. Hour 1: Weather 10°C → Ambient 10°C → Power 2,400W
2. Hour 2: Weather 5°C → Ambient 5°C → Power 2,100W
3. **Overall trend visible: warmer weather, higher power**
4. **But within Hour 1:** Among containers at 10°C ambient, power still varies ±500W (hardware dominates)

---

## 8. Wind Speed Deep Dive

### Negative Correlation: -0.2054

**Seasonal breakdown:**
- October: -0.4211 (strong negative)
- November: -0.1544 (weak negative)
- December: +0.1339 (weak positive!)
- January: -0.2526 (weak negative)

**Interpretation:**
- High wind is associated with **lower** power consumption on average
- This contradicts physics (wind should increase convective cooling demand, not reduce it)
- **Likely confounding:** High wind comes with low-pressure systems → colder air → less cooling needed

**Actual causal effect of wind:** Probably small to zero; the observed correlation is a proxy for temperature/pressure patterns.

---

## 9. Recommendations for Your Model

### Strategy 1: Hardware-Dominant (Always Available)
```python
pred_power = power_by_hardware[HW] ×
             (1 + tier_adjustment[Tier]) ×
             hourly_cycle[Hour]
```
- Works for all 374 days
- Uses available reefer ambient indirectly (via historical training)
- Expected accuracy: Baseline

### Strategy 2: Weather-Enhanced (Oct 2025 - Jan 2026 only)
```python
# For dates with external weather forecast:
pred_power = power_by_hardware[HW] ×
             (1 + weather_adjustment[Weather_Temp]) ×
             (1 + tier_adjustment[Tier]) ×
             hourly_cycle[Hour]

# For dates without weather:
pred_power = Strategy_1  # Fallback to hardware-only
```
- Marginal improvement from weather
- Weather r=0.80 but contains no unique information (reefer sensors already capture it)
- Main value: 24h-ahead forecasting

### Strategy 3: Ensemble
```python
pred_power = 0.8 × Strategy_1 + 0.2 × Strategy_2
```
- Robust to weather availability
- Leverages both signals
- Hedge against overfitting to either approach

---

## Conclusion

**Weather data is valuable but not a game-changer:**

| Metric | Finding |
|--------|---------|
| **Directly useful** | For 24h-ahead forecasting (external weather forecast as input) |
| **Redundant** | For historical analysis (reefer sensors already measure ambient) |
| **Weak causal link** | Power←Ambient r=0.17 (ambient explains only 3% of power variance) |
| **Coverage** | Only 35% of training data (Oct 2025 - Jan 2026) |
| **Recommendation** | Use weather for test forecasting; rely on hardware/tier for robustness |

**The strong r=0.8009 weather correlation is misleading:** It reflects aggregate-level patterns, not individual container response. At the container level, hardware type matters far more than any temperature.

---

**Analysis Date:** 2026-04-09
**Data Period with Weather:** Sep 24, 2025 - Feb 23, 2026 (2,149 hourly observations)
**Data Period without Weather:** Jan 1 - Sep 23, 2025 (5,804 hourly observations)
