# Reefer Power Consumption Model Training
## Critical Parameters Summary

**Based on analysis of 3.77M observations across 374 days**

---

## TIER 1: REQUIRED PARAMETERS (Non-Negotiable)
**These 3 parameters alone explain 65% of power variation**

### 1. Hardware Type (40% importance)
- **Impact**: 430% variation (1,000W - 7,000W range)
- **Examples**:
  - SCC6: 1,339W (baseline)
  - DecosVb: 1,335W (efficient)
  - ML3: 2,885W (moderate)
  - DecosIIIf: 7,060W (inefficient)
- **Action**: Encode all 19 hardware types. This is the PRIMARY predictor—everything else adjusts this baseline.

### 2. Container Wear (25% importance)
- **Impact**: 172% variation across container lifetime
- **Method**: Use TtlEnergyCons (cumulative energy) to create 5 quintiles
  - Q1 (least worn): 1,478W baseline
  - Q5 (most worn): 4,028W (+172%)
- **Action**: Apply multiplicative wear factor to hardware baseline
  - Q1 = 1.0× multiplier
  - Q2 = 1.4×
  - Q3 = 1.65×
  - Q4 = 2.1×
  - Q5 = 2.72×

### 3. Hardware × Container Size Pairing (15% importance)
- **Impact**: 447% variation (1,290W - 7,060W)
- **Best Pairing**: DecosVb + 20ft = 1,290W
- **Worst Pairing**: DecosIIIf + 40ft = 7,060W
- **Action**: Create interaction terms for hardware × container size combinations
- **Key Insight**: Don't use additive model. Combinations have non-linear effects.

---

## TIER 2: SECONDARY PARAMETERS (Mode-Specific)
**Use SEPARATE regression coefficients for COOLING vs HEATING modes**

| Parameter | Cooling Mode | Heating Mode | Action |
|-----------|--------------|--------------|--------|
| **Wind Speed** | -47.7 W/km/h | -27.8 W/km/h | Higher wind = lower power (convective cooling). Use mode-specific slopes. |
| **Temperature Delta** (SetPoint - Ambient) | +66.3 W/°C | -87.0 W/°C | **Direction reverses!** In cooling, more delta = more power. In heating, opposite. |
| **Combined R²** | 26% | 45% | Wind + Temp together explains 6-10× more than wind alone. |

### How to Implement
1. **Detect Mode**: Calculate `Gradient = SupplyTemp - ReturnTemp`
   - If Gradient < 0 → Cooling Mode
   - If Gradient > 0 → Heating Mode
2. **Apply Mode-Specific Coefficients**:
   - Cooling: `Power = Baseline - 47.7×Wind + 66.3×TempDelta`
   - Heating: `Power = Baseline - 27.8×Wind - 87.0×TempDelta`

---

## TIER 3: FINE-TUNING PARAMETERS (Optional)
**Add 3-5% accuracy improvement**

### Hourly Seasonality (±7% variation)
- Peak hours: 13:00-15:00 UTC at 2,537W
- Low hours: 04:00 UTC at 2,232W
- **Action**: Create hourly multiplier factors (24 factors total)

### Day-of-Week Anomaly (±1.8% variation)
- Tuesday: +3.6% (2,414W)
- Sunday: -1.8% (2,329W)
- **Action**: Optional. Add Tuesday multiplier only if needed.

### Cargo Type Proxy (±3-5% variation)
- Infer from SetPoint + Power combination
- Frozen cargo (<-10°C): 3,100W
- Cool cargo (0-5°C): 2,200W
- Warm cargo (>10°C): 1,100W
- **Action**: Create categorical feature from SetPoint binning

---

## DO NOT USE (Waste of Effort)

| Parameter | Why Skip |
|-----------|----------|
| Stack Tier (T1/T2/T3) | Only 5.6% variation. Nice to have, not essential. |
| Temperature Gradient | 67% records show inverted temps. Wait for sensor fixes. |
| Weather conditions beyond wind | Weather temp = reefer ambient (r=0.9973). Redundant. |
| Individual container history | Covered by wear quintiles. No additional value. |

---

## Complete Model Formula

```
POWER = [Hardware Baseline]
        × [Wear Multiplier]
        + [Hardware×Size Adjustment]
        + [Mode-Specific Wind Effect]
        + [Mode-Specific Temperature Effect]
        + [Hourly Seasonality]
        + [Cargo Type Adjustment]
```

### Example Calculation
**Cooling Mode: SCC6 + 40ft, Q3 Wear, 5 km/h wind, 20°C temp delta, 15:00 UTC**

```
= 1,768W (SCC6 baseline)
  × 1.65 (Q3 wear multiplier)
  - 32.4 W/km/h × 5 km/h (wind effect: -162W)
  + 66.3 W/°C × 20°C (temp effect: +1,326W)
  + 150W (hourly peak multiplier)

= 2,919W - 162W + 1,326W + 150W
= 4,233W
```

---

## Key Implementation Decisions

1. **Separate Cooling & Heating Models**
   - Don't merge them. Temperature delta reverses direction between modes.
   - Heating model: 45% R² vs Cooling model: 26% R²

2. **Hardware Type Always First**
   - This is the foundation. Everything else is adjustments to the hardware baseline.
   - Non-negotiable feature in all models.

3. **Use Wear Quintiles (Not Continuous)**
   - 5 buckets make more sense than continuous TtlEnergyCons.
   - Cleaner, more interpretable multipliers.

4. **Collect Wind Data if Missing**
   - 20% power savings at 10 km/h wind.
   - Critical for coastal/windy regions.
   - Achievable with existing weather stations.

5. **Hardware × Size Interactions Required**
   - One-size model won't capture pairing effects.
   - Use separate coefficients for efficient (DecosVa/DecosVb) vs inefficient (DecosIII) series.

6. **Test Target Accuracy**
   - **With Tier-1 factors only**: R² ≈ 0.65
   - **With Tier-1 + Tier-2**: R² ≈ 0.72
   - **Hourly predictions (wind+temp)**: R² ≈ 0.26-0.45 (harder problem)

---

## Quick-Start Feature Checklist

**Minimum Features (Tier 1 Only)**
```
✓ HardwareType (categorical, 19 categories)
✓ WearQuintile (categorical, 5 buckets)
✓ ContainerSize (categorical, 3 sizes)
✓ Mode (binary: Cooling/Heating)
```

**Recommended Features (Tier 1 + 2)**
```
✓ All above, plus:
✓ WindSpeed (continuous, km/h)
✓ TemperatureDelta (continuous, °C)
```

**Full Features (All Tiers)**
```
✓ All above, plus:
✓ HourOfDay (categorical, 24 hours)
✓ DayOfWeek (categorical, optional—only Tuesday matters)
✓ CargoTypeProxy (inferred from SetPoint)
```

**Evaluation Split**
```
✓ Don't use time-based split (temporal correlation)
✓ Split by Container ID (prevents leakage)
✓ Report R² separately for Cooling and Heating modes
✓ Report RMSE in watts for interpretability
```

---

## Data Preparation Checklist

```
□ Calculate WearQuintile from TtlEnergyCons
□ Calculate Mode from SupplyTemp - ReturnTemp
□ Extract HourOfDay from DateTime
□ Ensure no null values in Tier-1 features
□ Separate into Cooling and Heating datasets
□ For Tier-2: Create separate training sets with/without wind
□ For evaluation: stratify by Hardware type to ensure all types in test set
```

---

## Expected Performance Targets

| Model | Features | Cooling R² | Heating R² | Typical RMSE |
|-------|----------|-----------|-----------|--------------|
| Baseline (Hardware only) | 1 | 0.38 | 0.32 | 850W |
| With Wear | 2 | 0.55 | 0.48 | 680W |
| Full Tier-1 | 4 | 0.65 | 0.62 | 580W |
| With Wind + Temp | 6 | 0.72 | 0.70 | 520W |
| With all seasonality | 8 | 0.75 | 0.72 | 480W |

---

## Common Mistakes to Avoid

❌ **Don't**: Merge cooling and heating modes—use separate models
✓ **Do**: Build two models with mode-specific coefficients

❌ **Don't**: Use temperature delta as primary feature—hardware matters 20× more
✓ **Do**: Use hardware type as foundation, temperature as adjustment

❌ **Don't**: Forget wear degradation—containers age dramatically
✓ **Do**: Implement wear quintiles; Q5 containers use 172% more power

❌ **Don't**: Ignore hardware×size interactions
✓ **Do**: Create interaction terms for non-linear pairing effects

❌ **Don't**: Skip wind if available—20% power savings is significant
✓ **Do**: Collect and use wind data with mode-specific slopes

---

## Summary: The Three Numbers You Need to Remember

| What | Number | Meaning |
|------|--------|---------|
| **Hardware impact** | 430% | Difference between best (SCC6:1,339W) and worst (DecosIIIf:7,060W) |
| **Wear impact** | 172% | Power increase from newest (Q1:1,478W) to oldest (Q5:4,028W) container |
| **Wind effect** | -20% | Power reduction with 10 km/h wind in cooling mode (477W savings) |

These three factors determine 80% of your prediction accuracy. Everything else is optimization.

---

*Analysis completed: April 9, 2026*
*Ready to use with scikit-learn, XGBoost, LightGBM, or any tabular ML framework*
