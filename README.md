# Assessing the Economic Impacts of Space Weather Mitigation Investments in New Zealand

This repository contains the code and supporting materials used in the academic study:

**Oughton, E.J., Renton, A., Mac Manus, D., Bor, D., & Rodger, C.J. (2025)**  
*Assessing the economic benefits of space weather mitigation investment decisions: Evidence from Aotearoa New Zealand*  
[arXiv:2507.12495](https://arxiv.org/abs/2507.12495)

The project develops an integrated **physics–engineering–economic modelling framework** to estimate the direct and indirect macroeconomic impacts of extreme space weather events on New Zealand’s electricity transmission system and broader economy, with a particular focus on the value of mitigation investments.

---

## Overview

Space weather events such as extreme coronal mass ejections (CMEs) can induce geomagnetically induced currents (GICs) in high-voltage transmission networks, potentially leading to widespread power outages. As modern economies are highly dependent on electricity, these outages can trigger substantial cascading economic losses through supply chains.

This repository implements a spatially explicit modelling workflow that:

- Simulates electricity supply disruptions caused by GIC exposure under multiple plausible scenarios  
- Translates electricity outages into sectoral economic shocks using employment-weighted electricity demand  
- Estimates **direct GDP losses** using Value of Lost Load (VoLL) approaches  
- Estimates **indirect GDP losses** using a supply-driven (Ghosh) Input–Output economic model  
- Compares economic outcomes across alternative space weather mitigation investment strategies  

---

## Research Questions

1. What are the potential macroeconomic consequences of a major space weather event in New Zealand?
2. Which mitigation strategies are most effective when comparing avoided GDP losses with investment costs?

---

## Scenarios

The code evaluates seven disruption and mitigation scenarios, ranging from no mitigation to advanced combinations of operational strategies and hardware investments:

1. **National blackout (no mitigation)**  
2. **South Island blackout with North Island load shedding**  
3. **GIC threshold-based outages (no mitigation)**  
4. **Operational switching sequence mitigation**  
5. **Switching plus islanding of critical industrial loads**  
6. **Switching plus limited GIC blocker deployment**  
7. **Switching plus islanding and extensive GIC blocker deployment**

Scenarios differ in spatial extent, outage duration, restoration sequence, and investment cost, enabling explicit benefit–cost comparisons.

---

## Key Findings

- Unmitigated extreme scenarios can result in up to **NZ$8.36 billion** in lost GDP  
- More than **50% of total losses are indirect**, driven by supply-chain effects  
- Low-cost operational mitigation (e.g. switching and islanding) can deliver benefit–cost ratios of up to **740:1**  
- Hardware investments such as GIC blocking devices also provide strong returns (up to **80:1**)  
- Results are conservative and do not fully capture capital equipment losses at large industrial facilities  

---

## Acknowledgements

This research was supported by the U.S. National Science Foundation and the New Zealand Ministry of Business, Innovation and Employment. The authors thank Transpower New Zealand for data provision, technical input, and operational expertise.

---

## Disclaimer

This repository is intended for research and policy analysis purposes only. Results are scenario-based estimates subject to uncertainty in space weather severity, grid operations, economic behaviour, and recovery dynamics. They should not be interpreted as forecasts.
