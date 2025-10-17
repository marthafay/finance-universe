# finance-universe

Research prototypes exploring signal geometry, energy logic, and agent behavior in financial time series.

This repository contains three lightweight experimental prototypes for signal-driven agents in finance. All models use deterministic logic and energy-inspired features — without machine learning.

CAUTION
Deterministic modeling is vulnerable to unnatural market distortions and algorithmically triggered stop-loss orders. Independent risk management and strategies are essential.

DISCLAIMER (Research Only)
This repository contains a research prototype. It is provided for educational and research purposes only. It does NOT constitute financial, investment, legal, medical, or any other form of professional advice. No warranty is expressed or implied. Use entirely at your own risk. Before applying any outputs to real-world decisions, seek advice from qualified professionals and conduct independent verification.

Note on Terminology
-------------------
The terms, metaphors, and models used in this repository (e.g., "Pseudoscalar Score", "Schrödinger Zone", "Virial Candle") are conceptual research constructs intended for educational and experimental purposes. They are not commercial brand claims or endorsements. Any commercial use must include clear attribution to the original source and must not imply sponsorship, endorsement, or warranty by the author.

Licensing Notice
----------------Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


Elias, M. (2025). Applied Mathematics: Signal Geometric Framework for Finance and Agent-Based (Deterministic) Modeling. Zenodo. https://doi.org/10.5281/zenodo.17216401

V 2.0

---

## Contents

- `finance_universe_demo.py` — Virial Candles, OddSpin, Schrödinger Zone, CAPD scaling  
- `maxwell_agent_flux.py` — Momentum + curvature → signal flux, position logic  
- `gravity_agent.py` — G-index model for M&A or project dynamics (K/U ratio)

Each file is standalone. No installation needed beyond numpy, pandas, matplotlib.


---

## ☕ Support

I’d be happy if you like my work: https://buymeacoffee.com/marthafay: https://buymeacoffee.com/marthafay

---

## Example Output

1. finance_universe_demo.py example plot:

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/f0ec81a4-325e-4b78-b042-f1950fbe6342" />

2. python codez/maxwell_agent_flux.py --demo
== Maxwell-Flux Agent ==
   CAGR: -0.1573
 Sharpe: -1.1429
  MaxDD: -0.5803
    Hit:  0.1960
 FinalEq: 0.5069  Bars: 1000

3. python codez/gravity_agent.py
== Gravity M&A Agent Demo ==
       K      U    eta  multiplier    status
0  1.268  1.324  1.914       0.702       hot
1  1.926  0.541  7.115       0.700       hot
2  0.716  1.630  0.879       1.105  balanced
3  1.923  1.307  2.942       0.700       hot
4  0.968  0.995  1.946       0.702       hot
5  1.135  1.683  1.349       0.766       hot
6  1.742  0.955  3.648       0.700       hot
7  1.114  1.180  1.887       0.703       hot

MC Band (deal 0): 0.700 .. 0.767

