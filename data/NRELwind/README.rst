NREL Offshore Wind Turbine
==========================

Offshore wind turbines' access to faster and steadier wind offers a meaningful advantage over land turbines. This new method of power generation is currently being studied within different research communities, industry, and government agencies. One area of research is assessing the safety and reliability of offshore wind turbine designs (and knowledge of land turbines cannot be directly applied since offshore turbines must withstand significantly harsher environments). In particular, the more that is known about the fatigue loads on these turbines, the better the estimation of their durability, reliability, and lifespan.

The dataset considered in this thesis was developed the US Department of Energy with the National Renewable Energy Laboratory (NREL). This data was created using physics simulations such as TurbSim (which simulates turbulent wind) and FAST (which simulates turbine behavior). There are 5 inputs: wind speed, wave direction, wave height, wave period, and wind platform direction. We consider 7 outputs described below (along with their FAST nomenclature):

- RootMxc1: edge-wise blade bending moment
- RootMyc1: flap-wise blade bending moment
- TwrBsMxt: side-side tower base bending moment
- TwrBsMyt: fore-aft tower base Bending moment
- Anch1Ten, Anch2Ten, Anch3Ten: anchor tensions for the three mooring lines

Each dataset (one for each of the 7 outputs) contains 50,000 samples. The original source for this data can be found `here <https://github.com/paulcon/as-data-sets/tree/master/NREL_Wind>`_.
