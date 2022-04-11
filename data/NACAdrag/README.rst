NACA 0012 Airfoil
=================

The NACA airfoils are airfoil shapes developed by the National Advisory Committee for Aeronautics (NACA). Different airfoil shapes are labeled via the name "NACA" followed by a series of digits. In this work, we consider data based on the NACA 0012 airfoil. This is a symmetric airfoil (denoted by the "00") that is 12% as thick as it is long (denoted by the "12"). The simplicity of the structure makes it a standard test problem for flow in 2D space.

The quantity of interest in this data set is the drag coefficient of the NACA 0012 airfoil with respect to 18 Hicks-Henne shape parameters (characterizing the shape of the airfoil). This shape parameterization technique uses a weighted sum of bump functions to alter the original shape of the airfoil. Some benefits of using this technique is that it requires relatively few design variables, the resulting surface remains smooth, and localized deformations can be easily made. Given these parameters, the drag coefficient of the airfoil was computed using Stanford University Unstructured (SU2) computational fluid dynamics code. 

This dataset consists of 20,000 total samples and contains gradient information for each sample. The original source for this data can be found at `here <https://github.com/jeffrey-hokanson/varproridge>`_.