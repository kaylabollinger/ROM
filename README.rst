ROM
===

Description
-----------

Reduced order models (ROMs) are used to approximate high-dimensional complex systems by simpler, often interpretable low-dimensional systems. This python package contains two ROMs developed in `Bollinger (2022) <https://kilthub.cmu.edu/articles/thesis/Reduced_Order_Models_and_Regression_for_Applications_to_Structural_Modeling/21111022>`_. Examples (using Jupyter notebooks) of how to apply this code are found in the "examples" folder in this repository, and the datasets used in these examples (and in the work cited above) are found in the "data" folder.

How it works:
^^^^^^^^^^^^^

See the "How it works" section in the documentation on `ReadTheDocs <https://rom.readthedocs.io/en/latest/>`_ .

Requirements and Dependencies
-----------------------------
* `scikit-learn <https://scikit-learn.org/>`_ >=0.23
* `numpy <https://numpy.org/>`_
* `torch <https://pytorch.org/>`_
* `pymanopt <https://pymanopt.org/>`_

Installation
------------

To install the rom package, open the terminal/command line and clone the repository with the command

.. code-block:: bash

    git clone https://github.com/kaylabollinger/ROM.git  

Navigate into the ``ROM`` folder (where the setup.py file is located) and run the command

.. code-block:: bash

    python setup.py install
  
You should now be able to import the rom package in Python scripts with the command ``import rom``.

Documentation
-------------

Documentation can be found on `ReadTheDocs <https://rom.readthedocs.io/en/latest/>`_.
