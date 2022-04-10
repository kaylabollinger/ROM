.. ROM documentation master file, created by
   sphinx-quickstart on Thu Apr  7 09:48:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ROM's documentation!
===============================

ROM is a reduced order model regression package with implementations of the ROMs introduced in Bollinger (2022).


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   rom
   license

Example:
========

In this example, we walk through how to train ROM's RFE model on fluid vorticity data.

.. code-block:: python
   
   import numpy as np
   import ROM
   import plotly.graph_objects as go
   
The vorticity data can be found `here <https://github.com/kaylabollinger/ROM>`_.
To load the data, just replace the :code:`data_dir` variable with the appropriate path to the dataset.

.. code-block:: python

   data_dir = 'data/vorticity/X.npy'
   data = np.load(data_dir)
   X = data[:-1,:]
   Y = data[1:,:]
   
We will use the first 75 snapshots for training:

.. code-block:: python

   num_train =  75
   X_train = X[:num_train,:]
   Y_train = Y[:num_train,:]
   
To train the RFE model, we first learn the :math:`k` dimensional linear subspace :math:`U\in\mathbb{R}^{d \times k}`via POD.

.. code-block:: python

   k = 6

   ss = ROM.subspaces
   U = ss.POD(X_train,k)

   UTX = np.matmul(X,U)
   UTX_train = UTX[:num_train,:]
   
Then, we train the RFE surrogate model:

.. code-block:: python

    model = ROM.response_surfaces.RFE()
    model.train([UTX_train, Y_train])
    
Using the trained model, we then regenerate all 150 snapshots:

.. code-block:: python

    X_curr = [X[0,:].reshape(1,-1)]
    for num_snapshot in range(data.shape[0]-1):
        UTX_curr = np.matmul(X_curr[-1],U)
        X_curr.append(model.predict(UTX_curr))
    
    X_calc = np.concatenate(X_curr,axis=0)
    
    print(f'relative error = {ROM.utils.rel_error(data,X_calc)}')
    
To visualize our generated snapshot at time :code:`time_show`, we display its contour plot:

.. code-block:: python

   time_show = 100

   levels = 0.5
   color = 'IceFire'
   
   num_x = 199
   num_y = 449
   
   # vorticity
   PLOT_RFE = np.copy(X_calc)
   PLOT_RFE[PLOT_RFE>5.] = 5.
   PLOT_RFE[PLOT_RFE<-5.] = -5.
   
   # cylinder
   scale=2
   theta = np.linspace(0,1,1000)*2*np.pi
   x_cyl = (np.sin(theta))/scale
   y_cyl = (np.cos(theta))/scale
   
   fig = go.Figure()

   fig.add_trace(
      go.Contour(
         z = PLOT_RFE[time_show,:].reshape(num_y,num_x).T,
         x = np.linspace(-1,8,num_y),
         y = np.linspace(-2,2,num_x),
         colorscale = color,
         contours=dict(
            start=-5,
            end=5,
            size=levels,
            )
         )
      )

   fig.add_trace(
      go.Scatter(x=x_cyl, y=y_cyl,
      fill='toself', 
      fillcolor='gray',
      line_color='black',
      opacity=1.0)
   )
   
   fig.show()
   
.. image:: images/vort_t100.svg

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
