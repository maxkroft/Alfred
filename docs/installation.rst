.. _installation:

Installation
============

A small part of ``Alfred`` is written in C, so you will need to have a ``gcc`` (C compiler) installed before installing ``Alfred``.

``Alfred`` has not been tested on Windows operating systems. If you are using a Windows device, we recommend installing WSL and running ``Alfred`` in Linux.

We also recommend installing and running ``Alfred`` in a ``conda`` environment. Install ``anaconda`` or ``miniconda`` `here <https://www.anaconda.com/docs/getting-started/miniconda/main>`_.
The safest route is to install ``Alfred`` in a fresh environment. You can create a new one by running

.. code-block:: bash

    $ conda create -n alfred python

You can replace "alfred" with whatever environment name you want, and if you run into issues later on in the install, trying setting python=3.13.

Next, activate your new environment and ``pip`` install the package:

.. code-block:: bash

    $ conda activate alfred
    $ python -m pip install alfred-exoplanets

If you are planning to use Jupyter notebooks (.ipynb files), you will likely need to install ``ipykernel`` and/or ``ipywidgets``. You can do this with:

.. code-block:: bash

    $ conda install -c conda-forge ipykernel ipywidgets

Finally, if you are using WSL, you may run into font formatting issues with ``Alfred``'s GUIs. This can be resolved by installing a different version of ``tk``:

.. code-block:: bash

    $ conda install -c conda-forge tk=*=xft_*


Optionally, you can also install ``ExoCTK`` if you want to generate limb darkening grids for more bands than are included by default.
This is not required to run ``Alfred``, and I recommend not installing it unless you specifically need more limb darkening grids. If you do choose to
install ``ExoCTK``, follow the installation instructions `here <https://github.com/ExoCTK/exoctk>`_.