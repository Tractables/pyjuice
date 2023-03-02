Getting Started
===============

.. _installation:

Installation
------------

We are in early development, so to use `pyjuice`, first clone the github repository, then install it using pip, then install nightly build of `pytorch`.

.. code-block:: console
   
   (.venv) $ cd pyjuice
   (.venv) $ pip3 install --editable .
   (.venv) $ pip3 install -I torch --index-url https://download.pytorch.org/whl/nightly/cu118


Usage
-----

For example:

>>> import pyjuice as juice

