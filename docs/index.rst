.. currentmodule:: ldgm
.. ldgm documentation master file, created by Wilder Wohns

Welcome to tspyro's documentation
==================================

`tspyro <https://github.com/awohns/tspyro>`_ is a software package that uses `Pyro <http://pyro.ai>`_ to infer the time and location of ancestral haplotypes in a `tree sequences <https://tskit.dev/tutorials/what_is.html>`_.

The method operates on tree sequences in the `tskit format <https://tskit.dev>`_. These can either be simulated tree sequences (for example from `msprime <https://tskit.dev/msprime/docs/stable/intro.html>`_, `SLiM <https://messerlab.org/slim/>`_ or `Slendr <https://www.slendr.net>`_)  or inferred (for example by `tsinfer <https://tskit.dev/tsinfer>`_). 

Please refer to the :ref:`tutorial <sec_tutorial>` and :ref:`python-api <sec_python_api>` for details on using ``tspyro``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   python-api



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
