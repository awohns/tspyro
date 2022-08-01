.. currentmodule:: tspyro
.. _sec_tutorial:


========
Tutorial
========

.. _sec_tutorial_tspyro:

*********************************************
Inferring Time and Location in Tree Sequences
*********************************************

The goal of ``tspyro`` is to infer the age and location of ancestral haplotypes. To illustrate a typical use case of the software, let's first run a simple forwards-time simulation in `SLiM <https://messerlab.org/slim/>`_. We This script will produce a tree sequence.

This simple simulation is adapted from Recipe 15.1 (A simple 2D continuous-space model) in the ``SLiM`` manual and outputs a tree sequence. It can be run with the `SLiM command line tool or the SLiMgui <https://messerlab.org/slim/>`_.

::

    initialize() {
        setSeed(10);
        initializeSLiMOptions(dimensionality="xy");
        initializeTreeSeq(); 
        initializeMutationRate(1e-7);
        initializeMutationType("m1", 0.5, "f", 0.0);
        initializeGenomicElementType("g1", m1, 1.0);
        initializeGenomicElement(g1, 0, 99999);
        initializeRecombinationRate(1e-8);
    }
    1 late() {
        sim.addSubpop("p1", 500);
        
        // initial positions are random in ([0,1], [0,1])
        p1.individuals.x = runif(p1.individualCount);
        p1.individuals.y = runif(p1.individualCount);
    }
    modifyChild() {
        // draw a child position near the first parent, within bounds
        do child.x = parent1.x + rnorm(1, 0, 0.02);
        while ((child.x < 0.0) | (child.x > 1.0));
        
        do child.y = parent1.y + rnorm(1, 0, 0.02);
        while ((child.y < 0.0) | (child.y > 1.0));
        
        return T;
    }
    2000 late() { 
    sim.treeSeqOutput("~/slim_2d_continuous.trees");
    sim.outputFixedMutations(); }


You may want to change the path where the tree sequence will be saved.

To run ``tspyro``, we first need to load up the tree sequence with `tskit <https://tskit.dev/tskit/docs/stable/introduction.html>`_:

.. code-block:: python

    import tskit

    ts = tskit.load("~/slim_2d_continuous.trees")
    print(ts.num_samples,
          ts.num_trees,
          ts.num_nodes,
          ts.num_mutations)

The output of this code is:

.. code-block:: python

    1000 11 1844 136


``SLiM`` simulates entire populations, so let's randomly sample 100 nodes from the simulated tree sequence to perfrom inference on.

.. code-block:: python

   import numpy as np 
   rng = np.random.default_rng(20)
   random_sample = np.random.choice(np.arange(0, ts.num_samples), 100, replace=False)
   sampled_ts = ts.simplify(samples=random_sample)
   print(sampled_ts.num_samples,
         sampled_ts.num_trees,
         sampled_ts.num_nodes,
         sampled_ts.num_mutations)

The output of this code is:

.. code-block:: python

    100 5 200 79


Now we want to run ``tspyro`` on this simulated tree sequence. Let's first see an example of only inferring the times of the nodes in the tree sequence. ``tspyro.infer_geotime`` is the main function most users will need to run ``tspyro``.

.. code-block:: python

   inferred_times, inferred_locations, migration_scale, guide, losses = tspyro.infer_geotime(sampled_ts, leaf_location=None, Ne=1000, mutation_rate=1e-7)


Let's evaluate the output of the inference... TODO.


Next, let's infer the locations of ancestral haplotypes jointly with the times.





