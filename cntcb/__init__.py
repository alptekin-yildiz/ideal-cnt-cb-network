"""ideal-cnt-cb-network: stochastic ideal-network reference generator for
CNT/CB polymer composites.

Supported public workflows
--------------------------
Use the command-line tools under ``generate/`` for user-supplied geometries
and the frozen scripts under ``reproduce_paper/`` for manuscript regeneration.
Those two interfaces define the stable user-facing contract.

Reusable engine entry points
----------------------------
The ``cntcb.engines`` modules expose the same moment-matched, distribution-aware
simulation kernels used by the command-line tools:

    ``percolation_engine.run_hybrid_cnt_threshold_realization``
    ``percolation_engine.run_cb_threshold_realization``
    ``percolation_engine.run_cnt_threshold_ensemble``
    ``percolation_engine.run_cb_threshold_ensemble``
    ``conductivity_engine.run_step_conductivity_realization``
    ``simmons_conductivity_engine.run_simmons_conductivity_realization``
    ``simmons_conductivity_engine.run_simmons_conductivity_ensemble``
    ``gf_engine.run_simmons_gf_realization``
    ``gf_engine.run_simmons_gf_ensemble``

Material definitions live in ``cntcb.kernel.materials`` (``carbon_black``,
``carbon_nanotube``). The rest of ``cntcb.kernel`` is low-level implementation
infrastructure for spatial hashing, boundary flags, and monodisperse
percolation checks. For moment-matched polydisperse threshold calculations,
use ``generate/percolation_threshold.py`` or ``cntcb.engines.percolation_engine``.

Names with a leading underscore are internal implementation details and may
change between releases.

This module deliberately re-exports nothing: import from the submodules
listed above.
"""

__version__ = "1.0.0"
