AudibleLight documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Core API
========

.. autosummary::
   :toctree: _autosummary
   :caption: Core API:

   audiblelight.core.Scene
   audiblelight.worldstate.WorldState

Events and Emitters
===================

.. autosummary::
   :toctree: _autosummary
   :caption: Events and Emitters:

   audiblelight.event.Event
   audiblelight.ambience.Ambience
   audiblelight.worldstate.Emitter

Microphones
===========

.. autosummary::
   :toctree: _autosummary
   :caption: Microphones:

   audiblelight.micarrays.MicArray
   audiblelight.micarrays.Eigenmike32
   audiblelight.micarrays.AmbeoVR
   audiblelight.micarrays.MonoCapsule


Synthesis
=========

.. autosummary::
   :toctree: _autosummary
   :caption: Synthesis:

   audiblelight.synthesize.render_event_audio
   audiblelight.synthesize.render_audio_for_all_scene_events
   audiblelight.synthesize.generate_scene_audio_from_events
   audiblelight.synthesize.time_variant_convolution
   audiblelight.synthesize.time_invariant_convolution


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
