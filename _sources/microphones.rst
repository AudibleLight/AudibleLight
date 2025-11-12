Microphones
===========

These classes define microphones that can be added to a ``Scene``. They are separated based on the output format they provide:

"Mic" Format Output
-------------------

.. autosummary::
   :toctree: _autosummary

   audiblelight.micarrays.Eigenmike32
   audiblelight.micarrays.Eigenmike64
   audiblelight.micarrays.AmbeoVR
   audiblelight.micarrays.MonoCapsule

Ambisonics Output
-----------------

.. autosummary::
   :toctree: _autosummary

   audiblelight.micarrays.FOAListener

Binaural Output
---------------

.. autosummary::
   :toctree: _autosummary

   audiblelight.micarrays.Binaural

Base Classes & Utilities
------------------------

New microphones can be defined by inheriting from the parent ``MicArray`` class.

.. autosummary::
   :toctree: _autosummary

   audiblelight.micarrays.MicArray
   audiblelight.micarrays.dynamically_define_micarray