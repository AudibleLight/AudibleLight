Augmentations
=============

Event Augmentations
-------------------

These augmentations can be added to an ``Event`` with ``Event.register_augmentations``, or when initialising this ``Event`` with ``Scene.add_event`` or ``Event.__init``

.. autosummary::
   :toctree: _autosummary

   audiblelight.augmentation.Bitcrush
   audiblelight.augmentation.Chorus
   audiblelight.augmentation.Clipping
   audiblelight.augmentation.Compressor
   audiblelight.augmentation.Deemphasis
   audiblelight.augmentation.Delay
   audiblelight.augmentation.Distortion
   audiblelight.augmentation.Fade
   audiblelight.augmentation.GSMFullRateCompressor
   audiblelight.augmentation.Gain
   audiblelight.augmentation.HighShelfFilter
   audiblelight.augmentation.HighpassFilter
   audiblelight.augmentation.Invert
   audiblelight.augmentation.Limiter
   audiblelight.augmentation.LowShelfFilter
   audiblelight.augmentation.LowpassFilter
   audiblelight.augmentation.MP3Compressor
   audiblelight.augmentation.MultibandEqualizer
   audiblelight.augmentation.Phaser
   audiblelight.augmentation.PitchShift
   audiblelight.augmentation.Preemphasis
   audiblelight.augmentation.Reverse
   audiblelight.augmentation.SpeedUp
   audiblelight.augmentation.TimeWarpDuplicate
   audiblelight.augmentation.TimeWarpRemove
   audiblelight.augmentation.TimeWarpReverse
   audiblelight.augmentation.TimeWarpSilence


Base Classes
------------

These parent classes can be inherited from to define new augmentations.

.. autosummary::
   :toctree: _autosummary

   audiblelight.augmentation.Augmentation
   audiblelight.augmentation.EventAugmentation
   audiblelight.augmentation.SceneAugmentation
