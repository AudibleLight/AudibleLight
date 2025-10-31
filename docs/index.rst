AudibleLight 🔈💡
=================

A Controllable, End-to-End API for Soundscape Synthesis Across Ray-Traced & Real-World Measured Acoustics
---------------------------------------------------------------------------------------------------------

.. warning::

   *This project is currently under heavy development*. We have done our due diligence to ensure that it works as expected. However, if you encounter any errors, please `open an issue <https://github.com/AudibleLight/AudibleLight/issues>`_ and let us know.


What is ``AudibleLight``?
-------------------------

``AudibleLight`` is a unified API for soundscape synthesis supporting ray-traced, real-world, and parametric RIR generation. It enables flexible microphone array modeling and dynamic, fully annotated source trajectories within a single workflow. It is built upon `SpatialScaper <https://github.com/marl/SpatialScaper>`_, `SoundSpaces <https://github.com/facebookresearch/sound-spaces>`_ and `Pyroomacoustics <https://github.com/LCAV/pyroomacoustics>`_ for scalable soundscape generation with unprecedented acoustic diversity.

``AudibleLight`` is developed by researchers at the `Centre for Digital Music, Queen Mary University of London <https://www.c4dm.eecs.qmul.ac.uk/>`_ in collaboration with `Meta Reality Labs <https://www.meta.com/en-gb/emerging-tech>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   installation
   _examples/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   _examples/1.0.0_make_scene
   _examples/1.1.0_add_listeners
   _examples/1.2.0_add_events
   _examples/1.2.1_add_augmentations
   _examples/1.3.0_add_ambience
   _examples/2.0.0_synthesis

.. toctree::
   :maxdepth: 1
   :caption: API:

   core
   events
   microphones
   synthesis


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
