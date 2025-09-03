AudibleLight 🔈💡
================

Spatial soundscape synthesis using ray-tracing
----------------------------------------------

.. warning::

   *This project is currently under heavy development*. We have done our due diligence to ensure that it works as expected. However, if you encounter any errors, please `open an issue <https://github.com/AudibleLight/AudibleLight/issues>`_ and let us know.


What is ``AudibleLight``?
-------------------------

This project provides a platform for generating synthetic soundscapes by simulating arbitrary microphone configurations and dynamic sources in both parameterized and 3D-scanned rooms. Under the hood, ``AudibleLight`` uses Meta’s `open-source acoustic ray-tracing engine <https://github.com/beasteers/rlr-audio-propagation>`_ to simulate spatial room impulse responses and convolve them with recorded events to emulate array recordings of moving sources. The resulting soundscapes can prove useful in training models for a variety of downstream tasks, including acoustic imaging, sound event localisation and detection, direction of arrival estimation, etc.

In contrast to other projects (e.g., `sonicsim <https://github.com/JusperLee/SonicSim/tree/main/SonicSim-SonicSet>`_, `spatialscaper <https://github.com/marl/SpatialScaper>`_), ``AudibleLight`` provides a straightforward API without restricting the user to any specific dataset. You can bring your own mesh and your own audio files, and ``AudibleLight`` will handle all the spatial logic, validation, and synthesis necessary to ensure that the resulting soundscapes are valid for use in training machine learning models and algorithms.

``AudibleLight`` is developed by researchers at the `Centre for Digital Music, Queen Mary University of London <https://www.c4dm.eecs.qmul.ac.uk/>`_ in collaboration with `Meta Reality Labs <https://www.meta.com/en-gb/emerging-tech>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   installation
   _examples/example_generation
   _examples/augmentations
   _examples/plot_mic_arrays

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
