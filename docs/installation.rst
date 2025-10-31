Installation
------------

Prerequisites
^^^^^^^^^^^^^

- ``git``
- ``python3.10`` or above (tested up to ``python3.12``)
- ``poetry``
- ``make``
- A modern Linux distro: current versions of ``Ubuntu`` and ``Red Hat`` have been tested and confirmed to work.

  - Using another OS? Let us know so we can add it here!

Install via the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/AudibleLight/AudibleLight.git
   cd AudibleLight
   make install

Download datasets
^^^^^^^^^^^^^^^^^

We provide several helper scripts to download and prepare data (meshes, audio files) that may be useful in `AudibleLight`. To run these:

.. code-block:: bash

   make download

Generate a dataset
^^^^^^^^^^^^^^^^^^

To generate an example dataset, run:

.. code-block:: bash

   poetry run python scripts/experiments/generate_dataset.py

To see the available arguments that this script takes, add the ``--help`` argument

Running the tests
^^^^^^^^^^^^^^^^^

Before making a PR, ensure that you run the pre-commit hooks and tests:

.. code-block:: bash

   make fix
   make tests