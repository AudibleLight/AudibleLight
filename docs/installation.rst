Installation
------------

Prerequisites
^^^^^^^^^^^^^

- ``python3.10`` or above (tested up to ``python3.12``)
- A modern Linux distro: current versions of ``Ubuntu`` and ``Red Hat`` have been tested and confirmed to work.

  - Using another OS? Let us know so we can add it here!

If you're looking to develop ``AudibleLight``, you'll also need:

- ``git``
- ``poetry``
- ``make``

Install via pypi
^^^^^^^^^^^^^^^^

For non-development installs, the simplest way to install ``AudibleLight`` is via pypi:

.. code-block:: bash

   sudo apt install -y libsox-dev libsox-fmt-all freeglut3-dev pandoc
   pip install audiblelight


Install via the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to develop `AudibleLight`, you'll likely want to clone the repository and install it directly:

.. code-block:: bash

   git clone https://github.com/AudibleLight/AudibleLight.git
   cd AudibleLight
   make install

Download data
^^^^^^^^^^^^^

We provide several helper scripts to download and prepare data (3D meshes, sofa files, audio files) that may be useful in ``AudibleLight``.

You can run these scripts directly from the Python interpreter:

.. code-block:: python

   from audiblelight.download_data import download_fsd

   download_fsd(path="path/to/save/fsd", cleanup=True)

Alternatively, for a *development install*, you can run them from the command line:

.. code-block:: bash

   poetry run python scripts/download_data/download_fsd.py --path path/to/save/fsd --cleanup

From a development install, you can also run all download scripts at once using the ``Makefile``:

.. code-block:: bash

   make download

For further information, see :file:`scripts/download_data/README.md`.

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