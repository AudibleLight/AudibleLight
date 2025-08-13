Installation
------------

Prerequisites
^^^^^^^^^^^^^

- ``git``
- ``python3.10`` or above (tested up to 3.12)
- ``poetry``
- A modern Linux distro: current versions of ``Ubuntu`` and ``Red Hat`` have been tested and confirmed to work.

  - Using another OS? Let us know so we can add it here!

Install via the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt update
   sudo apt install libsox-dev libsox-fmt-all freeglut3-dev
   git clone https://github.com/AudibleLight/AudibleLight.git
   poetry install

Install via PyPI
^^^^^^^^^^^^^^^^

*Coming soon!*

Running ``pre-commit`` hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry run pre-commit install
   pre-commit run --all-files