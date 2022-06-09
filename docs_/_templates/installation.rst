.. _installation-instructions:

Installation Instructions
=========================

This toolkit requires and is tested on Python versions 3.6 and above. It may work with older Python versions with some fiddling, but is currently not tested nor supported. The easiest way to get started is to use the published pip package:

.. code-block:: bash

  pip install neurox

This will install the toolkit and all its dependencies. You can then use the toolkit by importing the relevant modules like:

.. code-block:: python
	:linenos:

	import neurox.data.extraction.transformers_extractor as transformers_extractor
	transformers_extractor.extract_representations(...)

	from neurox.analysis.visualization import TransformersVisualizer
	visualizer = TransformersVisualizer('bert-base-uncased')
	
Manual Installation
-------------------
If you wish to install this package manually (e.g. to modify or contribute to the code base), you can clone `the repository <https://github.com/fdalvi/NeuroX>`_ into a directory of your choice:

.. code-block:: bash

	git clone https://github.com/fdalvi/NeuroX.git

Create and activate a new virtual environment for the toolkit (This step can be skipped if you manage your environment in other ways like Conda or System-level installations):

.. code-block:: bash

	python -m venv .neurox-env
	source .neurox-env/bin/activate

Install the dependencies required to run the toolkit:

.. code-block:: bash

	pip install -e .
