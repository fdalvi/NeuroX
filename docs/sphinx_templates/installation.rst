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

Add the directory to your python path. This can be done dynamically at runtime using the `sys.path` list:

.. code-block:: python

	import sys
	sys.path.append("path/to/cloned/NeuroX/")

A `Conda <https://anaconda.org>`_ environment is provided with all the necessary dependencies for the toolkit. The toolkit primarily relies on PyTorch and NumPy for most of its operations. To create a new environment with all the dependencies, run:

.. code-block:: bash

	conda env create -f conda-environment.yml -n neurox-toolkit
	conda activate neurox-toolkit


If you wish to manage your enviroment in other ways, a standard ``requirements.txt`` is also provided for use by ``pip`` directly.