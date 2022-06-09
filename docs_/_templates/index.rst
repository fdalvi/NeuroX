.. NeuroX toolkit documentation master file

Welcome!
========

NeuroX is a Python library that encapsulates various methods for neuron interpretation and analysis, geared towards Deep NLP models. The library is a one-stop shop for activation extraction, probe training, clustering analysis, neuron selection and more. We currently support `transformers <https://github.com/huggingface/transformers>`_ models, with support for more toolkits coming soon.

Features
--------
- Support for extraction of activation from popular models including the entirety of `transformers <https://github.com/huggingface/transformers>`_, with extended support for other models like `OpenNMT-py <https://github.com/OpenNMT/OpenNMT-py>`_ planned in the near future
- Support for training linear probes on top of these activations, on the entire activation space of a model, on specific layers, or even on specific set of neurons.
- Support for neuron extraction related to specific concepts, using the *Linear Correlation Analysis* method (`What is one Grain of Sand in the Desert? Analyzing Individual Neurons in Deep NLP Models. <https://ojs.aaai.org/index.php/AAAI/article/view/4592/4470>`_). The toolkit can extract either a local ranking of neurons important to a particular target class, or a global ranking of neurons important to all the target classes.
- Support for ablation analysis by either removing or zeroing out specific neurons to determine their function and importance.
- Support for subword and character level aggregation across a variety of tokenizers, including BPE and all tokenizers in the `transformers <https://github.com/huggingface/transformers>`_ library.
- Support for activation visualization over regular text, to generate qualitative samples of neuron activity over particular sentences.

Getting Started
---------------
See the :ref:`installation-instructions` page for various ways of installing the toolkit. Browsing the methods in the `API Reference <neurox.html>`_ is the best way to explore the toolkit. A `Jupyter notebook <https://github.com/fdalvi/NeuroX/blob/master/examples/End%20to%20End%20Example.ipynb>`_ is also provided with a complete example all the way from extraction to visualizing top neurons.

Citation
--------
Please cite our AAAI'19 paper if you use this toolkit in your work.

.. code-block:: bibtex

   @article{dalvi2019neurox,
      title={NeuroX: A Toolkit for Analyzing Individual Neurons in Neural Networks},
      author={Dalvi, Fahim
         and Nortonsmith, Avery
         and Bau, D Anthony
         and Belinkov, Yonatan
         and Sajjad, Hassan
         and Durrani, Nadir
         and Glass, James},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
      year={2019}
   }

Indices and tables
==================

.. toctree::
   :maxdepth: 4
   
   installation
   Tutorial Notebook <https://github.com/fdalvi/NeuroX/blob/master/examples/End%20to%20End%20Example.ipynb>


.. toctree::
   :maxdepth: 4
   :caption: API Reference:

   neurox

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
