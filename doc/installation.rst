============
Installation
============

Requirements
------------

Nalyst requires Python 3.10 or later.

Core Dependencies
~~~~~~~~~~~~~~~~~

- numpy >= 1.21.0
- scipy >= 1.7.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For full functionality:

- pandas >= 1.3.0 (data handling)
- matplotlib >= 3.4.0 (visualization)
- seaborn >= 0.11.0 (statistical plots)

Installation
------------

Using pip
~~~~~~~~~

.. code-block:: bash

    pip install nalyst

From source
~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/nalyst/nalyst.git
    cd nalyst
    pip install -e .

Verify Installation
-------------------

.. code-block:: python

    import nalyst
    print(nalyst.__version__)
    nalyst.show_info()
