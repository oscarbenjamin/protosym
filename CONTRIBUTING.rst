Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `BSD-3-clause license`_ and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _BSD-3-clause license: https://opensource.org/license/bsd-3-clause/
.. _Source Code: https://github.com/oscarbenjamin/protosym
.. _Documentation: https://protosym.readthedocs.io/
.. _Issue Tracker: https://github.com/oscarbenjamin/protosym/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

You need Python 3.8+.

To install the project in editable mode:

.. code:: console

   $ pip install -e .

You can now run an interactive Python session,
or the command-line interface:

.. code:: console

   $ python
   >>> import protosym

How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ pip install nox
   $ nox

List the available Nox sessions:

.. code:: console

   $ nox --list-sessions

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

.. code:: console

   $ nox --session=tests

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/

For development work it is more convenient to install the development
dependencies and run the ``quicktest.sh`` script:

.. code:: console

   $ pip install -r requirements-all.txt
   $ pip install -e .
   $ ./quicktest.sh

See the ``quicktest.sh`` script for how to run individual commands.

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

It is recommended to open an issue before starting work on anything.

.. _pull request: https://github.com/oscarbenjamin/protosym/pulls
.. github-only
.. _Code of Conduct: CODE_OF_CONDUCT.rst
