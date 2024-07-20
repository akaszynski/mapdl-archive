######################
 MAPDL Archive Reader
######################

|pypi| |GH-CI| |MIT|

.. |pypi| image:: https://img.shields.io/pypi/v/mapdl-archive.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/mapdl-archive/

.. |GH-CI| image:: https://github.com/akaszynski/mapdl-archive/actions/workflows/testing-and-deployment.yml/badge.svg
   :target: https://github.com/akaszynski/mapdl-archive/actions/workflows/testing-and-deployment.yml

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

Read blocked Ansys MAPDL archive files written from MAPDL using
``CDWRITE``.

This is effectively `pymapdl-reader
<https://github.com/ansys/pymapdl-reader>`_ without the binary reader.
It's been isolated to allow greater flexibility in development.

**************
 Installation
**************

Installation through pip:

.. code::

   pip install mapdl-archive

**********
 Examples
**********

Load and Plot an MAPDL Archive File
===================================

ANSYS archive files containing solid elements (both legacy and modern),
can be loaded using Archive and then converted to a VTK object.

.. code:: python

   from mapdl_archive import Archive, examples

   # Sample *.cdb
   filename = examples.hexarchivefile

   # Read ansys archive file
   archive = Archive(filename)

   # Print raw data from cdb
   for key in archive.raw:
      print("%s : %s" % (key, archive.raw[key]))

   # Create an unstructured grid from the raw data and plot it
   grid = archive.parse_vtk(force_linear=True)
   grid.plot(color='w', show_edges=True)

   # write this as a vtk xml file
   grid.save('hex.vtu')

   # or as a vtk binary
   grid.save('hex.vtk')

.. figure:: https://github.com/akaszynski/mapdl-archive/blob/main/doc/hexbeam_small.png
   :alt: Hexahedral beam

You can then load this vtk file using `PyVista
<https://docs.pyvista.org/version/stable/>`_ or `VTK
<https://vtk.org/>`_.

.. code:: python

   import pyvista as pv
   grid = pv.UnstructuredGrid('hex.vtu')
   grid.plot()

************************
 Reading ANSYS Archives
************************

MAPDL archive ``*.cdb`` and ``*.dat`` files containing elements (both
legacy and modern) can be loaded using Archive and then converted to a
``vtk`` object:

.. code:: python

   import mapdl_archive
   from mapdl_archive import examples

   # Read a sample archive file
   archive = mapdl_archive.Archive(examples.hexarchivefile)

   # Print various raw data from cdb
   print(archive.nnum, archive.nodes)

   # access a vtk unstructured grid from the raw data and plot it
   grid = archive.grid
   archive.plot(color='w', show_edges=True)

You can also optionally read in any stored parameters within the archive
file by enabling the ``read_parameters`` parameter.

.. code:: python

   import mapdl_archive
   archive = mapdl_archive.Archive('mesh.cdb', read_parameters=True)

   # parameters are stored as a dictionary
   archive.parameters

************************
 Writing MAPDL Archives
************************

Unstructured grids generated using VTK can be converted to ANSYS APDL
archive files and loaded into any version of ANSYS using
``mapdl_archive.save_as_archive`` in Python followed by ``CDREAD`` in
MAPDL. The following example using the built-in archive file
demonstrates this capability.

.. code:: python

   import pyvista as pv
   from pyvista import examples
   import mapdl_archive

   # load in a vtk unstructured grid
   grid = pv.UnstructuredGrid(examples.hexbeamfile)
   script_filename = '/tmp/grid.cdb'
   mapdl_archive.save_as_archive(script_filename, grid)

   # Optionally read in archive in PyMAPDL and generate cell shape
   # quality report
   from ansys.mapdl.core import launch_mapdl
   mapdl = launch_mapdl()
   mapdl.cdread('db', script_filename)
   mapdl.prep7()
   mapdl.shpp('SUMM')

Resulting ANSYS quality report:

.. code::

   ------------------------------------------------------------------------------
              <<<<<<          SHAPE TESTING SUMMARY           >>>>>>
              <<<<<<        FOR ALL SELECTED ELEMENTS         >>>>>>
   ------------------------------------------------------------------------------
                      --------------------------------------
                      |  Element count        40 SOLID185  |
                      --------------------------------------

    Test                Number tested  Warning count  Error count    Warn+Err %
    ----                -------------  -------------  -----------    ----------
    Aspect Ratio                 40              0             0         0.00 %
    Parallel Deviation           40              0             0         0.00 %
    Maximum Angle                40              0             0         0.00 %
    Jacobian Ratio               40              0             0         0.00 %
    Warping Factor               40              0             0         0.00 %

    Any                          40              0             0         0.00 %
   ------------------------------------------------------------------------------

Supported Elements
==================

At the moment, only solid elements are supported by the
``save_as_archive`` function, to include:

-  ``vtk.VTK_TETRA``
-  ``vtk.VTK_QUADRATIC_TETRA``
-  ``vtk.VTK_PYRAMID``
-  ``vtk.VTK_QUADRATIC_PYRAMID``
-  ``vtk.VTK_WEDGE``
-  ``vtk.VTK_QUADRATIC_WEDGE``
-  ``vtk.VTK_HEXAHEDRON``
-  ``vtk.VTK_QUADRATIC_HEXAHEDRON``

Linear element types will be written as SOLID185, quadratic elements
will be written as SOLID186, except for quadratic tetrahedrals, which
will be written as SOLID187.

*****************************
 License and Acknowledgments
*****************************

The ``mapdl-archive`` library is licensed under the MIT license.
