# ![](https://github.com/zjmorgan/NeuXtalViz/blob/improve_viz/src/icons/neuxtalviz_logo.svg)

Neutron Scattering Single Crystal Visualization

*new-crystal-vis*

NeuXtalViz is an application for three-dimensional (3d) visualization of neutron scattering data.
It brings together two main libraries; PyVista is the main tool for displaying the 3d data while Mantid serves as the main library for working with reduced single crystal neutron diffraction data.

- [PyVista](https://pyvista.org/)
- [Mantid](https://github.com/mantidproject/mantid/)

The application also relies on several other packages.

- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scikit-image](https://scikit-image.org/)
- [PyVista](https://pyvista.org/)

Please see [arXiv](https://arxiv.org/abs/2606.25414) for additional information and citation.

### Documentation and tutorials

Detailed documentation with sample data is available [here](https://neutrons.github.io/NeuXtalViz-tools/source/NeuXtalViz.tutorials.html).

### Getting started

Create conda environment
`conda env create -f environment.yml`

Activate garnet environment
`conda activate nxv`

Install in editable mode for developlment
`python -m pip install -e .`

Run the GUI
`python src/NeuXtalViz.py`

Obtain latests changes
```git pull```

