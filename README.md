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

### Citation

If you use NeuXtalViz in your work, please cite:

Z. Morgan, Z. Xiao, S. Pawledzio, I. Ye, K. Loughlin, S. Jin, V. Lynch, T. Proffen, C. Hoffmann, and X. Wang, "*NeuXtalViz*: interactive three-dimensional visualization and analysis for single-crystal neutron diffraction," *J. Appl. Cryst.* **59**(6) (2026). [https://doi.org/10.1107/S1600576726009106](https://doi.org/10.1107/S1600576726009106)

```bibtex
@article{morgan2026neuxtalviz,
  author  = {Morgan, Zachary and Xiao, Zhongcan and Pawledzio, Sylwia and Ye, Iris and Loughlin, Kathleen and Jin, Shiyun and Lynch, Vickie and Proffen, Thomas and Hoffmann, Christina and Wang, Xiaoping},
  title   = {{NeuXtalViz}: interactive three-dimensional visualization and analysis for single-crystal neutron diffraction},
  journal = {Journal of Applied Crystallography},
  volume  = {59},
  number  = {6},
  year    = {2026},
  doi     = {10.1107/S1600576726009106}
}
```

The preprint is on [arXiv](https://arxiv.org/abs/2606.25414).

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

