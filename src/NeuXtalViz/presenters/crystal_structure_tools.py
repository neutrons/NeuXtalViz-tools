import functools

from NeuXtalViz.presenters.periodic_table import PeriodicTable
from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter


class CrystalStructure(NeuXtalVizPresenter):
    """
    Presenter for the crystal structure tool.

    Connects the crystal structure view's signals to model-driven actions
    for editing lattice parameters, atom sites, and space group settings,
    calculating structure factors and hkl equivalents, and loading/saving
    crystal structure files.

    Parameters
    ----------
    view : NeuXtalViz.views.crystal_structure_tools.CrystalStructureView
        View for the crystal structure tool.
    model : NeuXtalViz.models.crystal_structure_tools.CrystalStructure
        Model for the crystal structure tool.
    """

    def __init__(self, view, model):
        """
        Initialize the presenter and wire up view signals.

        Parameters
        ----------
        view : NeuXtalViz.views.crystal_structure_tools.CrystalStructureView
            View for the crystal structure tool.
        model : NeuXtalViz.models.crystal_structure_tools.CrystalStructure
            Model for the crystal structure tool.
        """
        super(CrystalStructure, self).__init__(view, model)

        self.view.connect_group_generator(self.generate_groups)
        self.view.connect_setting_generator(self.generate_settings)
        self.view.connect_F2_calculator(self.calculate_F2)
        self.view.connect_hkl_calculator(self.calculate_hkl)
        self.view.connect_row_highligter(self.highlight_row)
        self.view.connect_lattice_parameters(self.update_parameters)
        self.view.connect_atom_table(self.set_atom_table)
        self.view.connect_load_CIF(self.load_CIF)
        self.view.connect_save_INS(self.save_INS)
        self.view.connect_select_isotope(self.select_isotope)

        self.view.connect_calculate_absorption(self.calculate_absorption)

        self.view.connect_instrument_selector(self.select_instrument)
        self.view.connect_load_UB(self.load_UB)
        self.view.connect_clear_UB(self.clear_UB)
        self.view.connect_calculate_simulator(self.calculate_simulator)

        self.simulator_UB = None

        self.generate_groups()
        self.generate_settings()
        self.select_instrument()

    def highlight_row(self):
        """
        Highlight the selected atom row in the view.

        Parameters
        ----------
        None
        """
        scatterer = self.view.get_scatterer()
        self.view.set_atom(scatterer)

    def set_atom_table(self):
        """
        Set the atom table in the view and update atoms.

        Parameters
        ----------
        None
        """
        self.view.set_atom_table()
        self.update_atoms()

    def update_parameters(self):
        """
        Update lattice parameters and atom positions in the view.

        Parameters
        ----------
        None
        """
        params = self.view.get_lattice_constants()
        params = self.model.update_parameters(params)
        self.model.update_lattice_parameters(*params)
        self.view.set_lattice_constants(params)
        vol = self.model.get_unit_cell_volume()
        self.view.set_unit_cell_volume(vol)

        atom_dict = self.model.generate_atom_positions()
        self.view.add_atoms(atom_dict)

        self.view.draw_cell(self.model.get_unit_cell_transform())
        self.view.set_transform(self.model.get_transform())

    def generate_groups(self):
        """
        Generate space groups for the selected crystal system and update the view.

        Parameters
        ----------
        None
        """
        system = self.view.get_crystal_system()
        nos = self.model.generate_space_groups_from_crystal_system(system)
        self.view.update_space_groups(nos)

        self.generate_settings()

    def generate_settings(self):
        """
        Generate settings for the selected space group and update the view.

        Parameters
        ----------
        None
        """
        no = self.view.get_space_group()
        settings = self.model.generate_settings_from_space_group(no)
        self.view.update_settings(settings)

    def load_CIF(self):
        """
        Load a CIF file and update the crystal structure in the view and model.

        Parameters
        ----------
        None
        """
        filename = self.view.load_CIF_file_dialog()

        if filename:
            self.update_processing()

            self.update_processing("Loading CIF...", 10)

            self.model.load_CIF(filename)

            self.update_processing("Loading CIF...", 50)

            crystal_system = self.model.get_crystal_system()
            space_group = self.model.get_space_group()
            setting = self.model.get_setting()
            params = self.model.get_lattice_constants()
            scatterers = self.model.get_scatterers()

            self.view.set_crystal_system(crystal_system)
            self.generate_groups()
            self.view.set_space_group(space_group)
            self.generate_settings()
            self.view.set_setting(setting)
            self.view.set_lattice_constants(params)
            self.view.set_scatterers(scatterers)

            params = self.model.constrain_parameters()
            self.view.constrain_parameters(params)

            atom_dict = self.model.generate_atom_positions()
            self.view.add_atoms(atom_dict)

            self.update_processing("Loading CIF...", 80)

            self.view.draw_cell(self.model.get_unit_cell_transform())
            self.view.set_transform(self.model.get_transform())
            self.update_oriented_lattice()

            form, z = self.model.get_chemical_formula_z_parameter()
            self.view.set_formula_z(form, z)

            self.update_processing("Loading CIF...", 99)

            vol = self.model.get_unit_cell_volume()
            self.view.set_unit_cell_volume(vol)
            self.view.set_material_display(form, z, vol)

            self.refresh_magnetic_sites()

            self.update_complete("CIF loaded!")

        else:
            self.update_invalid()

    def update_atoms(self):
        """
        Update atom positions and related information in the view.

        Parameters
        ----------
        None
        """
        params = self.view.get_lattice_constants()
        setting = self.view.get_setting()
        scatterers = self.view.get_scatterers()

        self.model.set_crystal_structure(params, setting, scatterers)

        atom_dict = self.model.generate_atom_positions()
        self.view.add_atoms(atom_dict)

        form, z = self.model.get_chemical_formula_z_parameter()
        self.view.set_formula_z(form, z)

        self.view.draw_cell(self.model.get_unit_cell_transform())
        self.view.set_transform(self.model.get_transform())

        self.refresh_magnetic_sites()

    def refresh_magnetic_sites(self):
        """
        Rebuild the Simulator tab's magnetic sites table to match the
        current Structure tab atom site table.

        Parameters
        ----------
        None
        """
        labels = [scatterer[0] for scatterer in self.model.get_scatterers()]
        self.view.refresh_magnetic_sites(labels)

    def calculate_F2(self):
        """
        Start calculation of F2 factors using a worker thread.

        Parameters
        ----------
        None
        """
        d_min = self.view.get_minimum_d_spacing()
        params = self.view.get_lattice_constants()

        worker = self.view.worker(
            functools.partial(
                self.calculate_F2_process, d_min=d_min, params=params
            )
        )
        worker.connect_result(self.calculate_F2_complete)
        worker.connect_finished(self.update_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_F2_complete(self, result):
        """
        Complete F2 calculation and update the view with results.

        Parameters
        ----------
        result : tuple or None
            Result from F2 calculation.
        """
        if result is not None:
            self.view.set_factors(*result)

    def calculate_F2_process(
        self, progress, stop_event=None, d_min=None, params=None
    ):
        """
        Worker task that generates unique reflections and structure factors.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        d_min : float, optional
            Minimum d-spacing to generate reflections for. If None, it is
            derived from `params` (default None).
        params : tuple, optional
            Lattice constants used to derive `d_min` when it is not given,
            and to validate that the calculation should proceed
            (default None).

        Returns
        -------
        hkls : numpy.ndarray or None
            Array of unique HKL indices, or None if stopped or invalid.
        ds : numpy.ndarray or None
            Array of d-spacings corresponding to `hkls`.
        F2s : numpy.ndarray or None
            Array of squared structure factors corresponding to `hkls`.
        """
        if self.stop_processing(stop_event):
            return None

        if params is not None:
            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Calculating factors...", 10)

            if self.stop_processing(stop_event):
                return None

            if d_min is None:
                d_min = min(params[0:2]) * 0.2

            hkls, ds, F2s = self.model.generate_F2(d_min)

            progress("Factors calculated...", 99)

            progress("Factors calculated!", 100)

            return hkls, ds, F2s

        else:
            progress("Invalid parameters.", 0)

    def calculate_hkl(self):
        """
        Start calculation of hkl equivalents using a worker thread.

        Parameters
        ----------
        None
        """
        hkl = self.view.get_hkl()

        worker = self.view.worker(
            functools.partial(self.calculate_hkl_process, hkl=hkl)
        )
        worker.connect_result(self.calculate_hkl_complete)
        worker.connect_finished(self.update_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_hkl_complete(self, result):
        """
        Complete hkl calculation and update the view with results.

        Parameters
        ----------
        result : tuple or None
            Result from hkl calculation.
        """
        if result is not None:
            self.view.set_equivalents(*result)

    def calculate_hkl_process(self, progress, stop_event=None, hkl=None):
        """
        Worker task that calculates symmetry equivalents of a reflection.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        hkl : tuple, optional
            Miller indices (h, k, l) of the reflection to calculate
            equivalents for (default None).

        Returns
        -------
        hkls : list of V3D or None
            Symmetry-equivalent HKL reflections, or None if stopped or
            invalid.
        d : float or None
            d-spacing of the reflection.
        F2 : float or None
            Squared structure factor of the reflection.
        """
        if self.stop_processing(stop_event):
            return None

        if hkl is not None:
            progress("Processing...", 1)

            if self.stop_processing(stop_event):
                return None

            progress("Calculating equivalents...", 10)

            if self.stop_processing(stop_event):
                return None

            hkls, d, F2 = self.model.calculate_F2(*hkl)

            progress("Equivalents calculated...", 99)

            progress("Equivalents calculated!", 100)

            return hkls, d, F2

        else:
            progress("Invalid parameters.", 0)

    def calculate_absorption(self):
        """
        Start prediction of per-reflection transmission/absorption using
        a worker thread.

        Parameters
        ----------
        None
        """
        d_min = self.view.get_absorption_d_min()
        wavelength = self.view.get_wavelength()
        shape_params = self.view.get_absorption_shape_constants()
        sample_vectors = self.view.get_absorption_sample_vectors()
        shape_vectors = self.view.get_absorption_shape_vectors()

        worker = self.view.worker(
            functools.partial(
                self.calculate_absorption_process,
                d_min=d_min,
                wavelength=wavelength,
                shape_params=shape_params,
                sample_vectors=sample_vectors,
                shape_vectors=shape_vectors,
            )
        )
        worker.connect_result(self.calculate_absorption_complete)
        worker.connect_finished(self.update_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_absorption_complete(self, result):
        """
        Complete absorption prediction and update the view with results.

        Parameters
        ----------
        result : tuple or None
            Result from :meth:`calculate_absorption_process`.
        """
        if result is not None:
            hkls, ds, Ts, Tbars, abs_dict, mesh, T, r_incident = result
            self.view.set_absorption_results(hkls, ds, Ts, Tbars)
            self.view.set_absorption_parameters(abs_dict)
            mu = abs_dict["mu_a"] + abs_dict["mu_s"]
            # add_absorption_sample clears the scene first, so the corner
            # axes widget (driven by set_transform/show_axes) must be
            # (re-)added after it, or it would get wiped out again.
            self.view.add_absorption_sample(mesh, T, mu, r_incident)
            self.view.set_transform(T)

    def calculate_absorption_process(
        self,
        progress,
        stop_event=None,
        d_min=None,
        wavelength=None,
        shape_params=None,
        sample_vectors=None,
        shape_vectors=None,
    ):
        """
        Worker task that predicts per-reflection transmission/absorption.

        Intended to run on a background worker thread, reporting progress
        and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        d_min : float, optional
            Minimum d-spacing to generate reflections for. If None, it
            is derived from the lattice constants, the same way as the
            Factors tab's ``calculate_F2_process``.
        wavelength : float, optional
            Incident wavelength, in Angstrom.
        shape_params : list of float, optional
            Ellipsoid thickness, width, and height, in mm.
        sample_vectors : tuple of (list, list) or None, optional
            Sample U/V vectors defining the crystal's own orientation.
        shape_vectors : tuple of (list, list) or None, optional
            Shape U/V vectors orienting the shape mesh.

        Returns
        -------
        hkls : numpy.ndarray or None
            Miller indices, sorted by d-spacing then h, k, l, or None
            if stopped, invalid, or no reflection was reachable.
        ds : numpy.ndarray or None
            d-spacing per reflection.
        Ts : numpy.ndarray or None
            Transmission per reflection.
        Tbars : numpy.ndarray or None
            Absorption-weighted path length (cm) per reflection.
        abs_dict : dict or None
            Absorption/scattering parameters, from the model's
            ``get_absorption_dict``.
        mesh : numpy.ndarray or None
            Sample shape mesh, from the model's ``sample_mesh``.
        T : numpy.ndarray or None
            a*/b*/c* orientation matrix, from the model's
            ``get_transform_from_UB``, for drawing the axes arrows
            attached to the sample.
        r_incident : float or None
            Sample radius (cm) along the incident beam direction, from
            the model's ``get_incident_path_length``.
        """
        if self.stop_processing(stop_event):
            return None

        if wavelength is None or shape_params is None:
            progress("Invalid parameters.", 0)
            return None

        progress("Processing...", 1)

        if d_min is None:
            params = self.model.get_lattice_constants()
            d_min = min(params[0:2]) * 0.2

        if self.stop_processing(stop_event):
            return None

        progress("Building material...", 10)

        chem, Z = self.model.get_chemical_formula_z_parameter()
        vol = self.model.get_unit_cell_volume()
        mat_dict = self.model.get_material_dict(
            " ".join(chem.split("-")), float(Z), vol
        )

        if self.stop_processing(stop_event):
            return None

        progress("Building orientation...", 20)

        UB = None
        if sample_vectors is not None:
            UB = self.model.get_UB_from_vectors(*sample_vectors)
        if UB is None:
            UB = self.model.UB

        T = self.model.get_transform_from_UB(UB)

        shape_angles = (0, 0, 0)
        if shape_vectors is not None:
            values = self.model.get_euler_angles(*shape_vectors, UB)
            if values is not None:
                shape_angles = values

        if self.stop_processing(stop_event):
            return None

        progress("Generating reflections...", 30)

        hkls, ds = self.model.generate_hkl_list(d_min)

        if self.stop_processing(stop_event):
            return None

        progress("Calculating transmission (Monte Carlo)...", 50)

        result = self.model.predict_transmission(
            hkls,
            ds,
            wavelength,
            shape_params,
            mat_dict,
            *shape_angles,
            UB,
        )
        if result is None:
            progress("No reachable reflections at this wavelength.", 0)
            return None

        hkls_sorted, ds_sorted, Ts, Tbars, volume = result

        progress("Finalizing...", 90)

        abs_dict = self.model.get_absorption_dict(wavelength, volume)
        mesh = self.model.sample_mesh()

        u_vector = (
            sample_vectors[0] if sample_vectors is not None else (0, 0, 1)
        )
        r_incident = self.model.get_incident_path_length(mesh, UB, u_vector)

        progress("Transmission calculated!", 100)

        return hkls_sorted, ds_sorted, Ts, Tbars, abs_dict, mesh, T, r_incident

    def select_isotope(self):
        """
        Select an isotope and show the periodic table dialog.

        Parameters
        ----------
        None
        """
        atom = self.view.get_isotope()

        if atom != "":
            view = self.view.get_periodic_table()
            model = self.model.get_periodic_table(atom)

            self.periodic_table = PeriodicTable(view, model)
            self.periodic_table.view.connect_selected(self.update_selection)
            self.periodic_table.view.show()

    def update_selection(self, data):
        """
        Update the selected isotope and atom table in the view.

        Parameters
        ----------
        data : object
            Selected isotope data.
        """
        self.view.set_isotope(data)
        self.view.set_atom_table()
        self.update_atoms()

    def save_INS(self):
        """
        Save the current crystal structure to an INS file.

        Parameters
        ----------
        None
        """
        if self.model.has_crystal_structure():
            filename = self.view.save_INS_file_dialog()

            if filename:
                self.model.save_ins(filename)

    def select_instrument(self):
        """
        Update the Simulator d-min field for the newly selected instrument.

        Parameters
        ----------
        None
        """
        instrument = self.view.get_simulator_instrument()
        d_min = self.model.get_instrument_d_min(instrument)
        self.view.set_simulator_d_min(d_min)

    def load_UB(self):
        """
        Load a UB matrix from file for the Simulator tab.

        Parameters
        ----------
        None
        """
        filename = self.view.load_UB_file_dialog()

        if filename:
            self.simulator_UB = self.model.load_UB_from_file(filename)
            self.view.set_UB_status(filename)

    def clear_UB(self):
        """
        Clear the loaded Simulator UB matrix, reverting to u/v vectors.

        Parameters
        ----------
        None
        """
        self.simulator_UB = None
        self.view.set_UB_status(None)

    def calculate_simulator(self):
        """
        Start prediction of observable reflections and expected
        counts/I-sigma using a worker thread.

        Parameters
        ----------
        None
        """
        instrument = self.view.get_simulator_instrument()
        d_min = self.view.get_simulator_d_min()
        vectors = self.view.get_simulator_vectors()
        angles = self.view.get_simulator_goniometer()
        counting_time = self.view.get_simulator_counting_time()
        shape_params = self.view.get_absorption_shape_constants()
        shape_vectors = self.view.get_absorption_shape_vectors()
        magnetic_sites = self.view.get_magnetic_sites()
        moment_axis = self.view.get_moment_axis()

        worker = self.view.worker(
            functools.partial(
                self.calculate_simulator_process,
                instrument=instrument,
                d_min=d_min,
                vectors=vectors,
                angles=angles,
                counting_time=counting_time,
                shape_params=shape_params,
                shape_vectors=shape_vectors,
                UB_loaded=self.simulator_UB,
                magnetic_sites=magnetic_sites,
                moment_axis=moment_axis,
            )
        )
        worker.connect_result(self.calculate_simulator_complete)
        worker.connect_finished(self.update_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def calculate_simulator_complete(self, result):
        """
        Complete the simulation and update the view with results.

        Parameters
        ----------
        result : tuple or None
            Result from :meth:`calculate_simulator_process`.
        """
        if result is not None:
            (
                hkls,
                ds,
                lambdas,
                F2s,
                F2s_mag,
                Is,
                IsigmaIs,
                Is_mag,
                IsigmaIs_mag,
                R_sigma,
                R_sigma_mag,
                volume,
            ) = result
            self.view.set_simulator_results(
                hkls,
                ds,
                lambdas,
                F2s,
                F2s_mag,
                Is,
                IsigmaIs,
                Is_mag,
                IsigmaIs_mag,
            )
            self.view.set_R_sigma(R_sigma, R_sigma_mag)

    def calculate_simulator_process(
        self,
        progress,
        stop_event=None,
        instrument=None,
        d_min=None,
        vectors=None,
        angles=None,
        counting_time=None,
        shape_params=None,
        shape_vectors=None,
        UB_loaded=None,
        magnetic_sites=None,
        moment_axis=None,
    ):
        """
        Worker task that predicts observable reflections and their
        expected counts/I-sigma for a fixed instrument, orientation,
        goniometer setting, and counting time.

        Intended to run on a background worker thread, reporting
        progress and checking for a stop request between steps.

        Parameters
        ----------
        progress : callable
            Callback invoked as ``progress(status, value)`` to report
            status text and percent complete.
        stop_event : threading.Event, optional
            Event used to signal that the worker should stop early
            (default None).
        instrument : str, optional
            Instrument name.
        d_min : float, optional
            Minimum d-spacing to predict reflections for. If None, it
            is derived from the instrument default.
        vectors : tuple of (list, list) or None, optional
            Sample U/V vectors defining the crystal's orientation, used
            only if `UB_loaded` is None.
        angles : list of float, optional
            Goniometer omega, chi, and phi angles (degrees).
        counting_time : float, optional
            Requested counting time (minutes).
        shape_params : list of float, optional
            Ellipsoid thickness, width, and height (mm), from the
            Absorption tab.
        shape_vectors : tuple of (list, list) or None, optional
            Shape U/V vectors orienting the shape mesh, from the
            Absorption tab.
        UB_loaded : (3, 3) ndarray or None, optional
            UB matrix loaded from file, taking priority over `vectors`
            if not None.
        magnetic_sites : list of dict or None, optional
            Magnetic sites from the view's ``get_magnetic_sites``, each
            with keys ``"row"``, ``"ion"``, ``"g"``, ``"mu"`` -- `row`
            is resolved here against the model's current scatterers to
            attach the ``x``, ``y``, ``z``, ``occ`` the model's
            ``magnetic_structure_factor2`` needs.
        moment_axis : tuple of (str, list of float) or None, optional
            Easy-axis direction from the view's ``get_moment_axis``.

        Returns
        -------
        hkls, ds, lambdas, F2s, F2s_mag, Is, IsigmaIs, Is_mag,
        IsigmaIs_mag, R_sigma, R_sigma_mag, volume : tuple or None
            Result from the model's ``simulate_intensities``, or None
            if stopped, invalid, or no reflection was observable.
        """
        if self.stop_processing(stop_event):
            return None

        if angles is None or counting_time is None or shape_params is None:
            progress("Invalid parameters.", 0)
            return None

        progress("Processing...", 1)

        if d_min is None:
            d_min = self.model.get_instrument_d_min(instrument)

        if self.stop_processing(stop_event):
            return None

        progress("Loading instrument response...", 10)

        self.model.load_simulator_response(instrument)

        if self.stop_processing(stop_event):
            return None

        progress("Building material...", 20)

        chem, Z = self.model.get_chemical_formula_z_parameter()
        vol = self.model.get_unit_cell_volume()
        mat_dict = self.model.get_material_dict(
            " ".join(chem.split("-")), float(Z), vol
        )

        if self.stop_processing(stop_event):
            return None

        progress("Building orientation...", 30)

        UB = UB_loaded
        if UB is None and vectors is not None:
            UB = self.model.get_UB_from_vectors(*vectors)
        if UB is None:
            UB = self.model.UB

        shape_angles = (0, 0, 0)
        if shape_vectors is not None:
            values = self.model.get_euler_angles(*shape_vectors, UB)
            if values is not None:
                shape_angles = values

        if self.stop_processing(stop_event):
            return None

        progress("Predicting peaks (Monte Carlo absorption)...", 50)

        omega, chi, phi = angles

        resolved_sites = None
        if magnetic_sites:
            scatterers = self.model.get_scatterers()
            resolved_sites = []
            for site in magnetic_sites:
                _, x, y, z, occ, _ = scatterers[site["row"]]
                resolved_sites.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "occ": occ,
                        "ion": site["ion"],
                        "g": site["g"],
                        "mu": site["mu"],
                    }
                )

        result = self.model.simulate_intensities(
            instrument,
            d_min,
            UB,
            omega,
            chi,
            phi,
            shape_params,
            mat_dict,
            shape_angles,
            counting_time,
            magnetic_sites=resolved_sites,
            moment_axis=moment_axis,
        )

        if result is None:
            progress("No observable reflections at this setting.", 0)
            return None

        progress("Simulation complete!", 100)

        return result
