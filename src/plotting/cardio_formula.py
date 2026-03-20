from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame="circle"):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=14):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            raise ValueError(f"unknown value for 'frame': {frame}")

        def draw(self, renderer):
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gridline in gridlines:
                    gridline.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            if frame == "polygon":
                spine = Spine(axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {"polar": spine}
            raise ValueError(f"unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    return theta


def get_patient_values(delta_formula, x_in):
    x_values = x_in.to_numpy()
    delta = np.zeros((x_values.shape[0], x_values.shape[1] + 1))

    if isinstance(delta_formula, sp.Float):
        delta[:, -1] = float(delta_formula)
        return pd.DataFrame(delta, columns=x_in.columns.tolist() + ["const"])

    for row_index in range(x_values.shape[0]):
        for formula_term in delta_formula.args:
            current_term = deepcopy(formula_term)
            if isinstance(current_term, sp.Float):
                delta[row_index, -1] = float(current_term)
                continue
            if len(current_term.free_symbols) != 1:
                raise ValueError("Expected KAAM symbolic terms to depend on a single variable.")
            variable = list(current_term.free_symbols)[0]
            column_index = x_in.columns.get_loc(str(variable))
            delta[row_index, column_index] += float(current_term.subs(variable, x_values[row_index, column_index]))

    return pd.DataFrame(delta, columns=x_in.columns.tolist() + ["const"])


def get_delta(x_frame, formula):
    delta_formula = formula
    for index, column in enumerate(x_frame.columns):
        delta_formula = delta_formula.subs(sp.symbols(f"x_{index + 1}"), sp.symbols(column))

    actual_vars = sorted({str(symbol) for symbol in delta_formula.free_symbols}, key=x_frame.columns.get_loc)
    if actual_vars:
        x_frame = x_frame[actual_vars]
    else:
        x_frame = x_frame.iloc[:, :0]

    delta_frame = get_patient_values(delta_formula, x_frame)
    return delta_formula, [delta_frame]


def select_symbolic_formula_payload(formula_payloads, rng_seed=2026, model_name="kaam_mixed_symbolic"):
    eligible = [
        row
        for row in formula_payloads
        if row.get("model_name") == model_name and row.get("has_formulas") and row.get("formulas")
    ]
    if not eligible:
        raise ValueError(f"No symbolic formula payloads found for model '{model_name}'.")
    rng = np.random.default_rng(int(rng_seed))
    return eligible[int(rng.integers(0, len(eligible)))]


def choose_formula_targets(graph, formulas):
    formula_nodes = [node for node, formula in formulas.items() if formula is not None]
    if not formula_nodes:
        raise ValueError("No formula-bearing nodes are available.")

    radar_candidates = [node for node in formula_nodes if len(formulas[node].free_symbols) >= 2]
    pdp_candidates = [node for node in formula_nodes if len(formulas[node].free_symbols) >= 1]

    if not radar_candidates:
        radar_candidates = formula_nodes
    if not pdp_candidates:
        pdp_candidates = formula_nodes

    radar_node = "ischemia" if "ischemia" in radar_candidates else radar_candidates[0]
    pdp_node = "systolic" if "systolic" in pdp_candidates else pdp_candidates[0]
    return radar_node, pdp_node
