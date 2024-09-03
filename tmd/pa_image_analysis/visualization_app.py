import warnings

from dash import Dash, html, Input, Output, State, dcc, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_colorscales as dcs
import plotly_express as px
import plotly.graph_objects as go
import argparse
import base64
import numpy as np
import os
import pandas as pd
from skimage import draw
from scipy import ndimage
from typing import *

from simpa.io_handling import load_hdf5, save_data_field
from simpa.utils import get_data_field_from_simpa_output, SegmentationClasses, Tags

EXTERNAL_STYLESHEETS = [
    '../../docs/source/_static/dcc.css',
    dbc.themes.JOURNAL,
]
app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS, title="Data Visualization")

DEFAULT_COLORSCALE = ['rgb(5,48,97)', 'rgb(33,102,172)', 'rgb(67,147,195)', 'rgb(146,197,222)', 'rgb(209,229,240)',
                      'rgb(247,247,247)', 'rgb(253,219,199)', 'rgb(244,165,130)', 'rgb(214,96,77)',
                      'rgb(178,24,43)', 'rgb(103,0,31)']
GITHUB_LINK = 'https://github.com/CAMI-DKFZ/simpa'
SIMPA_DOCU_LINK = 'https://simpa.readthedocs.io/en/develop/'
CAMI_LINK = 'https://www.dkfz.de/en/cami/research/index.html'
BG_COLOR = "#506784"
FONT_COLOR = "#F3F6FA"
APP_TITLE = "Data Visualization"
PREVENT_3D_UPDATES = ['volume_axis', 'colorscale_picker', 'volume_slider']
PROPERTIES_TO_IGNORE_ON_CLICK = ['time_series_data', 'reconstructed_data']


class DataContainer:
    shape: Union[None, List, Tuple] = None
    axis: List = ['x', 'y', 'z']
    simpa_output: Union[None, Dict] = None
    simpa_data_fields: Union[None, Dict] = None
    wavelengths: Union[np.ndarray, None] = None
    segmentation_labels: Union[Dict, None] = None
    plot_types: List = ['imshow', 'hist-3D', 'hist-2D', 'box', 'violin', 'contour']
    click_points: Dict = {'x': [], 'y': [], 'z': [], 'param': []}
    shapes: List = []
    relayout_data: Dict = {'relayout': {}, 'param': ''}
    plot_layouts: Dict = {'plot_11': {}, 'plot_21': {}}


DATA = DataContainer()

# plotly figure controls configuration
GRAPH_CONFIG = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ],
}

app.layout = html.Div([
    dbc.Tooltip(
        "Select the shape tool on left Plotly Figure and draw a shape",
        target='annotate_switch',
        placement='top'
    ),
    dbc.Tooltip(
        "Fix the ranges in both axis for all Plotly Figures",
        target='fix_ranges',
        placement='top'
    ),
    html.Div(id='dummy', style={'display': None}),
    html.Div(id='dummy2', style={'display': None}),
    html.Div(id='dummy3', style={'display': None}),
    html.Div(id='dummy4', style={'display': None}),
    html.Div([
        dbc.Toast(
            [html.P("Found N infinite values in array", id='toast_content')],
            id="alert_toast",
            header="Not finite values",
            is_open=False,
            dismissable=True,
            icon="danger",
            style={"position": "fixed", "top": 66, "right": 10, "width": 250},
            duration=4000,
        ),
    ], style=dict(zIndex=1, position="relative")),
    dbc.Row([
        dbc.Col([
            html.H1("Data Visualization Tool"),
        ], width=9),
    ], style=dict(zIndex=0)),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Tabs(children=[
                dcc.Tab(label="Handlers", id="handler-tab", children=[
                    html.Hr(),
                    html.H6("Plotting / Handling settings"),
                    dbc.Input(
                        id="data_path",
                        type="text",
                        pattern=None,
                        placeholder="Path to simulation folder or file",
                        persistence=True,
                        persistence_type="session",
                        invalid=True,
                        autofocus=True
                    ),
                    dcc.Dropdown(
                        multi=False,
                        id="file_selection",
                        placeholder="Simulation files",
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Hr(),
                    html.H6("Data selection"),
                    html.P("Axis"),
                    dbc.RadioItems(
                        id="volume_axis",
                        persistence_type="session",
                        options=[{'label': 'x', 'value': 0},
                                 {'label': 'y', 'value': 1},
                                 {'label': 'z', 'value': 2}],
                        value=1,
                        inline=True
                    ),
                    html.Hr(),
                    html.H6("Visual settings"),
                    html.P("Color scale"),
                    dcs.DashColorscales(
                        id="colorscale_picker",
                        nSwatches=7,
                        fixSwatches=True,
                        colorscale=DEFAULT_COLORSCALE
                    ),
                    html.Hr(),
                    html.H6("General Information"),
                    html.Div([

                    ], id="general_info")
                ])
            ]),

        ], width=2),
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label="Visualization", id="tab-1", children=[
                    dbc.Col([
                        dbc.Row([
                            html.Div([
                                dbc.Switch(
                                    id="annotate_switch",
                                    value=True,
                                    label="Annotate",
                                    style={'display': 'inline-block', 'padding-right': '10px'}
                                ),
                                dbc.Switch(
                                    id="fix_ranges",
                                    value=True,
                                    label="Fix ranges",
                                    style={'display': 'inline-block', 'padding-right': '10px'}
                                ),
                            ]),

                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    id="plot_div_1",
                                    children=[
                                        html.Div([
                                            dcc.RangeSlider(
                                                id="plot_scaler1",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0., 1.],
                                                marks={i: {'label': f'{i:1.1f}'} for i in np.arange(0, 1, 0.1)},
                                                allowCross=False,
                                                pushable=0.05,
                                                tooltip=dict(always_visible=True, placement="left"),
                                                persistence_type="session",
                                                disabled=False,
                                                vertical=True,
                                            ),
                                        ], style={"width": "5%", "display": 'inline-block'}
                                        ),

                                        dcc.Graph(id="plot_11",
                                                  hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                                  style={"width": "90%", "display": 'inline-block', "padding": 25},
                                                  config=GRAPH_CONFIG),
                                    ]
                                )
                            ], style={"width": "100%"}),
                            html.Div([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="plot_type1",
                                        multi=False,
                                        placeholder="Plot type",
                                        persistence_type="session",
                                        disabled=False,
                                        style=dict(width='150px', display='inline-block')
                                    ),
                                    dcc.Dropdown(
                                        id="param1",
                                        multi=False,
                                        placeholder="Parameter to plot",
                                        persistence_type="session",
                                        disabled=True,
                                        style={'width': '150px', 'display': 'inline-block',
                                               'margin-left': '5px'}
                                    ),
                                    html.H6("Wavelength selector"),
                                    dcc.Slider(
                                        min=0,
                                        max=100,
                                        value=50,
                                        step=1,
                                        tooltip=dict(always_visible=True, placement="top"),
                                        updatemode="mouseup",
                                        persistence_type="session",
                                        id="channel_slider",
                                        marks={
                                            0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                            50: {'label': '50'},
                                            100: {'label': '100', 'style': {'color': '#f50'}}
                                        }
                                    ),
                                    html.H6("Volume slice selector"),
                                    dcc.Slider(
                                        min=0,
                                        max=10,
                                        value=5,
                                        step=1,
                                        tooltip=dict(always_visible=True, placement="top"),
                                        updatemode="mouseup",
                                        persistence_type="session",
                                        id="volume_slider",
                                        marks={
                                            0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                            5: {'label': '5'},
                                            10: {'label': '10', 'style': {'color': '#f50'}}
                                        }
                                    ),
                                    html.H6("Annotations"),
                                    dbc.Button(
                                        "Save Annotation",
                                        id="save_anno",
                                        style={'display': 'inline-block', 'verticalAlign': 'top',
                                               'margin-left': '5px'},
                                        outline=True,
                                        color="primary",
                                    ),
                                ])
                            ], style={"width": "35%"}),
                        ])
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Spectra and annotation visualization"),
                            html.Div(
                                id="plot_div_2",
                                children=[
                                    dcc.Graph(id="plot_21",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "100%", "display": 'inline-block'}),
                                    dcc.Dropdown(
                                        id="param3",
                                        multi=True,
                                        placeholder="Parameter to plot over wavelengths",
                                        persistence_type="session",
                                        disabled=True,
                                        style=dict(width='500px', display='inline-block')
                                    ),
                                    dbc.Button(
                                        "Reset graph",
                                        id="reset_points",
                                        style={'display': 'inline-block', 'verticalAlign': 'top',
                                               'margin-left': '5px'},
                                        outline=True,
                                        color="primary",
                                    ),
                                    dbc.Input(
                                        id="n_bins",
                                        type="number",
                                        min=10,
                                        max=100,
                                        step=1,
                                        value=10,
                                        placeholder="N bins",
                                        style={'display': 'inline-block',
                                               'verticalAlign': 'top',
                                               'margin-left': '8px',
                                               'width': '5%'
                                               },
                                    )
                                ]
                            )
                        ])
                    ], id="spectra-row")
                ]),
            ])
        ], width=10)
    ])
], style={'padding': 40, 'zIndex': 0})


def _get_data_properties(data_dict: Dict) -> List:
    """
    extracts all keywords from ``data_dict`` that match properties from ``Tags`` which start with ``DATA_FIELD``. This
    function iterates also in nested dictionaries of ``data_dict``

    :param data_dict: dictionary containing SIMPA output
    :return: list of keywords in ``data_dict`` also in nested dictionaries
    """
    expected_fields = [getattr(Tags, t) for t in Tags().__dir__() if t.startswith('DATA_FIELD_')]
    data_fields = []
    for k, item in data_dict.items():
        if k in expected_fields:
            data_fields.append(k)
        if isinstance(item, Dict):
            data_fields += _get_data_properties(data_dict=item)
    return data_fields


def _load_data_fields() -> None:
    """
    extracts all data fields from ``DATA.simpa_output`` and stores them in ``DATA.simpa_data_fields. It iterates over
    all tags in ``Tags`` and stores the ones that start with ``DATA_FIELD`` and exist in ``simpa_output``

    :return: None
    """
    DATA.wavelengths = DATA.simpa_output["settings"]["wavelengths"]
    segment_class = SegmentationClasses()
    DATA.segmentation_labels = [k for k in segment_class.__dir__() if '__' not in k]
    DATA.segmentation_labels = {getattr(segment_class, k): k for k in DATA.segmentation_labels}
    data_fields = dict()
    sim_props = _get_data_properties(data_dict=DATA.simpa_output)

    for data_field in sim_props:
        data_fields[data_field] = dict()
        for wavelength in DATA.wavelengths:
            try:
                data_wv = get_data_field_from_simpa_output(DATA.simpa_output, data_field, wavelength)
            except (ValueError, KeyError) as err:
                warnings.warn(f"failed to query {data_field}, got error: \n{err}")
                continue
            if isinstance(data_wv, np.ndarray):
                if "time_series" not in data_field:
                    if len(data_wv.shape) == 2:
                        data_wv = np.rot90(data_wv, k=1, axes=(0, 1))
                    elif len(data_wv.shape) == 3:
                        data_wv = np.rot90(data_wv, k=1, axes=(0, 2))
                data_fields[data_field][wavelength] = data_wv
                data_fields[data_field]["shape"] = len(data_wv.shape)
                data_fields[data_field]["axis_labels"] = _get_axis_labels(param=data_field)
        if not data_fields[data_field]:
            del data_fields[data_field]
    DATA.simpa_data_fields = data_fields


def _get_axis_labels(param: str) -> Dict:
    """
    creates a dictionary with labels for each axis depending on ``param``. Each key of returned dictionary is an
    ``int`` representing the axis of the data volume represented by ``param``

    :param param: parameter for which axis labels are desired
    :return: dictionary with labels
    """
    if param == "reconstructed_data":
        labels = {0: 'x', 1: 'y'}
    elif param == "time_series_data":
        labels = {0: 'No. elements [a.u.]', 1: 'Time'}
    else:
        labels = {0: 'x', 1: 'y', 2: 'z'}
    return labels


def to_pandas(data_dict: Dict) -> pd.DataFrame:
    """
    transforms dictionary to pandas dataframe and metls it according to variable ``Parameter`` and value name ``Value``

    :param data_dict: dictionary containing data to be transformed
    :return: pandas DataFrame
    """
    df = pd.DataFrame(data_dict)
    df = df.melt(var_name="Parameter", value_name="Value")
    return df


def get_structure_names(array: np.ndarray) -> np.ndarray:
    """
    transforms int values in ``array`` to their corresponding name of structure based on ``DATA.segmentation_labels``

    :param array: array of `ìnt values``
    :return: array of strings representing the names of the structures
    """
    names = array.copy().astype('str')
    unique_values = np.unique(names)
    for v in unique_values:
        names[names == v] = DATA.segmentation_labels.get(float(v))
    return names


def get_plot_range(datafield):
    mins, maxs = list(), list()
    for wl in DATA.wavelengths:
        mins.append(np.nanmin(datafield[wl]))
        maxs.append(np.nanmax(datafield[wl]))
    minimum = np.min(mins)
    maximum = np.max(maxs)
    step = 0.1*(maximum - minimum)
    return minimum, maximum, step, {i: {'label': f'{i:1.1f}'} for i in np.arange(minimum, maximum, step)}


def plot_data_field(plot_name: str,
                    data_field: Union[None, str],
                    colorscale: str,
                    wavelength: Union[int, float],
                    z_range: Union[List, Tuple],
                    plot_type: Union[None, str],
                    axis: int,
                    axis_ind: int,
                    _: Dict,
                    fix_ranges: bool) -> Tuple[go.Figure, bool, float, float, Union[float, int], dict, list]:
    """
    generates plot to update ``plot_11`` in ``app.layout``. In addition, it determines if the scaler
    sliders are activated or deactivated. All data used for plotting is extracted from ``DATA``

    :param plot_name: name of the plot in layout of app that triggered this function. Used to query the correct layout.
    :param data_field: parameter from simpa output to be plotted
    :param colorscale: colorscale used for plot
    :param wavelength: wavelength to be extracted
    :param z_range: range used to scale colorscale of plot
    :param plot_type: type of plot to be generated
    :param axis: axis along which data is extracted
    :param axis_ind: index along ``axis`` from which data is extracted
    :param _: dictionary containing clicked data from plot_11. Dummy parameter used to trigger callback
        on click
        :param fix_ranges: boolean value used to fix the ranges (zoom) on displayed plots
    :return: go.Figure and bool indicating if sliders for scaling are deactivated or activated
    """
    if data_field is None or wavelength is None:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        component_id = None
    else:
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    z_min, z_max, z_step, marks = get_plot_range(DATA.simpa_data_fields[data_field])
    if component_id == "param1":
        z_range = [z_min, z_max]
        plot_scaler_output = (False, z_min, z_max, z_step, marks, z_range)
    else:
        plot_scaler_output = (no_update, no_update, no_update, no_update, no_update, no_update)

    plot_data, axis_labels = get_current_frame(data_field=data_field,
                                               wavelength=wavelength,
                                               axis_ind=axis_ind,
                                               axis=axis)
    if plot_data is None:
        raise PreventUpdate

    if fix_ranges:
        new_ranges = get_zoom_layout(plot_name)
    else:
        new_ranges = {}
    if plot_type == "imshow":
        kwargs = dict()
        if data_field == "seg":
            custom_data = get_structure_names(plot_data)
            kwargs["hovertext"] = custom_data
        plot = [go.Heatmap(z=plot_data,
                           colorscale=colorscale,
                           zmin=z_range[0],
                           zmax=z_range[1],
                           **kwargs)]
        figure = go.Figure(data=plot)
        figure.update_layout(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
        )
        figure.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        for i, x in enumerate(DATA.click_points["x"]):
            if x:
                if (DATA.click_points["param"][
                        i] in PROPERTIES_TO_IGNORE_ON_CLICK or data_field in PROPERTIES_TO_IGNORE_ON_CLICK) \
                        and data_field != DATA.click_points["param"][i]:
                    continue
                figure.add_annotation(x=x,
                                      y=DATA.click_points["y"][i],
                                      text=str(i),
                                      showarrow=True,
                                      arrowhead=6,
                                      bgcolor="#ffffff",
                                      opacity=0.8
                                      )

        for i, shape in enumerate(DATA.shapes):
            if shape:
                figure.add_shape(
                    **shape
                )

    else:
        raise PreventUpdate
    figure.plotly_relayout(new_ranges)
    return figure, *plot_scaler_output


def get_current_frame(data_field: Union[List, str],
                      wavelength: Union[int, float],
                      axis_ind: int,
                      axis: int) -> Tuple[Union[np.ndarray, Dict], List]:
    """
    Extracts view of volume data. If ``data_field`` is a list, a dictionary is created where each item is an
    ``np.ndarray`` and each key the name of the parameter. All data is extracted from ``DATA``

    :param data_field: name of parameter from which the volume is extracted
    :param wavelength: wavelength that is desired to be extracted
    :param axis_ind: index along ``axis`` that is desired
    :param axis: axis that is desired to be extracted
    :return: array of dictionary of arrays with extracted data
    """
    plot_data = None
    axis_labels = None
    if isinstance(data_field, str):
        if len(DATA.simpa_data_fields[data_field][wavelength].shape) == 3:
            axis_labels = [DATA.simpa_data_fields[data_field]["axis_labels"][i] for i in range(2)]
            plot_data = np.take(DATA.simpa_data_fields[data_field][wavelength], indices=axis_ind, axis=axis)
        elif len(DATA.simpa_data_fields[data_field][wavelength].shape) == 2:
            axis_labels = [DATA.simpa_data_fields[data_field]["axis_labels"][i] for i in range(2)]
            plot_data = DATA.simpa_data_fields[data_field][wavelength]
    elif isinstance(data_field, list):
        plot_data = dict()
        for k in data_field:
            if len(DATA.simpa_data_fields[k][wavelength].shape) == 3:
                axis_labels = [DATA.simpa_data_fields[k]["axis_labels"][i] for i in range(2)]
                plot_data[k] = np.take(DATA.simpa_data_fields[k][wavelength], indices=axis_ind, axis=axis)
            elif len(DATA.simpa_data_fields[k][wavelength].shape) == 2:
                axis_labels = [DATA.simpa_data_fields[k]["axis_labels"][i] for i in range(2)]
                plot_data[k] = DATA.simpa_data_fields[k][wavelength]
    return plot_data, axis_labels


def get_data_from_last_shape(relayout_data: Dict,
                             plot_data: Dict,
                             data_field: str,
                             axis: int,
                             axis_ind: int,
                             return_mask: bool = False) -> Union[None, Dict]:
    """
    It extracts the data from ``plot_data`` given ``relayout_data`` incoming from updates in a plotly Figure.
    It extracts the data based on the last shape in ``relayout_data``. Only supports shapes of type
    ``path`` and ``rect``

    :param return_mask: If True, the mask instead of the spectra is returned
    :param relayout_data: dictionary containing all relayout data from a plotly Figure
    :param plot_data: dictionary containing all the data to be plotted
    :param data_field: data field selected in ``param1`` when callback was triggered
    :param axis_ind: index along axis from which to extract data
    :param axis: axis from which to extract data
    :return: None or dictionary containing the extracted data as flattened numpy arrays.
    """
    shapes = relayout_data.get('shapes')
    results = dict()
    for k in plot_data:
        if (k in PROPERTIES_TO_IGNORE_ON_CLICK or data_field in PROPERTIES_TO_IGNORE_ON_CLICK) and k != data_field:
            continue
        path = None
        array = plot_data[k]
        if shapes:
            last_shape = relayout_data["shapes"][-1]
            shape_type = last_shape.get("type")
            if shape_type in ["rect"]:
                x0, y0 = int(round(last_shape["x0"])), int(round(last_shape["y0"]))
                x1, y1 = int(round(last_shape["x1"])), int(round(last_shape["y1"]))
                x0, x1 = min([x0, x1]), max(x0, x1)
                y0, y1 = min([y0, y1]), max(y0, y1)
                mask = np.zeros(array.shape, dtype=bool)
                mask[y0:y1, x0:x1] = True
                mask = ndimage.binary_fill_holes(mask)
            elif shape_type in ["path"]:
                mask = path_to_mask(last_shape.get("path"), array.shape)
            if return_mask:
                return mask
            spectral_values = list()
            for wavelength in DATA.wavelengths:
                if len(DATA.simpa_data_fields[data_field][wavelength].shape) == 3:
                    array = np.take(DATA.simpa_data_fields[data_field][wavelength], indices=axis_ind,
                                    axis=axis)
                else:
                    array = DATA.simpa_data_fields[data_field][wavelength]
                masked_array = array[mask]
                spectral_values.append(np.mean(masked_array))
            return spectral_values
        elif any(["shapes" in k and "x0" in k for k in relayout_data]):
            keys = {"x0": [k for k in relayout_data if "x0" in k][0],
                    "x1": [k for k in relayout_data if "x1" in k][0],
                    "y0": [k for k in relayout_data if "y0" in k][0],
                    "y1": [k for k in relayout_data if "y1" in k][0]}
            x0, y0 = int(relayout_data[keys["x0"]]), int(relayout_data[keys["y0"]])
            x1, y1 = int(relayout_data[keys["x1"]]), int(relayout_data[keys["y1"]])
            x0, x1 = min([x0, x1]), max(x0, x1)
            y0, y1 = min([y0, y1]), max(y0, y1)
            mask = np.zeros(array.shape, dtype=bool)
            mask[y0:y1, x0:x1] = True
            if return_mask:
                return mask
            roi_array = array[mask]
            if not roi_array.size:
                return None
            else:
                return roi_array.ravel()
    return results if results else None


def point_already_in_data(x: int,
                          y: int,
                          points: dict) -> bool:
    """
    checks if a point ``(x,y)`` already exists in a dictionary of ``points``

    :param x: x value
    :param y: y value
    :param points: dictionary of points stored as ``{'x': [...], 'y': [...]}``
    :return:
    """
    tuples = [(a, b) for a, b in zip(points["x"], points["y"])]
    return (x, y) in tuples


def path_to_indices(path: str) -> np.ndarray:
    """
    transforms SVG path to numpy array of coordinates, each row being a (row, col) point

    :param path: string representing the SVG path to be transformed
    :return: numpy array with coordinates
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)


def path_to_mask(path: str, shape: Tuple) -> np.ndarray:
    """
    transforms SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.

    :param path: string representing the SVG path to be transformed
    :param shape: shape of mask to be generated
    :return:
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def update_relayout(relayout: Dict, p: str):
    """
    Updates the relayout data in the DATA class

    :param relayout: new relayout data to be updated
    :param p: data field that triggered the relayout
    :return:
    """
    global DATA
    if relayout:
        DATA.relayout_data['param'] = p
        DATA.relayout_data['relayout'] = relayout


def get_zoom_layout(plot_name: str) -> Dict:
    """
    queries the range layout from ``DATA.plot_layout`` based on the name of a plot component in the app layout. If the
    axis ranges are not found in ``DATA.plot_layout``, that axis is set to ``autorange``. The output cna be used to
    update the layout of a plotly plot by calling ``figure.plotly_relayout``

    :param plot_name: name of a plot in the layout of the app, e.g. plot_11
    :return: dictionary containing the layout of the axis ranges
    """
    keys_to_update = ['xaxis.range[0]',
                      'xaxis.range[1]',
                      'yaxis.range[0]',
                      'yaxis.range[1]',
                      ]
    layout = DATA.plot_layouts.get(plot_name)
    default = {'xaxis.autorange': True, 'yaxis.autorange': True}
    new_layout = {k: i for k, i in layout.items() if k in keys_to_update}
    default.update(new_layout)
    if default.get('xaxis.range[0]'):
        default.pop('xaxis.autorange')
    if default.get('yaxis.range[0]'):
        default.pop('yaxis.autorange')
    return default


@app.callback(Output("dummy2", "children"),
              Input("plot_11", "relayoutData"),
              Input("plot_21", "relayoutData"),
              Input("fix_ranges", "value"),
              )
def update_plot_layouts(layout1: Dict, layout3: Dict, _) -> []:
    """
    updates layouts in ``DATA.plot_layouts`` based on ``relayoutData`` triggered by ``Graph`` components in app and on
    switch controlling if axis ranges should be kept fixed in each plot.

    :param layout1: dictionary with relayout data from plot_11
    :param layout3: dictionary with relayout data from plot_13
    :param _: Dummy used to trigger callback when switch controlling plot axis ranges changes position
    :return: empty list, dummy used because all callbacks need to have an output
    """
    plot_names = ['plot_11', 'plot_21']
    for i, layout in enumerate([layout1, layout3]):
        if layout and any(['xaxis.range' in k or 'yaxis.range' for k in layout]):
            DATA.plot_layouts[plot_names[i]] = layout
    return []


@app.callback(
    Output("file_selection", "options"),
    [Input("data_path", "valid")],
    [State("data_path", "value")]
)
def populate_file_selection(valid, data_path: Union[bool, str]) -> List[Dict]:
    """
    queries all files in directory and filters them according to ending: ``.hdf5``. Creates a list of dictionaries
    based on such files: ``[{'label': ..., 'value': ...}, ...]``. If data path is invalid then ``PreventUpdate`` is
    raised

    :param data_path: path to file or folder containing ``.hdf5`` simulations
    :return: List of dictionaries ``[{'label': ..., 'value': ...}, ...]``
    """
    if valid:
        if os.path.isdir(data_path):
            file_list = os.listdir(data_path)
            file_list = [f for f in file_list if f.endswith('.hdf5')]
            options_list = [{'label': i, 'value': os.path.join(data_path, i)} for i in file_list]
            return options_list
        elif os.path.isfile(data_path):
            file_list = [os.path.basename(data_path)]
            options_list = [{'label': i, 'value': data_path} for i in file_list]
            return options_list
        else:
            raise PreventUpdate("Please select a file or folder!")
    else:
        raise PreventUpdate()


@app.callback(Output('data_path', 'invalid'),
              Output('data_path', 'valid'),
              Input('data_path', 'value'),
              )
def update_data_path_validity(data_path: Union[str, None]) -> Tuple[bool, bool]:
    """
    updates appearance of input component in ``app.layout`` to represent the validity of path to data

    :param data_path: path to file or folder containing ``.hdf5`` files containing simulations generated with ``SIMPA``
    :return: tuple of boolean values to update ``invalid`` and ``valid`` properties of ``data_path`` component
    """
    if isinstance(data_path, str):
        valid = True if (os.path.isfile(data_path) and data_path.endswith('.hdf5')) or \
                        os.path.isdir(data_path) else False
    else:
        valid = False
    return 1 - valid, valid


@app.callback(Output("n_bins", "disabled"),
              Input("annotate_switch", "value"))
def deactivate_n_bins(annotate: bool) -> bool:
    """
    enables ``n_bins`` input component when annotation process is started

    :param annotate: indicates if annotations process is started
    :return: bool that activates ``n_bins`` input component
    """
    return 1 - annotate


@app.callback(Output("dummy3", "children"),
              Input("plot_11", "relayoutData"),
              )
def save_shapes(ann):
    if "shapes" in ann:
        DATA.shapes = ann["shapes"]
    return []


@app.callback(Output("dummy4", "children"),
              Input("save_anno", "n_clicks"),
              State("file_selection", "value"),
              State("plot_11", "relayoutData"),
              State("param3", "value"),
              State("volume_slider", "value"),
              State("volume_axis", "value"),
              State("channel_slider", "value"),
              State("param1", "value"),
              prevent_initial_update=True
              )
def save_segmentation(_, file_path, relayout_data, data_field, axis_ind, axis, wavelength, data_field1):
    if data_field is None:
        raise PreventUpdate
    if DATA.relayout_data['param'] in PROPERTIES_TO_IGNORE_ON_CLICK:
        data_field = [DATA.relayout_data['param']]
    else:
        data_field = [d for d in data_field if d not in PROPERTIES_TO_IGNORE_ON_CLICK]
    plot_data, _ = get_current_frame(data_field=data_field,
                                     wavelength=wavelength,
                                     axis_ind=axis_ind,
                                     axis=axis)
    if plot_data is None:
        raise PreventUpdate
    try:
        mask = get_data_from_last_shape(relayout_data, plot_data, data_field=data_field1, axis=axis,
                                        axis_ind=axis_ind, return_mask=True)
    except IndexError:
        raise PreventUpdate

    save_data_field(np.rot90(mask, -1), file_path, Tags.DATA_FIELD_SEGMENTATION)

    return []


@app.callback(
    Output("param1", "options"),
    Output("param1", "disabled"),
    Output("channel_slider", "min"),
    Output("channel_slider", "max"),
    Output("channel_slider", "value"),
    Output("channel_slider", "step"),
    Output("channel_slider", "marks"),
    Output("volume_slider", "min"),
    Output("volume_slider", "max"),
    Output("volume_slider", "value"),
    Output("volume_slider", "step"),
    Output("volume_slider", "marks"),
    Output("volume_slider", "disabled"),
    Output("param3", "options"),
    Output("param3", "disabled"),
    Output("plot_type1", "value"),
    Output("plot_type1", "options"),
    Output("plot_type1", "disabled"),
    Input("file_selection", "value"),
    Input("volume_axis", "value")
)
def populate_file_params(file_path: Union[None, str],
                         axis: int) -> \
        Tuple[List[Dict[str, Any]], bool, int, int, Any, Union[int, Any], Dict, int, Union[int, Any], int, int, Dict,
              bool, List[Dict[str, Any]], bool, str, List[Dict[str, Any]], bool]:
    """
    loads data selected from ``file_selection``input component if this was the triggering component. The data is sliced
    along new axis when the triggering component is ``volume_axis``.

    :param file_path: path to ``.hdf5`` file containing simulations generated with SIMPA
    :param axis: axis along which a view of the data is desired
    :return:
    """
    global DATA
    ctx = callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "volume_axis" and DATA.shape is not None:
        if len(DATA.shape) == 3:
            n_slices = DATA.shape[axis]
            disable_vol_slider = False
            vol_slider_marks = {i: {'label': str(i)} for i in range(n_slices)[::int(n_slices / 10)]}
            vol_slider_min = 0
            vol_slider_max = n_slices - 1
            vol_slider_value = 0
        elif len(DATA.shape) == 2:
            disable_vol_slider = True
            vol_slider_marks = {i: {'label': str(i)} for i in range(0, 10, 11)}
            vol_slider_min = 0
            vol_slider_max = 10
            vol_slider_value = 0
        else:
            raise ValueError(f"Number of dimensions of mua is not supported: {DATA.shape}")
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update, vol_slider_min,
                vol_slider_max, vol_slider_value, 1, vol_slider_marks, disable_vol_slider,
                no_update, no_update, no_update, no_update, no_update)
    if file_path is None:
        raise PreventUpdate()
    if os.path.isfile(file_path):
        DATA.simpa_output = load_hdf5(file_path)
        _load_data_fields()
        try:
            DATA.shape = DATA.simpa_data_fields['mua'][DATA.wavelengths[0]].shape
        except KeyError as kerr:
            try:
                DATA.shape = DATA.simpa_data_fields['reconstructed_data'][DATA.wavelengths[0]].shape
            except KeyError as kerr:
                raise ("What's going on here?", kerr)
        if len(DATA.shape) == 3:
            try:
                n_slices = DATA.simpa_data_fields['mua'][DATA.wavelengths[0]].shape[1]
            except KeyError as kerr:
                try:
                    n_slices = DATA.simpa_data_fields['reconstructed_data'][DATA.wavelengths[0]].shape[1]
                except KeyError as kerr:
                    raise ("What's going on here?", kerr)
            disable_vol_slider = False
            vol_slider_marks = {i: {'label': str(i)} for i in range(n_slices)[::int(n_slices / 10)]}
            vol_slider_min = 0
            vol_slider_max = n_slices - 1
            vol_slider_value = 0
        elif len(DATA.shape) == 2:
            disable_vol_slider = True
            vol_slider_marks = {i: {'label': str(i)} for i in range(0, 10, 1)}
            vol_slider_min = 0
            vol_slider_max = 10
            vol_slider_value = 0
        else:
            raise ValueError(f"Number of dimensions of array is not supported: {DATA.shape}")
        plot_options = [{'label': t, 'value': t} for t in DATA.plot_types if "3d" not in t]
        options_list = [{'label': key, 'value': key} for key in DATA.simpa_data_fields.keys() if key != 'units']
        if len(DATA.wavelengths) > 20:
            marks = {int(wv): str(wv) for wv in DATA.wavelengths[::int(len(DATA.wavelengths) / 10)]}
        else:
            marks = {int(wv): str(wv) for wv in DATA.wavelengths}
        if len(DATA.wavelengths) == 1:
            wv_step = 1
        else:
            wv_step = DATA.wavelengths[1] - DATA.wavelengths[0]
        return (options_list, False, min(DATA.wavelengths), max(DATA.wavelengths), DATA.wavelengths[0], wv_step, marks,
                vol_slider_min, vol_slider_max, vol_slider_value, 1, vol_slider_marks, disable_vol_slider,
                options_list, False, "imshow", plot_options, False)


@app.callback(
    Output("plot_11", "figure"),
    Output("plot_scaler1", "disabled"),
    Output("plot_scaler1", "min"),
    Output("plot_scaler1", "max"),
    Output("plot_scaler1", "step"),
    Output("plot_scaler1", "marks"),
    Output("plot_scaler1", "value"),
    Input("param1", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value"),
    Input("plot_scaler1", "value"),
    Input("plot_type1", "value"),
    Input("volume_axis", "value"),
    Input("volume_slider", "value"),
    Input("plot_11", "clickData"),
    State("fix_ranges", "value"),
)
def update_plot_11(*args, **kwargs):
    return plot_data_field('plot_11', *args, **kwargs)


@app.callback(Output("dummy", "children"),
              Output("plot_11", "clickData"),
              Output("plot_11", "relayoutData"),
              Input("reset_points", "n_clicks"))
def rest_points(_):
    """
    Resets clicked data in ``DATA.click_points``, ``plot_11``

    :param _: dummy param to trigger callback
    :return: Nulls to update clicked data
    """
    global DATA
    DATA.click_points = {'x': [], 'y': [], 'z': [], 'param': []}
    DATA.shapes = {}
    return [], {}, {}


@app.callback(Output("param3", "value"),
              Input("param1", "value"),
              State("param3", "value"),
              prevent_initial_update=True
              )
def sync_data_fields(param1: Union[None, str, List], param3: Union[None, str, List]) -> List:
    """
    synchronizes data fields from ``param1`` and ``param3`` by appending ``param1``to all existing params in ``param3``

    :param param1: parameter to be appended to ``param3``
    :param param3: target parameter list to be extended
    :return:
    """
    if param1 is None:
        param1 = []
    if param3 is None:
        param3 = []
    if isinstance(param1, str):
        param1 = [param1]
    if isinstance(param3, str):
        param3 = [param3]
    return list(set(param1 + param3))


@app.callback(
    Output("plot_21", "figure"),
    Input("param3", "value"),
    Input("plot_11", "clickData"),
    Input("volume_slider", "value"),
    Input("volume_axis", "value"),
    Input("reset_points", "n_clicks"),
    Input("annotate_switch", "value"),
    Input("plot_11", "relayoutData"),
    Input("channel_slider", "value"),
    Input("n_bins", "value"),
    State("param1", "value"),
    State("fix_ranges", "value"),
)
def plot_spectrum(data_field: Union[None, str],
                  click_data1: Dict,
                  axis_ind: int,
                  axis: int,
                  _: Union[None, int],
                  annotate: bool,
                  relayout_data: Union[None, Dict],
                  wavelength: Union[float, int],
                  n_bins: int,
                  data_field1: Union[str, None],
                  fix_ranges: bool) -> go.Figure:
    """
    generates figure to update spectral visualization or histogram visualization of annotated regions extracted from
    ``plot_11``. Always last annotated shape is used. The type of plot is defined by ``annotate``.
    It is either a line plot or a histogram with rugged plot.

    :param data_field: parameter form simpa output to be plotted
    :param click_data1: clicked data from ``plot_11``
    :param axis_ind: index along axis from which to extract data
    :param axis: axis from which to extract data
    :param _: dummy parameter used to trigger plot reset functionality when button RESET_GRAPH is clicked
    :param annotate: indicates if the data extracted from last annotated shape should be plotted as histogram
    :param relayout_data: relayout data extracted from plotly Figure
    :param wavelength: wavelength to be extracted
    :param n_bins: number of binds used to plot histogram if ``annotate==True``,
    :param data_field1: data field in plot_11
        :param fix_ranges: boolean controlling if the ranges of the plots should be kept constant even when slicing a
    different volume axis or wavelength
    :return: ``go.Figure``
    """
    global DATA
    layout = {}
    if isinstance(data_field, str):
        data_field = [data_field]
    lookup = {0: data_field1}
    ctx = callback_context
    if not ctx.triggered:
        component_id = None
        component_prop = None
    else:
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]
        component_prop = ctx.triggered[0]['prop_id'].split('.')[1]
    if component_id == 'reset_points':
        return go.Figure(data={})
    if component_prop == "relayoutData" and relayout_data and (
            "shapes" in relayout_data or any(["shapes" in k for k in relayout_data])):
        update_relayout(relayout_data, data_field1)
    if fix_ranges:
        new_ranges = get_zoom_layout('plot_21')
    else:
        new_ranges = {}
    if annotate:
        if data_field is None:
            raise PreventUpdate
        if DATA.relayout_data['relayout']:
            if DATA.relayout_data['param'] in PROPERTIES_TO_IGNORE_ON_CLICK:
                data_field = [DATA.relayout_data['param']]
            else:
                data_field = [d for d in data_field if d not in PROPERTIES_TO_IGNORE_ON_CLICK]
            plot_data, _ = get_current_frame(data_field=data_field,
                                             wavelength=wavelength,
                                             axis_ind=axis_ind,
                                             axis=axis)
            if plot_data is None:
                raise PreventUpdate
            try:
                plot_data = get_data_from_last_shape(relayout_data, plot_data, data_field=data_field1, axis=axis,
                                                     axis_ind=axis_ind)
            except IndexError:
                raise PreventUpdate
            if plot_data is None:
                raise PreventUpdate
            layout = go.Layout(xaxis={'autorange': False})
            plot_data = [go.Scatter(x=DATA.wavelengths,
                                    y=plot_data,
                                    mode="lines+markers",
                                    )
                         ]
            figure = go.Figure(data=plot_data, layout=layout)
            figure.plotly_relayout(new_ranges)
            return figure

    if data_field is None or click_data1 is None:
        raise PreventUpdate
    else:
        for i, click_data in enumerate([click_data1]):
            if not click_data:
                continue
            x_c = click_data["points"][0]["x"]
            y_c = click_data["points"][0]["y"]
            if not isinstance(x_c, int) or not isinstance(y_c, int):
                continue
            if point_already_in_data(x_c, y_c, DATA.click_points):
                continue
            DATA.click_points["x"].append(x_c)
            DATA.click_points["y"].append(y_c)
            DATA.click_points["param"].append(lookup.get(i))
        if not DATA.click_points["x"] or not DATA.click_points["y"]:
            raise PreventUpdate
        plot_data = list()
        for i, x in enumerate(DATA.click_points["x"]):
            y = DATA.click_points["y"][i]
            p = DATA.click_points["param"][i]
            for param in data_field:
                if (p in PROPERTIES_TO_IGNORE_ON_CLICK or param in PROPERTIES_TO_IGNORE_ON_CLICK) and param != p:
                    continue
                spectral_values = list()
                for wavelength in DATA.wavelengths:
                    if DATA.simpa_data_fields[param]["shape"] == 3:
                        array = np.take(DATA.simpa_data_fields[param][wavelength], indices=axis_ind,
                                        axis=axis)
                    else:
                        array = DATA.simpa_data_fields[param][wavelength]
                    spectral_values.append(array[y, x])
                layout = go.Layout(xaxis={'autorange': True})
                plot_data += [go.Scatter(x=DATA.wavelengths,
                                         y=spectral_values,
                                         mode="lines+markers",
                                         name=param + f" x={x}, y={y}, ID: {i}",
                                         )
                              ]
        figure = go.Figure(data=plot_data, layout=layout)
        figure.plotly_relayout(new_ranges)
        return figure


@app.callback(
    Output("general_info", "children"),
    Output("toast_content", "children"),
    Output("alert_toast", "is_open"),
    Input("param1", "value"),
    State("channel_slider", "value"),
)
def update_general_info(param1: Union[None, str],
                        wv: Union[int, float]) -> Tuple[List, List, bool]:
    """
    updates general info markdown in ``app.layout`` given `param_1` and a wavelength. Checks if there are invalid values
    in the volume, such as ``np.NaN``, not finite values. It also outputs data shape along each dimension.

    :param param1: parameter to be analyzed
    :param wv: wavelength to be extracted
    :return: list of toast children and if the toast is open or not
    """
    if not param1:
        raise PreventUpdate
    global DATA
    array = DATA.simpa_data_fields[param1][wv]
    shape = array.shape
    size = array.size * len(DATA.wavelengths)
    n_finite = [np.isfinite(DATA.simpa_data_fields[param1][w]).sum() for w in DATA.wavelengths]
    n_finite = np.sum(n_finite)
    n_not_finite = size - n_finite
    children = [
        dcc.Markdown(f'''
        Data shape per channel is `{shape}`\n
        Data size per parameter is `{size}`\n
        Not finite points: `{n_not_finite} -> {100 * n_not_finite / size}%`\n
        ''')
    ]
    if n_not_finite:
        open_toast = True
        toast_child = [
            dcc.Markdown(f"Found not finite values in array: `{n_not_finite} -> {100 * n_not_finite / size}%`")]
    else:
        open_toast = False
        toast_child = []
    return children, toast_child, open_toast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run app in debug mode. Changes in code will update the app automatically.")
    parser.add_argument("-p", "--port", type=int,
                        help="Specify the port where the app should run (localhost:port), default is 8050")
    parser.add_argument("-host", "--host_name", type=str,
                        help="Name of the host on which the app is run, by default it is localhost")
    arguments = parser.parse_args()
    port = 8050
    host_name = "localhost"
    if arguments.port:
        port = arguments.port
    if arguments.host_name:
        host_name = arguments.host_name
    app.run_server(debug=False, host=host_name, port=port)
