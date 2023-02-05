from copy import deepcopy

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from packaging.version import parse as parse_version
from qtpy.QtWidgets import (
    QGridLayout,
    QDoubleSpinBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data

    res_layer.metadata["viewer_name"] = name

    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict={}):
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class ExampleWidget(QWidget):
    """
    Dummy widget showcasing how to place additional widgets to the right
    of the additional viewers.
    """

    def __init__(self):
        super().__init__()
        self.btn = QPushButton("Perform action")
        self.spin = QDoubleSpinBox()
        layout = QVBoxLayout()
        layout.addWidget(self.spin)
        layout.addWidget(self.btn)
        layout.addStretch(1)
        self.setLayout(layout)


class MultipleViewerWidget(QWidget):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        # the main viewer
        self.viewer = viewer

        # The secondary viewer
        self.viewer_model1 = ViewerModel(title="model1")
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)

        # The tab widget to add additional widgets
        self.tab_widget = QTabWidget()
        w1 = ExampleWidget()
        w2 = ExampleWidget()
        self.tab_widget.addTab(w1, "Sample 1")
        self.tab_widget.addTab(w2, "Sample 2")

        # The napari qt viewer is already in a layout (box)
        # we add the parent to a temp layout to remove it from the window
        tmp_layout = QGridLayout()
        tmp_layout.addWidget(self.viewer.window._qt_viewer.parent())
        tmp_widget = QWidget()
        tmp_widget.setLayout(tmp_layout)
        # now we add just the viewer to the grid layout
        # this way the two viewers will alway be the same size
        layout = QGridLayout()
        # add stretch factor to ensure even resize
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(self.viewer.window._qt_viewer, 0, 0)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.qt_viewer1, 0, 1)
        # no stretch factor for the status widget
        layout.setColumnStretch(2, 0)
        layout.addWidget(self.tab_widget, 0, 2)
        self.setLayout(layout)

        # connect layers etc.
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)

        # sync camera
        self.viewer.camera.events.zoom.connect(self._viewer_zoom)
        self.viewer_model1.camera.events.zoom.connect(self._viewer_zoom)
        self.viewer.camera.events.center.connect(self._viewer_center)
        self.viewer_model1.camera.events.center.connect(self._viewer_center)
        self.viewer.camera.events.angles.connect(self._viewer_angles)
        self.viewer_model1.camera.events.angles.connect(self._viewer_angles)

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model1.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            return

        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
            event.value.name
        ]

    def _point_update(self, event):
        for model in [self.viewer, self.viewer_model1]:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        self.viewer_model1.layers.insert(
            event.index, copy_layer(event.value, "model1")
        )

        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.viewer_model1.layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
        if event.value.name != ".cross":
            self.viewer_model1.layers[event.value.name].events.data.connect(
                self._sync_data
            )

        event.value.events.name.connect(self._sync_name)

        self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        self.viewer_model1.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_model1.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(
                self.viewer_model1.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False

    def _viewer_zoom(self, event):
        """
        Syncs the zoom between all the viewers
        :param event: The camera event
        """
        self.viewer.camera.zoom = event.source.zoom
        self.viewer_model1.camera.zoom = event.source.zoom

    def _viewer_center(self, event):
        """
        Syncs the center between all the viewers
        :param event: The camera event
        """
        self.viewer.camera.center = event.source.center
        self.viewer_model1.camera.center = event.source.center

    def _viewer_angles(self, event):
        """
        Syncs the angles between all the viewers
        :param event: The camera event
        """
        self.viewer.camera.angles = event.source.angles
        self.viewer_model1.camera.angles = event.source.angles


if __name__ == "__main__":
    view = napari.Viewer()
    # create the multi view and make it central
    multi_view = MultipleViewerWidget(view)
    view.window._qt_window.setCentralWidget(multi_view)
    napari.run()
