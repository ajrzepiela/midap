from copy import deepcopy
from pathlib import Path

import napari
import numpy as np
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info
from packaging.version import parse as parse_version
from qtpy.QtWidgets import (
    QGridLayout,
    QWidget,
)

from midap.correction.data_handler import CorrectionData, LineageOPException
from midap.correction.side_panel_gui import InfoBox, FrameSlider, SaveButton

# TODO: remove timers
from time import perf_counter

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


class MultipleViewerWidget(QWidget):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer, correction_data: CorrectionData):
        """
        Inits the widget
        :param viewer: The napari viewer to use
        :param correction_data: The correction data that should be displayed
        """
        super().__init__()

        # get the number of frames
        self.correction_data = correction_data
        self.n_frames, self.data_width, self.data_height = self.correction_data.images.shape

        # check if we have at least to images
        if self.n_frames < 2:
            raise ValueError("The viewer needs at least the track of 2 images!")
        # we need at least 1 tracked cell
        if len(self.correction_data.tracking_data.track_df) == 0:
            raise ValueError("No cell tracks in the track data frame!")

        # the main viewer
        self.main_viewer = viewer

        # The secondary viewer (do not change the title)
        self.side_viewer = ViewerModel(title="model1")
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.side_viewer)

        # The info box with al
        self.info_box = InfoBox(viewer=viewer, correction_data=correction_data, change_frame_callback=self.change_frame,
                                update_modifier_callback=self.update_settings)

        # the slider in the bottom
        self.slider = FrameSlider(max_value=self.n_frames-2,
                                  frame_change_callback=self.change_frame,
                                  update_on_same=False)

        # save button
        # TODO: Format and add function, create a subclass for this as info box to get the style
        self.save_btn = SaveButton()

        # The napari qt viewer is already in a layout (box)
        # we add the parent to a temp layout to remove it from the window
        tmp_layout = QGridLayout()
        tmp_layout.addWidget(self.main_viewer.window._qt_viewer.parent())
        tmp_widget = QWidget()
        tmp_widget.setLayout(tmp_layout)
        # now we add just the viewer to the grid layout
        # this way the two viewers will alway be the same size
        layout = QGridLayout()
        # add stretch factor to ensure even resize
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(self.main_viewer.window._qt_viewer, 0, 0)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.qt_viewer1, 0, 1)
        # no stretch factor for the status widget
        layout.setColumnStretch(2, 0)
        layout.addWidget(self.info_box, 0, 2)
        layout.setRowStretch(1, 0)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        layout.addWidget(self.save_btn, 1, 2)
        self.setLayout(layout)

        # create the image layers
        self.current_frame = 0

        # the special names are layer that are differently named in both viewer, other layers are the same and copied
        self.special_names = ["Image", "Labels", "Selection"]

        # the image layers
        self.main_img_layer = self.main_viewer.add_image(data=self.correction_data.get_image(self.current_frame),
                                                         name="Image",
                                                         rgb=False,
                                                         contrast_limits=[0,1],
                                                         scale=(1,1))
        self.side_img_layer = self.side_viewer.add_image(data=self.correction_data.get_image(self.current_frame + 1),
                                                         name="SideImage",
                                                         rgb=False,
                                                         contrast_limits=[0, 1],
                                                         scale=(1, 1))

        # add the label images
        self.main_label_layer = self.main_viewer.add_labels(data=self.correction_data.get_label(self.current_frame),
                                                            name="Labels",
                                                            num_colors=50)
        self.side_label_layer = self.side_viewer.add_labels(data=self.correction_data.get_label(self.current_frame + 1),
                                                            name="SideLabels",
                                                            num_colors=50)

        # the selection
        self.selection = None
        selection_color_dict = {1: "yellow",
                                2: "orange",
                                3: "blue",
                                4: "red"}
        test_select = np.zeros_like(self.correction_data.get_label(frame_number=0))
        self.main_select_layer = self.main_viewer.add_labels(data=test_select,
                                                             name="Selection",
                                                             color=selection_color_dict,
                                                             opacity=1.0)
        self.side_select_layer = self.side_viewer.add_labels(data=test_select,
                                                             name="SideSelection",
                                                             color=selection_color_dict,
                                                             opacity=1.0)

        # General settings
        self.mark_orphans = False
        self.mark_dying = False
        self.sync_viewers = True
        self.rotate_view = False

        # info box update
        self.info_box.update_info(current_frame=self.current_frame, selection=self.selection)

        # connect the special layers
        self.main_img_layer.events.visible.connect(self._visibility_chane)
        self.main_label_layer.events.visible.connect(self._visibility_chane)
        self.main_select_layer.events.visible.connect(self._visibility_chane)

        # sync layers (one directional because the tools are only for the main viewer)
        self.main_viewer.layers.events.inserted.connect(self._layer_added)
        self.main_viewer.layers.events.removed.connect(self._layer_removed)
        self.main_viewer.layers.events.moved.connect(self._layer_moved)
        self.main_viewer.layers.selection.events.active.connect(self._layer_selection_changed)

        # sync status and resets (one directional)
        self.main_viewer.events.reset_view.connect(self._reset_view)
        self.side_viewer.events.status.connect(self._status_update)

        # sync dims (just for completion, should not be relevant)
        self.main_viewer.dims.events.current_step.connect(self._point_update)
        self.side_viewer.dims.events.current_step.connect(self._point_update)
        self.main_viewer.dims.events.order.connect(self._order_update)

        # sync camera
        self.main_viewer.camera.events.zoom.connect(self._viewer_zoom)
        self.side_viewer.camera.events.zoom.connect(self._viewer_zoom)
        self.main_viewer.camera.events.center.connect(self._viewer_center)
        self.side_viewer.camera.events.center.connect(self._viewer_center)
        self.main_viewer.camera.events.angles.connect(self._viewer_angles)
        self.side_viewer.camera.events.angles.connect(self._viewer_angles)

        # key binds
        self.main_viewer.bind_key('Left', self.left_arrow_key_bind)
        self.main_viewer.bind_key('Right', self.right_arrow_key_bind)
        self.main_viewer.bind_key('Ctrl-u', self.undo_ops)
        self.main_viewer.bind_key('Ctrl-r', self.redo_ops)

        # mouse clicks
        self.main_select_layer.mouse_drag_callbacks.append(self.track_clicked)
        self.side_select_layer.mouse_drag_callbacks.append(self.track_clicked)

    def _status_update(self, event):
        """
        Updates the status of the viewer (message displayer in the lower left corner) only needs to sync in 1 direction
        :param event: The event that triggered the status change
        """
        self.main_viewer.status = event.value

    def _reset_view(self):
        """
        Syncs the reset of the view one-directional
        """
        self.side_viewer.reset_view()

    def _layer_selection_changed(self, event):
        """
        Syncs the reset of the view one-directional
        """
        if self._block:
            return

        if event.value is None:
            self.side_viewer.layers.selection.active = None
            return

        # catch special names
        if (name := event.value.name) in self.special_names:
            name = f"Side{name}"

        self.side_viewer.layers.selection.active = self.side_viewer.layers[name]

    def _point_update(self, event):
        """
        Syncs the point updates of the view one-directional
        :param event: The trigger event
        """
        for model in [self.main_viewer, self.side_viewer]:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        """
        Order sync for the dims (one-directional)
        """
        order = list(self.main_viewer.dims.order)
        if len(order) <= 2:
            self.side_viewer.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.side_viewer.dims.order = order

    def _layer_added(self, event):
        """
        Add layer to additional viewers and connect all required events
        :param event: The event that trigger the layer addition
        """
        self.side_viewer.layers.insert(event.index, copy_layer(event.value, "model1"))

        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(own_partial(self._property_sync, name))

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.side_viewer.layers[event.value.name].events.set_data.connect(self._set_data_refresh)

        self.side_viewer.layers[event.value.name].events.data.connect(self._sync_data)

        event.value.events.name.connect(self._sync_name)

        self._order_update()

    def _sync_name(self, event):
        """
        Sync the name of layers (if not special)
        :param event: The event that triggered the rename
        """

        if (name := event.source.name) not in self.special_names:
            index = self.main_viewer.layers.index(event.source)
            self.side_viewer.layers[index].name = name

    def _sync_data(self, event):
        """
        Sync data modification from additional viewers  (not special layers)
        :param event: The event that changed the data
        """
        if self._block or (name := event.source.name) in self.special_names:
            return
        for model in [self.main_viewer, self.side_viewer]:
            layer = model.layers[name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        Synchronize data refresh between layers (not special layers)
        :param event: Event that triggered the refresh
        """
        if self._block or (name := event.source.name) in self.special_names:
            return
        for model in [self.main_viewer, self.side_viewer]:
            layer = model.layers[name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """
        Remove layer in all viewers
        :param event: Event that triggered the removeal
        """
        self.side_viewer.layers.pop(event.index)

    def _layer_moved(self, event):
        """
        Update order of layers
        :param event: Event that triggered the move
        """
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.side_viewer.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """
        Sync layers properties (except the name)
        :param name: Name of the property
        :param event: event that triggered sync
        """
        if event.source not in self.main_viewer.layers:
            return
        try:
            # catch special names
            if (layer_name := event.source.name) in self.special_names:
                layer_name = f"Side{name}"
            self._block = True
            setattr(
                self.side_viewer.layers[layer_name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False

    def _visibility_chane(self, event):
        """
        Syncs the change of visibility from the main to the side special layers
        :param event: The event for the change
        """
        # catch special names
        if (name := event.source.name) in self.special_names:
            name = f"Side{name}"
        self.side_viewer.layers[name].visible = event.source.visible

    def _viewer_zoom(self, event):
        """
        Syncs the zoom between all the viewers
        :param event: The camera event
        """
        if self.sync_viewers:
            self.main_viewer.camera.zoom = event.source.zoom
            self.side_viewer.camera.zoom = event.source.zoom

    def _viewer_center(self, event):
        """
        Syncs the center between all the viewers
        :param event: The camera event
        """
        if self.sync_viewers:
            self.main_viewer.camera.center = event.source.center
            self.side_viewer.camera.center = event.source.center

    def _viewer_angles(self, event):
        """
        Syncs the angles between all the viewers
        :param event: The camera event
        """
        if self.sync_viewers:
            self.main_viewer.camera.angles = event.source.angles
            self.side_viewer.camera.angles = event.source.angles

    def change_frame(self, frame: int, update_on_same=True):
        """
        Changes the current frame to frame
        :param frame: The int of the new frame of the main viewer
        :param update_on_same: Update the frame data even though the number did not change
        """

        # if the frame did not change and we don't want to update -> return
        if not update_on_same and frame == self.current_frame:
            return

        # catch our of bounds
        if frame >= self.n_frames - 1:
            frame = self.n_frames - 2

        # update
        t0 = perf_counter()
        self.main_img_layer.data = self.correction_data.get_image(frame_number=frame)
        self.main_label_layer.data = self.correction_data.get_label(frame_number=frame)
        t1 = perf_counter()
        print(f"Main update: {t1 - t0}")

        t0 = perf_counter()
        self.side_img_layer.data = self.correction_data.get_image(frame_number=frame + 1)
        self.side_label_layer.data = self.correction_data.get_label(frame_number=frame + 1)
        t1 = perf_counter()
        print(f"Side update: {t1 - t0}")

        self.current_frame = frame

        # selection
        t0 = perf_counter()
        self.update_selection(selection=self.selection)
        t1 = perf_counter()
        print(f"Selection update: {t1 - t0}")

        # info
        t0 = perf_counter()
        self.info_box.update_info(current_frame=self.current_frame, selection=self.selection)
        t1 = perf_counter()
        print(f"Info update: {t1 - t0}")

        # the slider
        t0 = perf_counter()
        self.slider.set_value(self.current_frame)
        t1 = perf_counter()
        print(f"Slider update: {t1 - t0}")

        # set the focus
        self.main_viewer.window._qt_viewer.setFocus()

    def left_arrow_key_bind(self, *args):

        """
        Key bind for the left arrow key (decrease frame number)
        """
        if self.current_frame == 0:
            self.main_viewer.status = "Already at first image"
            return
        self.change_frame(self.current_frame - 1)

    def right_arrow_key_bind(self, *args):
        """
        Key bind for the right arrow key (increase frame number)
        """

        if self.current_frame + 2 == self.n_frames:
            self.main_viewer.status = "Already at last image"
            return
        self.change_frame(self.current_frame + 1)

    def track_clicked(self, layer, event):
        """
        Handle for the mouse clicks on the tracks
        :param event: The event of the mouse click
        """
        # Mouse down
        dragged = False
        yield
        # if we are on the move, we do nothing
        while event.type == 'mouse_move':
            dragged = True
            yield
        if dragged:
            # if we dragged the side viewer, we need to reset the focus manually
            self.main_viewer.window._qt_viewer.setFocus()
            return

        # extract the value
        data_coordinates = layer.world_to_data(event.position)
        x, y = np.round(data_coordinates).astype(int)
        # a click into nothing-ness
        if x < 0 or x >= self.data_width:
            return
        if y < 0 or y >= self.data_height:
            return
        # get the val
        if "Side" in layer.name:
            val = self.correction_data.get_label(self.current_frame + 1)[x, y]
            clicked_frame = self.current_frame + 1
        else:
            val = self.correction_data.get_label(self.current_frame)[x, y]
            clicked_frame = self.current_frame

        # we disconnect something
        if len(event.modifiers) == 1 and "Control" in event.modifiers:
            # disconnect the lineage
            try:
                self.correction_data.disconnect_lineage(selection=self.selection,
                                                        track_id=val,
                                                        frame_number=clicked_frame)
            except LineageOPException as e:
                show_info(f"{e}")
                return

            # update data
            self.change_frame(self.current_frame)

        # we join lineages
        elif len(event.modifiers) == 1 and "Shift" in event.modifiers:
            # join lineage
            try:
                self.correction_data.join_lineage(selection=self.selection,
                                                  track_id=val,
                                                  frame_number=clicked_frame)
            except LineageOPException as e:
                show_info(f"{e}")
                return

            # update data
            self.change_frame(self.current_frame)

        # we reconnect lineages
        elif len(event.modifiers) == 2 and "Shift" in event.modifiers and "Control" in event.modifiers:
            # join lineage
            try:
                self.correction_data.reconnect_lineage(selection=self.selection,
                                                       track_id=val,
                                                       frame_number=clicked_frame)
            except LineageOPException as e:
                show_info(f"{e}")
                return

            # update data
            self.change_frame(self.current_frame)

        # An actual click no modifiers
        else:
            # click on the same selection (unselect)
            if val == self.selection:
                self.update_selection(None)
            # normal update
            else:
                self.update_selection(val)

    def update_selection(self, selection=None):
        """
        Updates the selection to the new value
        :param selection: The new selection value
        """

        # selection attribute
        if selection is None or selection == 0:
            self.selection = None
        else:
            self.selection = selection

        # we update main
        self.main_select_layer.data = self.correction_data.get_selection(frame_number=self.current_frame,
                                                                         selection=self.selection,
                                                                         mark_orphans=self.mark_orphans,
                                                                         mark_dying=self.mark_dying)

        # update side
        self.side_select_layer.data = self.correction_data.get_selection(frame_number=self.current_frame + 1,
                                                                         selection=self.selection,
                                                                         mark_orphans=self.mark_orphans,
                                                                         mark_dying=self.mark_dying)

        # update the info box
        self.info_box.update_info(self.current_frame, self.selection)

    def update_settings(self, mark_orphans: bool, mark_dying: bool, sync_viewers: bool, rotate_view: bool):
        """
        Updates the general settings of the multiviewer
        :param mark_orphans: Whether orphans should be selected in the selection layer
        :param mark_dying: Whether dying cells should be selected in the selection layer
        :param sync_viewers: Whether to sync the viewers or not
        :param rotate_view: Whter to rot the viewer or not
        """

        # update the params
        self.mark_orphans = mark_orphans
        self.mark_dying = mark_dying
        self.sync_viewers = sync_viewers
        self.rotate_view = rotate_view

        # rotate the view if necessary
        if rotate_view:
            rotation = 90.0
        else:
            rotation = 0.0
        self.main_img_layer.rotate = rotation
        self.main_label_layer.rotate = rotation
        self.main_select_layer.rotate = rotation
        self.side_img_layer.rotate = rotation
        self.side_label_layer.rotate = rotation
        self.side_select_layer.rotate = rotation

        # if viewers are synced we need to do it manually here
        if self.sync_viewers:
            self.side_viewer.camera.zoom = self.main_viewer.camera.zoom
            self.side_viewer.camera.center = self.main_viewer.camera.center
            self.side_viewer.camera.angles = self.main_viewer.camera.angles

        # update the selection to update
        self.update_selection(self.selection)

        # set the focus
        self.main_viewer.window._qt_viewer.setFocus()

    def undo_ops(self, *args):
        """
        Performs the undo action of the data correction
        """

        self.correction_data.undo()
        self.change_frame(self.current_frame)

    def redo_ops(self, *args):
        """
        Performs the redo action of the data correction
        """

        self.correction_data.redo()
        self.change_frame(self.current_frame)


def main():
    # read in the data
    #path = Path("../../../Tests/tracking_tool/test_data/many_cells/pos1/GFP/track_output")
    path = Path("../../example_data/Pos58/TXRED/track_output")
    data_file = path.joinpath("tracking_delta.h5")
    csv_file = path.joinpath("track_output_delta.csv")
    #data_file = path.joinpath("tracking_bayesian.h5")
    #csv_file = path.joinpath("track_output_bayesian.csv")
    with CorrectionData(csv_file=csv_file, data_file=data_file) as correction_data:
        view = napari.Viewer()
        # create the multi view and make it central
        multi_view = MultipleViewerWidget(view, correction_data=correction_data)
        view.window._qt_window.setCentralWidget(multi_view)

        # set the size of the controls
        view.window._qt_viewer.controls.setMaximumWidth(350)

        # run napari
        napari.run()

if __name__ == "__main__":
    main()

