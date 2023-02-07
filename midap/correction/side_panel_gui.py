from typing import Optional, Callable

import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QPushButton,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class GenericBox(QWidget):
    """
    A Generic box with the right background color etx
    """
    def __init__(self):
        """
        Inits the widget
        """
        # proper init
        super().__init__()

        # style sheet
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("GenericBox { "
                           "background-color: #414851;"
                           "border-radius: 3%; }"
                           "QPushButton { "
                           "background-color : #5a626c; "
                           "border-radius: 3%; "
                           "font-family:Arial, sans-serif;"
                           "font-size:14px;"
                           "text-align:left;}"
                           "QPushButton:hover:!pressed:!disabled { "
                           "border: 1px solid black; }"
                           "QPushButton:disabled { "
                           "background-color: #262930;  }"
                           "QCheckBox {"
                           "font-family:Arial, sans-serif;"
                           "font-size:14px;"
                           "text-align:left; }"
                           "QCheckBox::indicator {"
                           "background-color: #262930; }"
                           )

        # table style for html elements
        self.table_style = f"""
        <style type="text/css">
        .tg  {{border-collapse:collapse;border-spacing:0;}}
        .tg td{{border:None;font-family:Arial, sans-serif;font-size:14px;
          overflow:hidden;padding:2px 2px;word-break:normal;}}
        .tg th{{border:None;;font-family:Arial, sans-serif;font-size:14px;
          font-weight:normal;overflow:hidden;padding:2px 2px;word-break:normal;}}
        .tg .tg-syad{{background-color:#414851;border:inherit;color:#ffffff;text-align:left;vertical-align:top;}}
        .tg .tg-tibk{{background-color:#414851;border:inherit;color:#ffffff;text-align:right;vertical-align:top;
        width:35px;}}
        </style>
        """


class SelectionBox(GenericBox):
    """
    A box displaying the selection box
    """

    def __init__(self, track_df: pd.DataFrame, change_frame_callback: Callable):
        """
        Inits the widget
        :param track_df: The tracking data frame indicating track IDs, cell divisions etc.
        :param change_frame_callback: A function that changes the frame of the viewer, take the number as input
        """
        # proper init
        super().__init__()

        # set attributes
        self.track_df = track_df
        self.change_frame_callback = change_frame_callback

        # add the subsections
        layout = QVBoxLayout()
        self.label = QLabel()
        # bitwise or to combine alignments
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setTextFormat(Qt.RichText)
        layout.addWidget(self.label)

        # buttons
        self.first_btn = QPushButton("Go to first occurrence: ")
        self.first_btn.clicked.connect(lambda: self.go_to_click(self.first_btn.text()))
        layout.addWidget(self.first_btn)
        self.last_btn = QPushButton("Go to last occurrence: ")
        self.last_btn.clicked.connect(lambda: self.go_to_click(self.last_btn.text()))
        layout.addWidget(self.last_btn)
        self.split_btn = QPushButton("Go to split: ")
        self.split_btn.clicked.connect(lambda: self.go_to_click(self.split_btn.text()))
        layout.addWidget(self.split_btn)

        # finalize
        self.setLayout(layout)
        self.setFixedHeight(180)

    def update_info(self, current_frame: int, selection: Optional[int]):
        """
        Updates the info box for a new frame and selection
        :param current_frame: The index of the current frame (left viewer)
        :param selection: The label (track ID) of the current selection, can be None
        """

        # default is not in frame
        in_frame = "No"

        # selection number
        if selection is None:
            selection = "N/A"
        # if the selection is in the frame
        elif selection in self.track_df[self.track_df["frame"] == current_frame]["trackID"].values or \
                selection in self.track_df[self.track_df["frame"] == current_frame + 1]["trackID"].values:
            in_frame = "Yes"

        label = f"""
        <h2><u> Selection </u></h2>
        {self.table_style}
        <table class="tg">
        <tbody>
          <tr>
            <td class="tg-syad">Current Selection:</td>
            <td class="tg-tibk">{selection}</td>
          </tr>
          <tr>
            <td class="tg-syad">In frame: </td>
            <td class="tg-tibk">{in_frame}</td>
          </tr>
        </tbody>
        </table>
        """

        # update label
        self.label.setText(label)

        # update the buttons
        if selection == "N/A":
            self.first_btn.setText(f"Go to first occurrence: {selection}")
            self.first_btn.setDisabled(True)
            self.last_btn.setText(f"Go to last occurrence: {selection}")
            self.last_btn.setDisabled(True)
            self.split_btn.setText(f"Go to split: {selection}")
            self.split_btn.setDisabled(True)
        else:
            first_occurrence = int(self.track_df.iloc[(self.track_df["trackID"] == selection).argmax()]["first_frame"])
            self.first_btn.setText(f"Go to first occurrence: {first_occurrence}")
            self.first_btn.setDisabled(False)
            last_occurrence = int(self.track_df.iloc[(self.track_df["trackID"] == selection).argmax()]["last_frame"])
            self.last_btn.setText(f"Go to last occurrence: {last_occurrence}")
            self.last_btn.setDisabled(False)

            # the splitting
            if np.any(self.track_df[self.track_df["trackID"] == selection]["split"] == 1):
                current_selection = self.track_df[self.track_df["trackID"] == selection]
                split_frame = int(current_selection.iloc[(current_selection["split"] == 1).argmax()]["frame"])
                self.split_btn.setText(f"Go to split: {split_frame}")
                self.split_btn.setDisabled(False)
            else:
                self.split_btn.setText(f"Go to split: N/A")
                self.split_btn.setDisabled(True)


    def go_to_click(self, button_text):
        """
        This function extract the target frame from the button text and moves there
        :param button_text: The text on the button
        """

        frame_number = int(button_text.split(":")[-1])
        self.change_frame_callback(frame_number)

        # clear the focus of the buttons
        for button in [self.first_btn, self.last_btn, self.split_btn]:
            if button.hasFocus():
                button.clearFocus()


class FrameInfo(GenericBox):
    """
    A class to display the Frame info
    """

    def __init__(self, labels: np.ndarray, track_df: pd.DataFrame):
        """
        Inits the widget
        :param labels: An array of labels corresponding to the images
        :param track_df: The tracking data frame indicating track IDs, cell divisions etc.
        """
        # proper init
        super().__init__()

        # set attributes
        self.labels = labels
        self.track_df = track_df

        # set necessary attributes
        layout = QVBoxLayout()
        self.label = QLabel()
        # bitwise or to combine alignments
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setTextFormat(Qt.RichText)
        layout.addWidget(self.label)

        # finalize
        self.setLayout(layout)
        self.setFixedHeight(170)

    def update_info(self, current_frame: int):
        """
        Updates the info box for a new frame and selection
        :param current_frame: The index of the current frame (left viewer)
        """

        # get infos
        # TODO: Currently splits are considered dying and the kids are orphans
        left_frame = current_frame
        right_frame = current_frame + 1
        left_n_cells = np.sum(self.track_df["frame"] == left_frame)
        right_n_cells = np.sum(self.track_df["frame"] == right_frame)
        if left_frame == 0:
            left_orphans = "N/A"
        else:
            left_orphans = sum((self.track_df["first_frame"] == left_frame) & (self.track_df["frame"] == left_frame))
        right_orphans = sum((self.track_df["first_frame"] == right_frame) & (self.track_df["frame"] == right_frame))
        left_dying = sum((self.track_df["last_frame"] == left_frame) & (self.track_df["frame"] == left_frame))
        if right_frame == len(self.labels) - 1:
            right_dying = "N/A"
        else:
            right_dying = sum((self.track_df["last_frame"] == right_frame) & (self.track_df["frame"] == right_frame))

        # to table
        frame_info = f"""
        <h2><u> Frames </u></h2>
        {self.table_style}
        <table class="tg">
        <thead>
          <tr>
            <th class="tg-syad"></th>
            <th class="tg-tibk"><b>Left</b></th>
            <th class="tg-tibk"><b>Right</b></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="tg-syad"># Frame:</td>
            <td class="tg-tibk">{left_frame}</td>
            <td class="tg-tibk">{right_frame}</td>
          </tr>
          <tr>
            <td class="tg-syad"># Cells:</td>
            <td class="tg-tibk">{left_n_cells}</td>
            <td class="tg-tibk">{right_n_cells}</td>
          </tr>
          <tr>
            <td class="tg-syad"># Orphans:</td>
            <td class="tg-tibk">{left_orphans}</td>
            <td class="tg-tibk">{right_orphans}</td>
          </tr>
          <tr>
            <td class="tg-syad"># Dying:</td>
            <td class="tg-tibk">{left_dying}</td>
            <td class="tg-tibk">{right_dying}</td>
          </tr>
        </tbody>
        </table>
        """

        # update label text
        self.label.setText(frame_info)


class GeneralBox(GenericBox):
    """
    A class to display the help messages
    """

    def __init__(self, update_modifier_callback: Callable):
        """
        Inits the widget
        :param update_modifier_callback: A function that changes the general settings for the multiviewer
        """
        # proper init
        super().__init__()

        # set attributes
        self.update_modifier_callback = update_modifier_callback

        # The title
        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setTextFormat(Qt.RichText)
        # bitwise or to combine alignments
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setText("""
        <h2><u> General </u></h2>
        """)
        layout.addWidget(self.label)

        # check boxes
        self.mark_orphans = QCheckBox("Mark orphans")
        self.mark_orphans.setChecked(False)
        self.mark_orphans.toggled.connect(self.box_clicked)
        layout.addWidget(self.mark_orphans)
        self.mark_dying = QCheckBox("Mark dying")
        self.mark_dying.setChecked(False)
        self.mark_dying.toggled.connect(self.box_clicked)
        layout.addWidget(self.mark_dying)
        self.sync_viewer = QCheckBox("Sync viewer")
        self.sync_viewer.setChecked(True)
        self.sync_viewer.toggled.connect(self.box_clicked)
        layout.addWidget(self.sync_viewer)

        # finalize
        self.setLayout(layout)
        self.setFixedHeight(130)

    def box_clicked(self):
        """
        The call back when a checkbox is clicked
        """

        self.update_modifier_callback(mark_orphans=self.mark_orphans.isChecked(), mark_dying=self.mark_dying.isChecked(),
                                      sync_viewers=self.sync_viewer.isChecked())


class HelpBox(GenericBox):
    """
    A class to display the help messages
    """

    def __init__(self):
        """
        Inits the widget
        """
        # proper init
        super().__init__()

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setTextFormat(Qt.RichText)
        # bitwise or to combine alignments
        self.label.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.label.setText(f"""
        <h2><u> Controls </u></h2>
        {self.table_style}
        <table class="tg">
        <tbody>
          <tr>
            <td class="tg-syad">Click: </td>
            <td class="tg-tibk">Select cell</td>
          </tr>
          <tr>
            <td class="tg-syad">Arrow keys:</td>
            <td class="tg-tibk">Change frame</td>
          </tr>
        </tbody>
        </table>
        """)
        layout.addWidget(self.label)

        # finalize
        self.setLayout(layout)


class InfoBox(QWidget):
    """
    Infobox with all the frames etc.
    """
    def __init__(self, viewer: napari.Viewer, labels: np.ndarray, track_df: pd.DataFrame,
                 change_frame_callback: Callable, update_modifier_callback: Callable):
        """
        Inits the widget
        :param viewer: The napari viewer
        :param labels: An array of labels corresponding to the images
        :param track_df: The tracking data frame indicating track IDs, cell divisions etc.
        :param change_frame_callback: A function that changes the frame of the viewer, take the number as input
        :param update_modifier_callback: A function that changes the general settings for the multiviewer
        """
        # proper init
        super().__init__()

        # set attributes
        self.viewer = viewer
        self.labels = labels
        self.track_df = track_df

        # add the subsections
        layout = QVBoxLayout()

        # define some constants
        width = 200

        # frame stuff
        self.frame_info = FrameInfo(labels=labels, track_df=track_df)
        self.frame_info.setFixedWidth(width)
        layout.addWidget(self.frame_info)

        # Selection stuff
        self.selection_box = SelectionBox(track_df=track_df, change_frame_callback=change_frame_callback)
        self.selection_box.setFixedWidth(width)
        layout.addWidget(self.selection_box)

        # General Stuff
        self.general_box = GeneralBox(update_modifier_callback=update_modifier_callback)
        self.general_box.setFixedWidth(width)
        layout.addWidget(self.general_box)

        # Help
        self.help_box = HelpBox()
        self.help_box.setFixedWidth(width)
        layout.addWidget(self.help_box)

        # finalize
        self.setLayout(layout)
        self.setFixedWidth(width + 15)

    def update_info(self, current_frame: int, selection: Optional[int]):
        """
        Updates the info box for a new frame and selection
        :param current_frame: The index of the current frame (left viewer)
        :param selection: The label (track ID) of the current selection, can be None
        """

        # update frame
        self.frame_info.update_info(current_frame)
        self.selection_box.update_info(current_frame, selection)

        # set the focus to the viewer in case something else was clicked
        self.viewer.window._qt_viewer.setFocus()
