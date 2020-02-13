import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import numpy as np
import pandas as pd
from datetime import datetime
import time
from spring import pre_processing, SPRING, get_trait, refine_trait


class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value) for value in values]


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(400, 150, 1000, 750)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        MenuWidget(self)
        self.plotArea = PlotWidget(self.centralWidget)
        self.dock_field = DockWidget('Field', self, Qt.RightDockWidgetArea)
        self.dock_ref = DockWidget('Ref.', self, Qt.RightDockWidgetArea)
        self.dock_query = DockWidget('Query', self, Qt.RightDockWidgetArea)
        self.dock_output = DockWidget('Output', self, Qt.BottomDockWidgetArea)
        self.dock_output.setTitleBarWidget(QtGui.QWidget(None))

        self.init_dock_field()
        self.init_dock_ref()
        self.init_dock_query()
        self.init_dock_output()

    def open_csv(self):
        _filter = "csv(*.csv *.txt)"
        self.path, _ = QFileDialog.getOpenFileName(filter=_filter)
        if self.path == '':
            return

        # def dateparse(dates): return pd.datetime.strptime(
        #     dates, "%d-%B-%y %H:%M:%S")
        def dateparse(dates): return pd.datetime.strptime(dates, "%d/%m/%Y %H:%M")
        self.df = pd.read_csv(self.path, index_col=[0], delimiter='\t', parse_dates=[
            0], date_parser=dateparse)
        self.df.interpolate(inplace=True)
        self.list_field.clear()
        self.list_field.addItems(list(self.df))
        x = pd.DatetimeIndex.to_pydatetime(self.df.index)
        self.x = np.vectorize(lambda x: x.timestamp())(x)  # numpy.ndarray

### FIELD ###
    def init_dock_field(self):
        self.list_field = QListWidget(self.dock_field)
        self.dock_field.setWidget(self.list_field)
        self.list_field.itemDoubleClicked.connect(self.plot_selected)

    def plot_selected(self, instance):
        self.col_selected = self.list_field.selectedIndexes()[0].row()
        self.y = np.array(self.df.iloc[:, self.col_selected])
        self.plotArea.plot(self.x, self.y)

### REF ###
    def init_dock_ref(self):
        self.list_ref = QListWidget(self.dock_ref)
        self.btn_set_ref = QPushButton('Set Reference', self.dock_ref)
        self.btn_confirm_ref = QPushButton('Confirm', self.dock_ref)
        self.container = QWidget()
        layout = QGridLayout()
        self.container.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(layout)
        layout.addWidget(self.list_ref, 0, 0, 1, 2)
        layout.addWidget(self.btn_set_ref, 1, 0)
        layout.addWidget(self.btn_confirm_ref, 1, 1)
        self.dock_ref.setWidget(self.container)
        self.btn_set_ref.clicked.connect(self.set_ref)
        self.btn_confirm_ref.clicked.connect(self.confirm_ref)

    def set_ref(self):
        self.plotArea.add_ref_region(self.x, self.y)

    def confirm_ref(self):
        self.ssq_store, self.a_min_store = self.plotArea.get_ssq(
            self.x, self.df, self.col_selected)
        # print(self.ssq_store)
        self.ssq_store_list = []
        for ssq in self.ssq_store:
            t_start = datetime.fromtimestamp(
                self.x[ssq[0]]).replace(second=0, microsecond=0)
            t_end = datetime.fromtimestamp(
                self.x[ssq[1]]).replace(second=0, microsecond=0)
            self.ssq_store_list.append((t_start, t_end, ssq[2]))
        self.list_output_1.addItems([str(ssq) for ssq in self.ssq_store_list])
        # self.plotArea.clearRegion()

### QUERY ###
    def init_dock_query(self):
        self.list_query = QListWidget(self.dock_query)
        self.btn_select_query = QPushButton('Select query', self.dock_query)
        self.btn_confirm_query = QPushButton('Confirm', self.dock_query)
        self.container = QWidget()
        layout = QGridLayout()
        self.container.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(layout)
        layout.addWidget(self.list_query, 0, 0, 1, 2)
        layout.addWidget(self.btn_select_query, 1, 0)
        layout.addWidget(self.btn_confirm_query, 1, 1)
        self.dock_query.setWidget(self.container)

        self.btn_select_query.clicked.connect(self.select_query)
        self.btn_confirm_query.clicked.connect(self.confirm_query)

    def select_query(self):
        self.plotArea.add_query_line(self.x, self.y)

    def confirm_query(self):
        self.matched_queries = self.plotArea.get_matched_queries(
            self.x, self.y, 50, 10)
        self.matched_queries_list = []
        for idx in self.matched_queries:
            t = datetime.fromtimestamp(self.x[idx]).replace(
                second=0, microsecond=0)
            self.matched_queries_list.append((t, self.y[idx]))
        self.list_output_2.addItems([str(mq)
                                     for mq in self.matched_queries_list])

    def to_csv(self):
        try:
            self.matched_queries_list
            df_1 = pd.DataFrame(self.ssq_store_list, columns=[
                                'subsequence_start', 'subsequence_end', 'error'])
            df_2 = pd.DataFrame(self.matched_queries_list, columns=[
                                'matched_query_time', 'matched_query'])
            df_out = pd.concat([df_1, df_2], axis=1)
            fileName, _ = QFileDialog.getSaveFileName(
                self, "QFileDialog.getSaveFileName()", "", "csv(*.csv)")
            df_out.to_csv(fileName)

        except AttributeError:
            # popup
            pass

### OUTPUT ###
    def init_dock_output(self):
        self.tabWidget = QTabWidget(self.dock_output)
        self.tabWidget.setStyleSheet("QTabWidget::tab-bar {width: 999999px}")
        self.dock_output.setWidget(self.tabWidget)

        # tab_1
        self.tab_1 = QWidget()
        self.tabWidget.addTab(self.tab_1, "Matched Subsequences")
        self.list_output_1 = QListWidget(self.tab_1)
        self.cb_lr = QCheckBox("Set Linear Region Visible")
        self.cb_lr.setChecked(True)
        self.cb_lr.stateChanged.connect(self.cb_lr_state)
        self.hbox_1 = QHBoxLayout()
        self.hbox_1.addWidget(self.cb_lr)
        self.vbox_1 = QVBoxLayout()
        self.vbox_1.addWidget(self.list_output_1)
        self.vbox_1.addLayout(self.hbox_1)
        self.tab_1.setLayout(self.vbox_1)

        # tab_2
        self.tab_2 = QWidget()
        self.tabWidget.addTab(self.tab_2, "Matched Queries")
        self.list_output_2 = QListWidget(self.tab_2)
        self.cb_ln = QCheckBox("Set Line Visible")
        self.cb_ln.setChecked(True)
        self.cb_ln.stateChanged.connect(self.cb_ln_state)
        self.hbox_2 = QHBoxLayout()
        self.hbox_2.addWidget(self.cb_ln)
        self.vbox_2 = QVBoxLayout()
        self.vbox_2.addWidget(self.list_output_2)
        self.vbox_2.addLayout(self.hbox_2)
        self.tab_2.setLayout(self.vbox_2)

    def cb_ln_state(self):
        if self.cb_ln.isChecked() == True:
            self.plotArea.addAllLine(self.x, self.matched_queries)
        else:
            self.plotArea.clearLines()

    def cb_lr_state(self):
        if self.cb_lr.isChecked() == True:
            self.plotArea.addAllRegion(self.x, self.ssq_store)
        else:
            self.plotArea.clearRegion()


class MenuWidget():
    def __init__(self, parent):
        menubar = parent.menuBar()
        fileMenu = menubar.addMenu('File')
        editMenu = menubar.addMenu('Edit')

        openAction = QAction('Open', parent)
        openAction.triggered.connect(parent.open_csv)
        fileMenu.addAction(openAction)

        exportAction = QAction('Export', parent)
        exportAction.triggered.connect(parent.to_csv)
        fileMenu.addAction(exportAction)

        closeAction = QAction('Exit', parent)
        closeAction.triggered.connect(parent.close)
        fileMenu.addAction(closeAction)


class DockWidget(QDockWidget):
    def __init__(self, title, parent, position):
        super(QDockWidget, self).__init__(title, parent)
        self.setFeatures(QtGui.QDockWidget.DockWidgetMovable)
        parent.addDockWidget(position, self)
        self.setStyleSheet("QDockWidget {border: 0px}")


class PlotWidget():
    def __init__(self, parent):
        self.parent = parent
        self.display = pg.PlotWidget(parent, axisItems={
                                     'bottom': TimeAxisItem(orientation='bottom')})
        self.display.setXRange(1577836800, 1577836800)

        self.viewbox = self.display.getViewBox()
        grid = QGridLayout(parent)
        grid.setSpacing(10)
        grid.addWidget(self.display, 0, 0)

    def plot(self, x, y):
        self.display.clear()
        self.curve = self.display.plot(x, y, symbol='o', symbolSize=3)
        self.clearItems()
        self.display.setXRange(x[-1000], x[-1])
        self.display.setLimits(xMin=x[0] - 86400, xMax=x[-1] + 86400)
        self.display.setMouseEnabled(y=False)
        # self.display.enableAutoRange(enable=True)
        self.label = pg.TextItem(color=(255, 255, 255), anchor=(0, 0))
        self.label.setParentItem(self.viewbox)
        self.label.setPos(10, 0)
        self.curve.scene().sigMouseMoved.connect(self.on_mouse_move)

    def clearLines(self):
        for widget in self.display.allChildItems():
            if isinstance(widget, pg.InfiniteLine):
                self.display.removeItem(widget)

    def clearItems(self):
        for widget in self.display.allChildItems():
            if isinstance(widget, (pg.InfiniteLine, pg.LinearRegionItem, pg.TextItem)):
                self.display.removeItem(widget)
        try:
            self.viewbox.removeItem(self.label)
        except AttributeError:
            pass

    def clearRegion(self):
        for widget in self.display.allChildItems():
            if isinstance(widget, pg.LinearRegionItem):
                self.display.removeItem(widget)

    def add_ref_region(self, x, y):
        try:
            self.region
        except AttributeError:
            self.addRegion(x, -1000, -500, True)
        self.get_ref(x, y)
        self.region.sigRegionChanged.connect(lambda: self.get_ref(x, y))

    def addRegion(self, x, region_start_pos, region_end_pos, bool_movable):
        self.region = pg.LinearRegionItem(
            [x[region_start_pos], x[region_end_pos]], movable=bool_movable)
        self.display.addItem(self.region)

    def get_ref(self, x, y):
        ref_s, ref_e = [self.fromtimestamp(ref)
                        for ref in self.region.getRegion()]
        y_s, y_e = [y[self.find_nearest(x, ref)]
                    for ref in self.region.getRegion()]
        s = "y = {:.2f}, t = {}".format(y_s, ref_s)
        e = "y = {:.2f}, t = {}".format(y_e, ref_s)
        self.parent.parent().list_ref.clear()
        self.parent.parent().list_ref.addItems([s, e])

    def addAllRegion(self, x, ssq_store):
        if not all(isinstance(widget, pg.LinearRegionItem) for widget in self.display.allChildItems()):
            for ssq in self.ssq_store:
                self.addRegion(x, ssq[0], ssq[1], False)

    def addAllLine(self, x, ssq_t_store):
        if not all(isinstance(widget, pg.InfiniteLine) for widget in self.display.allChildItems()):
            for ssq_t in self.ssq_t_store:
                self.addLine(x, int(ssq_t), False)

    def get_ssq(self, x, df, col):
        ref_s, ref_e = [self.fromtimestamp(ref)
                        for ref in self.region.getRegion()]
        spring_x, spring_y = pre_processing(df, col, ref_s, ref_e)
        self.ssq_store, self.a_min_store = SPRING(spring_x, spring_y, 5000)
        self.clearRegion()
        self.addAllRegion(x, self.ssq_store)
        return self.ssq_store, self.a_min_store

    def add_query_line(self, x, y):
        try:
            self.line
        except AttributeError:
            self.addLine(x, -500, True)
        self.get_query(x, y)
        self.line.sigPositionChanged.connect(lambda: self.get_query(x, y))

    def get_query(self, x, y):
        query = self.fromtimestamp(self.line.value())
        y_query = y[self.find_nearest(x, self.line.value())]
        self.parent.parent().list_query.clear()
        self.parent.parent().list_query.addItem(
            "y = {:.2f}, t = {}".format(y_query, query))

    def addLine(self, x, x_pos, bool_movable):
        self.line = pg.InfiniteLine(pos=x[x_pos], movable=bool_movable)
        self.display.addItem((self.line))

    def get_matched_queries(self, x, y, testrange, testsize):
        query = self.find_nearest(x, self.line.value())
        self.ssq_t_store = get_trait(
            query, self.ssq_store, self.a_min_store)
        self.ssq_t_store = refine_trait(
            query, self.ssq_t_store, y, testrange, testsize)
        for ssq_t in self.ssq_t_store:
            self.addLine(x, int(ssq_t), False)
        # print(self.ssq_t_store)
        return self.ssq_t_store

    def on_mouse_move(self, point):
        p = self.display.plotItem.vb.mapSceneToView(point)
        self.label.setText("t = {},  y = {:.2f}".format(
            self.fromtimestamp(p.x()), p.y()))

    def fromtimestamp(self, timestamp):
        return datetime.fromtimestamp(timestamp).replace(second=0, microsecond=0)

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
