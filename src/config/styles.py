"""
Qt Stylesheet definitions.
"""

STYLE = """
QMainWindow, QWidget {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

QLabel { color: #E6EDF3; }
QLabel[class="title"] { font-size: 20px; font-weight: bold; color: #58A6FF; }
QLabel[class="section"] { font-size: 14px; font-weight: bold; color: #7EE787; }
QLabel[class="info"] { font-size: 11px; color: #8B949E; }

QFrame[class="panel"] {
    background-color: #161B22;
    border: 2px solid #30363D;
    border-radius: 10px;
}

QPushButton {
    background-color: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}
QPushButton:hover { background-color: #30363D; border-color: #58A6FF; }
QPushButton[class="primary"] { background-color: #238636; border: none; }
QPushButton[class="primary"]:hover { background-color: #2EA043; }
QPushButton[class="danger"] { background-color: #DA3633; border: none; }
QPushButton:checked { background-color: #238636; border: 2px solid #7EE787; }

QRadioButton {
    color: #E6EDF3;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #30363D;
    background: #21262D;
}
QRadioButton::indicator:checked {
    background: #238636;
    border-color: #7EE787;
}

QComboBox {
    background-color: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 8px;
    min-width: 120px;
}
QComboBox QAbstractItemView {
    background-color: #161B22;
    selection-background-color: #238636;
}

QSlider::groove:horizontal { background: #30363D; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #58A6FF; width: 16px; margin: -5px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #238636; border-radius: 3px; }

QProgressBar {
    background-color: #21262D;
    border: none;
    border-radius: 5px;
    height: 10px;
}
QProgressBar::chunk { background-color: #238636; border-radius: 5px; }

QGroupBox {
    font-weight: bold;
    color: #8B949E;
    border: 1px solid #30363D;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }

QSplitter::handle { background: #30363D; width: 3px; }
QSplitter::handle:hover { background: #58A6FF; }
"""
