import sys
# from PySide6.QtWidgets import QApplication
# from ui.windows.main_window import MainWindow

from cli import main as cli_main

if __name__ == "__main__":
    # print("Starting QApplication...")
    # app = QApplication(sys.argv)
    # print("QApplication initialized.")
    # main_window = MainWindow()
    # print("MainWindow created.")
    # main_window.show()
    # print("MainWindow shown. Entering event loop...")
    # sys.exit(app.exec_())
    # print("Application exited.")
    cli_main()
