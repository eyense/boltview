#include <QApplication>
#include <QFileDialog>
#include <QDebug>
#include "fileopen.h"
void QFileDialogTester::openFile(QString & filename) {
    filename = QFileDialog::getOpenFileName(
          this,
          "Open Document",
          QDir::currentPath(),
          "All files (*.*) ;; Document files (*.doc *.rtf);; PNG files (*.png)");

    if( !filename.isNull() )
    {
      qDebug() << "selected file path : " << filename.toUtf8();
    }
}



