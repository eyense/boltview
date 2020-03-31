
#include <QApplication>
#include <QtWidgets/QWidget>
#include <iostream>
#include "mainwindow.h"
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow widget;

    widget.show();

    return a.exec();
}
