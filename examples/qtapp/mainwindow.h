#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include <QScrollBar>
#include <QScrollArea>
#include "phasecorrelation.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
     void OpenImage(QLabel* pix, QImage &im);
private slots:

    void on_anisotropic_push_clicked();

    void on_sample_push_clicked();

    void on_Anisotropic_button_clicked();

    void on_Sample_button_clicked();

    void on_openAct_triggered();

    void on_ZoomInAct_triggered();

    void scaleImage(double factor);

    void on_ZoomOutAct_triggered();

    void on_CloseApplocationAct_triggered();

    void on_Close_button_box_accepted();

    void on_Close_button_box_rejected();

    void on_SaveAct_triggered();

    void on_threshold_but_clicked();

    void on_edge_det_btn_clicked();


    void on_pushButton_clicked();

private:
    Ui::MainWindow *_ui;
    QImage _my_image;
    double _scale_factor;
    Phasecorrelation *_m_second_window;

};

#endif // MAINWINDOW_H
