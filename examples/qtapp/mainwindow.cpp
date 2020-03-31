#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QLabel>
#include <QTimer>
#include <QDialog>
#include "anisotropic.h"
#include "fileopen.h"
#include <iostream>
#include <QDialogButtonBox>
#include <QDebug>
#include "diffusion_with_const_kernel.h"
#include <QValidator>
#include "thresholding.h"
#include "edge_detection.h"
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::MainWindow)
{
    _ui->setupUi(this);
    _ui->anisotropic_push->setVisible(false);
    _ui->sample_push->setVisible(false);
    _ui->iterations->setVisible(false);
    _ui->group_box->setVisible(false);
    _ui->iterat_label->setVisible(false);
    _ui->close_box->setVisible(false);
    _ui->m_image->setScaledContents(true);
    _ui->out_im->setScaledContents(true);
    _ui->threshold_lyt->setVisible(false);
    _ui->edge_det_btn->setVisible(false);
    _m_second_window = new Phasecorrelation();
}


MainWindow::~MainWindow() {
    delete _ui;
}

void MainWindow::OpenImage(QLabel* pix, QImage &im) {
    pix->setAlignment(Qt::AlignCenter);
    pix->setPixmap( QPixmap::fromImage(im));
    pix->setGeometry(0, 0, 640, 620);
    pix->adjustSize();
    pix->show();
}
void MainWindow::on_anisotropic_push_clicked() {
    QImage out_image(_my_image.size(), _my_image.format());
    AnisotropicDiffusion(_my_image, out_image, _ui->iteration_count->value(), _ui->coeficient->value(), _ui->speed->value());
    OpenImage(_ui->out_im, out_image);
}


void MainWindow::on_sample_push_clicked() {
    QImage out_image(_my_image.size(), _my_image.format());
    DiffusionWithConstKernel(_my_image, out_image, _ui->iterations->value());
    OpenImage(_ui->out_im, out_image);
}


void MainWindow::on_Close_button_box_accepted()
{
    this->close();
}


void MainWindow::on_Anisotropic_button_clicked() {
    _ui->sample_push->setVisible(false);
    _ui->iterat_label->setVisible(false);
    _ui->iterations->setVisible(false);
    _ui->anisotropic_push->setVisible(true);
    _ui->group_box->setVisible(true);
}


void MainWindow::on_Sample_button_clicked() {
    _ui->anisotropic_push->setVisible(false);
    _ui->group_box->setVisible(false);
    _ui->sample_push->setVisible(true);
    _ui->iterations->setVisible(true);
    _ui->iterat_label->setVisible(true);
}


void MainWindow::on_openAct_triggered() {
    QString filename;
    QFileDialogTester test;
    test.openFile(filename);
    _my_image.load(filename);
    _scale_factor = 1.0;
    OpenImage(_ui->m_image, _my_image);
    _ui->threshold_lyt->setVisible(true);
    _ui->edge_det_btn->setVisible(true);
}


void MainWindow::on_ZoomInAct_triggered() {
    scaleImage(1.25);
}


void MainWindow::on_ZoomOutAct_triggered() {
    scaleImage(0.8);
}


void MainWindow::scaleImage(double factor) {
    _scale_factor *= factor;
    if (_ui->m_image->pixmap() != nullptr) {
        _ui->m_image->resize(_scale_factor * _ui->m_image->pixmap()->size());
    }
    if (_ui->out_im->pixmap() != nullptr) {
        _ui->out_im->resize(_scale_factor * _ui->out_im->pixmap()->size());
    }
}


void MainWindow::on_CloseApplocationAct_triggered() {
    _ui->close_box->setVisible(true);
}


void MainWindow::on_Close_button_box_rejected() {
    _ui->close_box->setVisible(false);
}


void MainWindow::on_SaveAct_triggered() {
    bool b = _ui->out_im->pixmap()->save("test.png");
}


void MainWindow::on_threshold_but_clicked() {
    Thresholding(_my_image, _ui->threshold_spin->value());
    OpenImage(_ui->m_image, _my_image);
}


void MainWindow::on_edge_det_btn_clicked() {
    QImage out_image(_my_image.size(), _my_image.format());
    EdgeDetection(_my_image, out_image);
    OpenImage(_ui->out_im, out_image);
}



void MainWindow::on_pushButton_clicked()
{
    _m_second_window->show();
}
