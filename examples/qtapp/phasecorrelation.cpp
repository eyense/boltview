#include "phasecorrelation.h"
#include "ui_phasecorrelation.h"
#include "fileopen.h"
#include "phase_correlation_help.h"
#include <QString>
Phasecorrelation::Phasecorrelation(QWidget *parent) :
    QWidget(parent),
    _ui(new Ui::Phasecorrelation)
{
     _ui->setupUi(this);
}

Phasecorrelation::~Phasecorrelation()
{
}

void Phasecorrelation::OpenImage(QLabel* pix, QImage &im) {
    pix->setAlignment(Qt::AlignCenter);
    pix->setPixmap( QPixmap::fromImage(im));
    pix->setGeometry(0, 0, 640, 620);
    pix->adjustSize();
    pix->show();
}
void Phasecorrelation::on_pushButton_clicked()
{
    QString filename;
    QFileDialogTester test;
    test.openFile(filename);
    _image_in.load(filename);
    QString filename_phase;
    QFileDialogTester test_phase;
    test_phase.openFile(filename_phase);
    _image_phase.load(filename_phase);
    OpenImage(_ui->in_lbl, _image_in);
    OpenImage(_ui->cor_lbl, _image_phase);
}


void Phasecorrelation::on_pushButton_2_clicked()
{
    QImage out_image(_ui->size_x->value(), _ui->size_y->value(), _image_in.format());
    InverseCorrelation(_image_in, _image_phase, out_image, _ui->coord_x->value(), _ui->coord_y->value(), _ui->size_x->value(), _ui->size_y->value());
    OpenImage(_ui->out_lbl, out_image);
}
