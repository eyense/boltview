#ifndef PhasecORRELATION_H
#define PhasecORRELATION_H

#include <QWidget>
#include <QImage>
#include <QLabel>

namespace Ui {
class Phasecorrelation;
}

class Phasecorrelation : public QWidget
{
    Q_OBJECT

public:
    explicit Phasecorrelation(QWidget *parent = 0);
    void OpenImage(QLabel* pix, QImage &im);
    ~Phasecorrelation();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::Phasecorrelation *_ui;
    QImage _image_in;
    QImage _image_phase;
};

#endif // PhasecORRELATION_H
