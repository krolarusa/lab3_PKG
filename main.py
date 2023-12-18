import cv2  # Подключение библиотеки OpenCV для обработки изображений
import numpy as np  # Подключение библиотеки NumPy для работы с массивами
import matplotlib as mp
from matplotlib import pyplot as plt  # Подключение библиотеки Matplotlib для построения графиков
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ImageProcessing():
    @staticmethod
    def average_blur(infilepath):
        img = cv2.imread(infilepath)  # Чтение изображения
        # Применение усредняющего размытия
        blur = cv2.blur(img, (5, 5))
        cv2.imshow("Average blur", blur)  # Отображение изображения с размытием

    @staticmethod
    def gaussian_blur(infilepath):
        img = cv2.imread(infilepath)  # Чтение изображения
        # Применение гауссова размытия
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("Gaussian blur", blur)  # Отображение изображения с размытием

    @staticmethod
    def median_blur(infilepath):
        img = cv2.imread(infilepath)  # Чтение изображения
        # Применение медианного размытия
        median = cv2.medianBlur(img, 5)
        cv2.imshow("Median blur", median)  # Отображение изображения с размытием

    @staticmethod
    def equalization_hist_hsv(infilepath):
        image = cv2.imread(infilepath)  # Чтение изображения
        H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))  # Разделение каналов HSV
        eq_V = cv2.equalizeHist(V)  # Применение гистограммного выравнивания к каналу V
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]),
                                cv2.COLOR_HSV2BGR)  # Объединение каналов и обратное преобразование в BGR
        return eq_image

    @staticmethod
    def equalization_hist_rgb(infilepath):
        image = cv2.imread(infilepath)  # Чтение изображения
        channels = cv2.split(image)  # Разделение каналов RGB
        eq_channels = []
        for ch, color in zip(channels, ['B', 'G', 'R']):
            eq_channels.append(cv2.equalizeHist(ch))  # Применение гистограммного выравнивания к каждому каналу
        eq_image = cv2.merge(eq_channels)  # Объединение каналов
        return eq_image

    @staticmethod
    def linear_contrasting_grayscale(infilepath):
        img = cv2.imread(infilepath, cv2.IMREAD_GRAYSCALE)  # Чтение изображения в оттенках серого
        norm_img1 = cv2.normalize(
            img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Выравнивание контраста
        # Масштабирование до uint8
        norm_img1 = (255 * norm_img1).astype(np.uint8)
        # Отображение входного и обоих выходных изображений
        return norm_img1

    @staticmethod
    def equalization_hist_grayscale(infilepath):
        img = cv2.imread(infilepath, 0)  # Чтение изображения в оттенках серого
        equ = cv2.equalizeHist(img)  # Применение гистограммного выравнивания
        return equ


class main_window(QWidget):

    # Метод для блокировки / разблокировки кнопок
    def blockButtons(self, is_blocked):
        self.btnBlur.blockSignals(is_blocked)
        self.btnChoose.blockSignals(is_blocked)
        self.btnColorful.blockSignals(is_blocked)
        self.btnGrayscale.blockSignals(is_blocked)

    # Метод для обработки цветного изображения
    def colorful(self):
        if self.fname == "":
            self.showDialog(1)  # Показ предупреждения, если изображение не выбрано
        else:
            # Применение методов гистограммного выравнивания к изображению в RGB и HSV
            rgb = ImageProcessing.equalization_hist_rgb(self.fname)
            hsv = ImageProcessing.equalization_hist_hsv(self.fname)

            # Построение и отображение гистограмм по каналам RGB с использованием Matplotlib
            plt.subplot(1, 2, 1)
            channels = ('b', 'g', 'r')
            plt.title("Histogram equalization by RGB")
            for i, color in enumerate(channels):
                histogram = cv2.calcHist([rgb], [i], None, [256], [0, 256])
                plt.plot(histogram, color=color)
            plt.xlim([0, 256])

            # Построение и отображение гистограмм по каналам HSV с использованием Matplotlib
            plt.subplot(1, 2, 2)
            plt.title("Histogram equalization by HSV")
            for i, color in enumerate(channels):
                histogram = cv2.calcHist([hsv], [i], None, [256], [0, 256])
                plt.plot(histogram, color=color)
            plt.xlim([0, 256])
            plt.show()
            cv2.imshow("Histogram equalization by RGB", rgb)  # Отображение обработанных изображений
            cv2.imshow("Histogram equalization by HSV", hsv)
            main_window.blockButtons(self, True)  # Блокировка кнопок
            cv2.waitKey(0)
            main_window.blockButtons(self, False)  # Разблокировка кнопок
            plt.close()
            #cv2.destroyAllWindows()

    # Метод для применения различных методов размытия к изображению
    def blur(self):
        if self.fname == "":
            self.showDialog(1)  # Показ предупреждения, если изображение не выбрано
        else:
            # Применение усредненного, гауссова и медианного размытия к изображению
            ImageProcessing.average_blur(self.fname)
            ImageProcessing.gaussian_blur(self.fname)
            ImageProcessing.median_blur(self.fname)
            main_window.blockButtons(self, True)  # Блокировка кнопок
            cv2.waitKey(0)
            main_window.blockButtons(self, False)  # Разблокировка кнопок
            #cv2.destroyAllWindows()

    # Метод для показа диалогового окна с предупреждением
    def showDialog(self, id):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        if id == 1:
            msgBox.setText("No image to process")  # Предупреждение о том, что изображение не выбрано
            msgBox.setWindowTitle("Warning")
        elif id == 2:
            msgBox.setText("Max image size is (1000, 1000)")  # Предупреждение о максимальном размере изображения
            msgBox.setWindowTitle("Warning")

        rar = msgBox.exec()

    # Определение метода для обработки изображений в оттенках серого
    def grayscale(self):
        if self.fname == "":
            self.showDialog(1)  # Показ предупреждения, если изображение не выбрано
        else:
            # Применение методов гистограммного выравнивания и линейной коррекции контраста
            pic_hist = ImageProcessing.equalization_hist_grayscale(self.fname)
            pic_linear = ImageProcessing.linear_contrasting_grayscale(self.fname)
            # Отображение обработанных изображений
            cv2.imshow('Histogram equalization', pic_hist)
            cv2.imshow('Linear contrasting', pic_linear)

            # Построение и отображение гистограмм обработанных изображений при помощи Matplotlib
            plt.subplot(1, 2, 1)
            hist, bins = np.histogram(pic_hist.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color='b')
            plt.hist(pic_hist.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.title("Histogram equalization")
            plt.legend(('cdf', 'histogram'), loc='upper left')

            plt.subplot(1, 2, 2)
            hist, bins = np.histogram(pic_linear.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color='b')
            plt.hist(pic_linear.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.title("Linear constasting")
            plt.legend(('cdf', 'histogram'), loc='upper left')

            plt.show()  # Отображение гистограмм
            main_window.blockButtons(self, True)  # Блокировка кнопок
            cv2.waitKey(0)
            plt.close
            main_window.blockButtons(self, False)  # Разблокировка кнопок

            #cv2.destroyAllWindows()

    def __init__(self, parent=None):
        super(main_window, self).__init__(parent)
        layout = QVBoxLayout()
        self.fname = ""
        self.btnChoose = QPushButton("Choose image...")  # Кнопка выбора изображения
        self.btnChoose.clicked.connect(self.getfile)  # Привязка обработчика событий к кнопке
        self.btnGrayscale = QPushButton(
            "Histogram equalization and linear contrasting")  # Кнопка для обработки изображения в оттенках серого
        self.btnGrayscale.clicked.connect(self.grayscale)  # Привязка обработчика событий к кнопке
        self.btnColorful = QPushButton(
            "Histogram equalization for RGB and HSV")  # Кнопка для обработки цветного изображения
        self.btnColorful.clicked.connect(self.colorful)  # Привязка обработчика событий к кнопке
        self.btnBlur = QPushButton("Smoothing filters")  # Кнопка для применения размытия
        self.btnBlur.clicked.connect(self.blur)  # Привязка обработчика событий к кнопке

        layout.addWidget(self.btnChoose)
        layout.addWidget(self.btnGrayscale)
        layout.addWidget(self.btnColorful)
        layout.addWidget(self.btnBlur)
        self.le = QLabel("Hello", self)
        self.le.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.le.setAlignment(Qt.AlignCenter)
        self.le.setStyleSheet("QLabel {background-color: black;}")

        layout.addWidget(self.le)
        self.setLayout(layout)
        self.setWindowTitle("Image processing")  # Установка заголовка окна
        self.setGeometry(100, 100, 100, 100)  # Установка размеров окна

    def getfile(self):
        self.fname, _ = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:', "Image files (*.jpg *.jpeg *.png *.bmp)")  # Диалоговое окно выбора изображения
        pic = QPixmap(self.fname, _)  # Загрузка изображения
        self.path = _ + "\\" + self.fname
        if (pic.width() > 1000 or pic.height() > 1000):  # Если изображение слишком большое
            self.showDialog(2)  # Показать предупреждение
        else:
            self.le.setPixmap(pic)  # Отображение изображения
            self.resize(pic.width(), pic.height())  # Изменение размеров окна

# Запуск главного окна приложения
def main():
    app = QApplication(sys.argv)
    ex = main_window()
    ex.show()
    sys.exit(app.exec_())


# Проверка, что скрипт запускается как основной скрипт
if __name__ == '__main__':
    main()