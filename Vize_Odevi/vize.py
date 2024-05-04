import sys
import numpy as np
import cv2
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QFileDialog, QDialog, QMdiSubWindow 
from PyQt5.QtWidgets import QInputDialog, QPushButton
from PyQt5.QtWidgets import QMdiArea, QAction, QHBoxLayout, QTableView
from PyQt5.QtGui import QPixmap, QFont, QColor, qGray, QImage
from math import radians, cos, sin

class ImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resim Görüntüleyici")
        self.image_label = QLabel()
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.image_label.setPixmap(self.pixmap)
        self.change_tone_button = QPushButton("Gri Ton Değiştir")
        self.change_tone_button.clicked.connect(self.ton_degistir)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.change_tone_button)
        self.setLayout(layout)
        
    def ton_degistir(self):
        image = self.pixmap.toImage()
        width = image.width()
        height = image.height()
        for x in range(width):
            for y in range(height):
                color = QColor(image.pixel(x, y))
                gray_value = qGray(color.rgb())
                new_color = QColor(gray_value, gray_value, gray_value)
                image.setPixelColor(x, y, new_color)
        self.pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(self.pixmap)
        
class ImageDialog_2(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resim Görüntüleyici")
        self.image_label = QLabel()
        self.pixmap = QPixmap(image_path)
        
        if not self.pixmap.isNull():
            self.image_label.setPixmap(self.pixmap)
            
        self.button1 = QPushButton("Görüntü Boyutunu Büyütme")
        self.button1.clicked.connect(self.buyut)
        
        self.button2 = QPushButton("Görüntü Boyutunu Küçültme")
        self.button2.clicked.connect(self.kucult)
        
        self.button3 = QPushButton("Zoom In")
        self.button3.clicked.connect(self.zoom_in)
        
        self.button4= QPushButton("Zoom Out")
        self.button4.clicked.connect(self.zoom_out)
        
        self.button5 = QPushButton("Görüntü Döndürme")
        self.button5.clicked.connect(self.dondur)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)
        layout.addWidget(self.button5)
        self.setLayout(layout)
    
    def buyut(self):
        percent, ok = QInputDialog.getDouble(self, "Büyüt", "Büyütme yüzdesini giriniz:", 10)
        if ok:
            factor = 1+percent / 100
            image = self.pixmap.toImage()
            width = image.width()
            height = image.height()
            new_width = int(width * factor)
            new_height = int(height * factor)
            new_image = QImage(new_width, new_height, QImage.Format_RGB32)
            for x in range(new_width):
                for y in range(new_height):
                    source_x = int(x / factor)
                    source_y = int(y / factor)
                    alpha = (x / factor) - source_x
                    beta = (y / factor) - source_y
                    color1 = self.get_pixel_color(image, source_x, source_y)
                    color2 = self.get_pixel_color(image, source_x + 1, source_y)
                    color3 = self.get_pixel_color(image, source_x, source_y + 1)
                    color4 = self.get_pixel_color(image, source_x + 1, source_y + 1)
                    new_color = self.bilinear_interpolation(color1, color2, color3, color4, alpha, beta)
                    new_image.setPixelColor(x, y, new_color)
            self.pixmap = QPixmap.fromImage(new_image)
            self.image_label.setPixmap(self.pixmap)
    
    def kucult(self):
        percent, ok = QInputDialog.getDouble(self, "Küçült", "Küçültme yüzdesini giriniz:", 50)
        if ok:
            factor = 1 - percent / 100
            image = self.pixmap.toImage()
            width = image.width()
            height = image.height()
            new_width = int(width * factor)
            new_height = int(height * factor)
            new_image = QImage(new_width, new_height, QImage.Format_RGB32)
            for x in range(new_width):
                for y in range(new_height):
                    source_x = x / factor
                    source_y = y / factor
                    alpha = source_x - int(source_x)
                    beta = source_y - int(source_y)
                    color1 = self.get_pixel_color(image, int(source_x), int(source_y))
                    color2 = self.get_pixel_color(image, int(source_x) + 1, int(source_y))
                    color3 = self.get_pixel_color(image, int(source_x), int(source_y) + 1)
                    color4 = self.get_pixel_color(image, int(source_x) + 1, int(source_y) + 1)
                    new_color = self.bilinear_interpolation(color1, color2, color3, color4, alpha, beta)
                    new_image.setPixelColor(x, y, new_color)
            self.pixmap = QPixmap.fromImage(new_image)
            self.image_label.setPixmap(self.pixmap)
    
    def zoom_in(self):
        factor = 1.25
        image = self.pixmap.toImage()
        width = image.width()
        height = image.height()
        new_width = int(width * factor)
        new_height = int(height * factor)
        new_image = QImage(new_width, new_height, QImage.Format_RGB32)
        for x in range(new_width):
            for y in range(new_height):
                source_x = x / factor
                source_y = y / factor
                new_color = self.average_interpolation(image, source_x, source_y, factor)
                new_image.setPixelColor(x, y, new_color)
        self.pixmap = QPixmap.fromImage(new_image)
        self.image_label.setPixmap(self.pixmap)
    
    def zoom_out(self):
        factor = 0.75
        image = self.pixmap.toImage()
        width = image.width()
        height = image.height()
        new_width = int(width * factor)
        new_height = int(height * factor)
        new_image = QImage(new_width, new_height, QImage.Format_RGB32)
        for x in range(new_width):
            for y in range(new_height):
                source_x = x / factor
                source_y = y / factor
                new_color = self.average_interpolation(image, source_x, source_y, factor)
                new_image.setPixelColor(x, y, new_color)
        self.pixmap = QPixmap.fromImage(new_image)
        self.image_label.setPixmap(self.pixmap)

    def dondur(self):
        angle, ok = QInputDialog.getDouble(self, "Döndür", "Dönme açısını derece cinsinden giriniz:", 90)
        if ok:
            image = self.pixmap.toImage()
            width = image.width()
            height = image.height()
            angle_rad = radians(angle)
            center_x = width / 2
            center_y = height / 2
            new_image = QImage(width, height, QImage.Format_RGB32)
            for x in range(width):
                for y in range(height):
                    new_x = (x - center_x) * cos(angle_rad) - (y - center_y) * sin(angle_rad) + center_x
                    new_y = (x - center_x) * sin(angle_rad) + (y - center_y) * cos(angle_rad) + center_y
                    new_color = self.bicubic_interpolation(image, new_x, new_y)
                    new_image.setPixelColor(x, y, new_color)
            self.pixmap = QPixmap.fromImage(new_image)
            self.image_label.setPixmap(self.pixmap)
    
    def get_pixel_color(self, image, x, y):
        x = max(0, min(image.width() - 1, int(x)))
        y = max(0, min(image.height() - 1, int(y)))
        return QColor(image.pixel(x, y))

    def bilinear_interpolation(self, color1, color2, color3, color4, alpha, beta):
        r = (1 - alpha) * (1 - beta) * color1.red() + alpha * (1 - beta) * color2.red() + \
            (1 - alpha) * beta * color3.red() + alpha * beta * color4.red()
        g = (1 - alpha) * (1 - beta) * color1.green() + alpha * (1 - beta) * color2.green() + \
            (1 - alpha) * beta * color3.green() + alpha * beta * color4.green()
        b = (1 - alpha) * (1 - beta) * color1.blue() + alpha * (1 - beta) * color2.blue() + \
            (1 - alpha) * beta * color3.blue() + alpha * beta * color4.blue()
        return QColor(int(r), int(g), int(b))
    
    def average_interpolation(self, image, x, y, factor):
        source_x = int(x)
        source_y = int(y)
        color1 = self.get_pixel_color(image, source_x, source_y)
        color2 = self.get_pixel_color(image, source_x + 1, source_y)
        color3 = self.get_pixel_color(image, source_x, source_y + 1)
        color4 = self.get_pixel_color(image, source_x + 1, source_y + 1)
        r = (color1.red() + color2.red() + color3.red() + color4.red()) / 4
        g = (color1.green() + color2.green() + color3.green() + color4.green()) / 4
        b = (color1.blue() + color2.blue() + color3.blue() + color4.blue()) / 4
        return QColor(int(r), int(g), int(b))
    
    def bicubic_interpolation(self, image, x, y):
        x_int = int(x)
        y_int = int(y)
        p11 = self.get_pixel_color(image, x_int - 1, y_int - 1)
        p12 = self.get_pixel_color(image, x_int, y_int - 1)
        p13 = self.get_pixel_color(image, x_int + 1, y_int - 1)
        p14 = self.get_pixel_color(image, x_int + 2, y_int - 1)
        p21 = self.get_pixel_color(image, x_int - 1, y_int)
        p22 = self.get_pixel_color(image, x_int, y_int)
        p23 = self.get_pixel_color(image, x_int + 1, y_int)
        p24 = self.get_pixel_color(image, x_int + 2, y_int)
        p31 = self.get_pixel_color(image, x_int - 1, y_int + 1)
        p32 = self.get_pixel_color(image, x_int, y_int + 1)
        p33 = self.get_pixel_color(image, x_int + 1, y_int + 1)
        p34 = self.get_pixel_color(image, x_int + 2, y_int + 1)
        p41 = self.get_pixel_color(image, x_int - 1, y_int + 2)
        p42 = self.get_pixel_color(image, x_int, y_int + 2)
        p43 = self.get_pixel_color(image, x_int + 1, y_int + 2)
        p44 = self.get_pixel_color(image, x_int + 2, y_int + 2)
        r = self.bicubic_channel_interpolation(p11.red(), p12.red(), p13.red(), p14.red(),
                                               p21.red(), p22.red(), p23.red(), p24.red(),
                                               p31.red(), p32.red(), p33.red(), p34.red(),
                                               p41.red(), p42.red(), p43.red(), p44.red(),
                                               x - x_int, y - y_int)
        g = self.bicubic_channel_interpolation(p11.green(), p12.green(), p13.green(), p14.green(),
                                               p21.green(), p22.green(), p23.green(), p24.green(),
                                               p31.green(), p32.green(), p33.green(), p34.green(),
                                               p41.green(), p42.green(), p43.green(), p44.green(),
                                               x - x_int, y - y_int)
        b = self.bicubic_channel_interpolation(p11.blue(), p12.blue(), p13.blue(), p14.blue(),
                                               p21.blue(), p22.blue(), p23.blue(), p24.blue(),
                                               p31.blue(), p32.blue(), p33.blue(), p34.blue(),
                                               p41.blue(), p42.blue(), p43.blue(), p44.blue(),
                                               x - x_int, y - y_int)
        return QColor(int(r), int(g), int(b))
    
    def bicubic_channel_interpolation(self, p11, p12, p13, p14, p21, p22, p23, p24,
                                      p31, p32, p33, p34, p41, p42, p43, p44, dx, dy):
        def cubic_hermite(a, b, c, d, t):
            return a + 0.5 * t * (b - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)))
        interpolated_value = cubic_hermite(p22, p32, p12, p22, dy)
        interpolated_value = cubic_hermite(interpolated_value, cubic_hermite(p23, p33, p13, p23, dy),
                                           cubic_hermite(p21, p31, p11, p21, dy),
                                           cubic_hermite(p24, p34, p14, p24, dy), dx)
        interpolated_value = max(0, min(255, interpolated_value))
        return interpolated_value        
        
class ImageDialog_3_SCurve(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resim Görüntüleyici")
        self.image_path = image_path
        self.image_label = QLabel()
        self.pixmap = QPixmap(self.image_path)
        
        if not self.pixmap.isNull():
            self.image_label.setPixmap(self.pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
    def standard_sigmoid(self, pixel_value):
        return 1 / (1 + np.exp(-0.01 * (pixel_value - 128)))

    def shifted_sigmoid(self, pixel_value):
        return 1 / (1 + np.exp(-(pixel_value - 128) / 32))

    def sloped_sigmoid(self, pixel_value, slope):
        return 1 / (1 + np.exp(-slope * (pixel_value - 128)))

    def kendi_fonksiyonum(self, pixel_values):
        return 128 * pixel_values / (128 + pixel_values)

    def apply_s_curve(self, image, sigmoid_function, function_name):
        normalized_image = cv2.normalize(image.astype('float32'), 
                   None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        transformed_image = sigmoid_function(normalized_image)
        transformed_image = cv2.normalize(transformed_image, 
                   None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        transformed_image_with_text = cv2.putText(transformed_image, function_name, 
                   (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return transformed_image_with_text

class ImageDialog_YolCizgiTespit(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resim Görüntüleyici")
        self.image_path = image_path
        layout = QHBoxLayout()
        self.original_image_label = QLabel()
        self.lines_image_label = QLabel()
        self.load_images()
        layout.addWidget(self.original_image_label)
        layout.addWidget(self.lines_image_label)
        self.setLayout(layout)

    def load_images(self):
        original_pixmap = QPixmap(self.image_path)
        lines_pixmap = self.detect_and_draw_lines()
        self.original_image_label.setPixmap(original_pixmap)
        self.lines_image_label.setPixmap(lines_pixmap)

    def detect_and_draw_lines(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

class ImageDialog_GozTespit(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resim Görüntüleyici")
        self.image_label = QLabel()
        self.image_label.setPixmap(self.detect_eyes_hough(image_path))
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
    
    def detect_eyes_hough(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 170)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 
                  dp=1, minDist=20, param1=50, param2=30,
                  minRadius=10, maxRadius=50)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(image, center, radius, (0, 255, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap
       
class ExcelDialog(QDialog):
    def __init__(self, excel_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Excel Gösterici")
        self.df = pd.read_excel(excel_path)
        self.model = QStandardItemModel(self)
        self.model.setColumnCount(self.df.shape[1])
        self.model.setRowCount(self.df.shape[0])
        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[1]):
                item = QStandardItem(str(self.df.iat[i, j]))
                self.model.setItem(i, j, item)
        self.model.setHorizontalHeaderLabels(list(self.df.columns))

        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.resizeColumnsToContents()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijital Görüntü İşleme Dersi Arabirimi")
        self.resize(2000,1000)
        
        menu_bar = self.menuBar()
        odev1_menu = menu_bar.addMenu("Ödev 1: Temel İşlevselliği Oluştur")
        odev2_menu = menu_bar.addMenu("Ödev 2: Temel Görüntü Operasyonları")
        odev3_menu = menu_bar.addMenu("Ödev 3: Vize Ödevi")

        self.ogrenci_bilgileri_label = QLabel()
        self.baslik_bilgileri()
        font = QFont()  
        font.setBold(True)  
        font.setPointSize(10)
        self.ogrenci_bilgileri_label.setFont(font)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.ogrenci_bilgileri_label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.mdi_area = QMdiArea()
        layout.addWidget(self.mdi_area)
        
        odev1_action = QAction("Resim aç", self)
        odev1_action.triggered.connect(self.resmi_goster)
        odev1_menu.addAction(odev1_action)
        
        odev2_action = QAction("Resim aç", self)
        odev2_action.triggered.connect(self.resmi_goster_2)
        odev2_menu.addAction(odev2_action)
        
        odev3_action1=QAction("S_Curve", self)
        odev3_action1.triggered.connect(self.resmi_goster_3)
        odev3_menu.addAction(odev3_action1)
        
        odev3_action2 = QAction("Yol Çizgi Tespit", self)
        odev3_action2.triggered.connect(self.yol_cizgi)
        odev3_menu.addAction(odev3_action2)
        
        odev3_action3 = QAction("Göz Tespiti", self)
        odev3_action3.triggered.connect(self.goz_tespit)
        odev3_menu.addAction(odev3_action3)
        
        odev3_action3 = QAction("Nesne Sayma", self)
        odev3_action3.triggered.connect(self.nesnesayma)
        odev3_menu.addAction(odev3_action3)
        

    def baslik_bilgileri(self):
        bilgi_metni = (
        "Öğrenci Bilgileri: 211229024-Alişahin Asya "
        "Ders Adı: Dijital Görüntü İşleme")
        self.ogrenci_bilgileri_label.setText(bilgi_metni)


    def resmi_goster_3(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            image_dialog = ImageDialog_3_SCurve(file_path, parent=self)
            standard_s_curve_image = image_dialog.apply_s_curve(image, 
                            image_dialog.standard_sigmoid, "Standard Sigmoid")
            shifted_s_curve_image = image_dialog.apply_s_curve(image, 
                              image_dialog.shifted_sigmoid, "Shifted Sigmoid")
            slope = 0.05
            sloped_s_curve_image = image_dialog.apply_s_curve(image, 
            lambda x: image_dialog.sloped_sigmoid(x, slope), "Sloped Sigmoid")
            custom_s_curve_image = image_dialog.apply_s_curve(image, 
                          image_dialog.kendi_fonksiyonum, "Kendi Fonksiyonum")

            original_image_with_text = cv2.putText(image.copy(),"Original Image",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            combined_image = np.hstack((original_image_with_text,  
                standard_s_curve_image, shifted_s_curve_image,
                sloped_s_curve_image, custom_s_curve_image))
            
            qt_image = QImage(combined_image.data, combined_image.shape[1], 
                        combined_image.shape[0], combined_image.shape[1], 
                        QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            
            image_dialog.image_label.setPixmap(pixmap)
            image_dialog.exec_()
          
    def yol_cizgi(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            image_dialog = ImageDialog_YolCizgiTespit(file_path, parent=self)
            mdi_subwindow = QMdiSubWindow()
            mdi_subwindow.setWidget(image_dialog)
            self.mdi_area.addSubWindow(mdi_subwindow)
            mdi_subwindow.resize(1800, 900)
            mdi_subwindow.show()            
            
    def goz_tespit(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            image_dialog = ImageDialog_GozTespit(file_path, parent=self)
            mdi_subwindow = QMdiSubWindow()
            mdi_subwindow.setWidget(image_dialog)
            self.mdi_area.addSubWindow(mdi_subwindow)
            mdi_subwindow.resize(900, 900)  
            mdi_subwindow.show()
            
    def nesnesayma(self):
        file_path = "image.jpg"
        image = cv2.imread(file_path)
        lower_green = np.array([0, 100, 0])
        upper_green = np.array([100, 255, 100])
        mask = cv2.inRange(image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            diagonal = np.sqrt(w**2 + h**2)
            moments = cv2.moments(contour)
            energy = -1 * np.sum([p * np.log(p + 1e-6) for p in moments.values() if p > 0])
            entropy = -1 * np.sum([p / np.sum(list(moments.values())) * np.log(p / np.sum(list(moments.values()))) 
                                   for p in moments.values() if p > 0])
            mean_val = np.mean(image[y:y+h, x:x+w])
            median_val = np.median(image[y:y+h, x:x+w])
            data.append([i+1, (x+w/2, y+h/2), w, h, diagonal, energy, entropy, mean_val, median_val])
        df = pd.DataFrame(data, columns=["No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])
        excel_path = "koyu_yesil_bolgeler.xlsx"
        df.to_excel(excel_path, index=False)
        excel_dialog = ExcelDialog(excel_path, parent=self)
        mdi_subwindow = QMdiSubWindow()
        mdi_subwindow.setWidget(excel_dialog)
        self.mdi_area.addSubWindow(mdi_subwindow)
        mdi_subwindow.resize(1000, 1000)
        mdi_subwindow.show()            
            
    def resmi_goster(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            image_dialog = ImageDialog(file_path, parent=self)
            mdi_subwindow = QMdiSubWindow()
            mdi_subwindow.setWidget(image_dialog)
            self.mdi_area.addSubWindow(mdi_subwindow)
            mdi_subwindow.resize(1900, 900)
            mdi_subwindow.show()
          
    def resmi_goster_2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            image_dialog = ImageDialog_2(file_path, parent=self)
            mdi_subwindow = QMdiSubWindow()
            mdi_subwindow.setWidget(image_dialog)
            self.mdi_area.addSubWindow(mdi_subwindow)
            mdi_subwindow.resize(1900, 900)
            mdi_subwindow.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
        
