#include <iostream>
#include <fstream>
#include <vector>

#pragma pack(push, 1) // Выравнивание структуры по 1 байту

struct BMPHeader {
    uint16_t bfType;      // Тип файла (должен быть 'BM')
    uint32_t bfSize;      // Размер файла в байтах
    uint16_t bfReserved1; // Зарезервировано
    uint16_t bfReserved2; // Зарезервировано
    uint32_t bfOffBits;   // Смещение до данных изображения
};

struct DIBHeader {
    uint32_t biSize;          // Размер DIB-заголовка
    int32_t biWidth;          // Ширина изображения
    int32_t biHeight;         // Высота изображения
    uint16_t biPlanes;        // Количество цветовых плоскостей
    uint16_t biBitCount;      // Количество бит на пиксель
    uint32_t biCompression;    // Тип сжатия
    uint32_t biSizeImage;     // Размер изображения в байтах
    int32_t biXPelsPerMeter;   // Горизонтальное разрешение
    int32_t biYPelsPerMeter;   // Вертикальное разрешение
    uint32_t biClrUsed;       // Количество используемых цветов
    uint32_t biClrImportant;   // Количество важных цветов
};
#pragma pack(pop)

void readBMP(const std::string &inputFile, std::vector<char> &pixelData, std::vector<char> &palette, BMPHeader &bmpHeader, DIBHeader &dibHeader) {
    std::ifstream file(inputFile, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + inputFile);
    }

    // Чтение заголовка BMP
    file.read(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));
    if (bmpHeader.bfType != 0x4D42) { // 'BM'
        throw std::runtime_error("Файл не является BMP");
    }

    // Чтение DIB заголовка
    file.read(reinterpret_cast<char*>(&dibHeader), sizeof(DIBHeader));

    // Проверяем, что изображение 8-битное
    if (dibHeader.biBitCount != 8) {
        throw std::runtime_error("Поддерживаются только 8-битные BMP файлы");
    }

    // Определяем размер строки (с учетом выравнивания)
    size_t rowSize = ((dibHeader.biWidth + 3) & (~3)); // Выравнивание по 4 байта
    pixelData.resize(rowSize * dibHeader.biHeight);

    // Чтение палитры (256 цветов)
    palette.resize(1024); // 256 цветов * 4 байта на цвет (RGB + зарезервированный)
    file.read(reinterpret_cast<char*>(palette.data()), palette.size());

    // Чтение данных пикселей
    file.seekg(bmpHeader.bfOffBits, std::ios::beg);
    file.read(reinterpret_cast<char*>(pixelData.data()), pixelData.size());

    file.close();
}

void writeBMP(const std::string &outputFile, const std::vector<char> &pixelData, const std::vector<char> &palette, const BMPHeader &bmpHeader, const DIBHeader &dibHeader) {
    std::ofstream file(outputFile, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + outputFile);
    }

    // Запись заголовка BMP
    file.write(reinterpret_cast<const char*>(&bmpHeader), sizeof(BMPHeader));
    
    // Запись DIB заголовка
    file.write(reinterpret_cast<const char*>(&dibHeader), sizeof(DIBHeader));
    
    // Запись палитры
    file.write(reinterpret_cast<const char*>(palette.data()), palette.size());
    
    // Запись данных пикселей
    file.write(reinterpret_cast<const char*>(pixelData.data()), pixelData.size());

    file.close();
}


__global__ void Kernel(char* input, char* output, int n) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    char window[9] = {0,0,0,0,0,0,0,0,0};
    if(x >= 1 && x < n - 1 && y >= 1 && y < n - 1 ){
        window = {0,0,0,0,0,0,0,0,0};
        idx = 0;
        for (int i = -1; i < 2; ++i){
            for (int j = -1; j < 2; ++j){
                window[idx] = input[y+i,x+j];
                idx+=1;
            }
        }
    }
}

int main() {
    const std::string inputFile = "input.bmp";  // Путь к входному файлу BMP
    const std::string outputFile = "output_8bit.bmp"; // Путь к выходному файлу BMP

    BMPHeader bmpHeader;
    DIBHeader dibHeader;
    std::vector<char> pixelData;
    std::vector<char> palette;

    try {
        readBMP(inputFile, pixelData, palette, bmpHeader, dibHeader);
        // for (int i = 0; i < pixelData.size(); i++){
        //     pixelData[i] = 0;
        // }
        writeBMP(outputFile, pixelData, palette, bmpHeader, dibHeader);
        std::cout << "GREAT! " << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

   

    return 0;
}
