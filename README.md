# Intelligent_Placer
## Постановка задачи
### Общее описание
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику определить, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Также заранее известно направление вертикальной оси Z у этих предметов.

### Входные/выходные данные
**Вход**: Фотография, на которой изображены:
+ предметы, лежащие на поверхности;
+ многоугольник, нарисованный темным маркером на белом листе бумаги А4, сфотографированный вместе с предметами.

**Вывод**: Ответ в текстовом формате:
+ "Yes" - предметы можно поместить в многоугольник;
+ "No" - предметы нельзя поместить в многоугольник;
+ В случае положительного ответа также выдается двуцветное изображение, демонстрирующее удачный "плейсинг"

### Требования к входным данным
#### Требования к фотографиям
+ Без цветовой коррекции;
+ Формат фотографии .jpg;
+ Съемка производится горизонтально с допустимым отклонением (возможна ошибка до 10°);
+ Высота съемки для всех фотографий 30-45 см;
+ Отсутствие размытости фотографии;
+ Освещение фотографии должно быть одинаковым.

#### Требования к предметам
+ Предметы не касаются и не перекрывают друг друга;
+ Все предметы находятся строго ниже листа с многоугольником;
+ Описывающие прямоугольники предметов также не пересекаются;
+ Предметы целиком находятся внутри фотографии;
+ На фотографии отсутствуют предметы помимо тех, что представлены;
+ Каждый предмет представлен единожды.

#### Требования к многоугольнику
+ Многоугольник нарисован на полностью попадающим в кадр чистом белом листе А4;
+ Линии многоугольника нарисованы синим перманентным маркером толщиной 1-3 мм;
+ Многоугольник должен быть выпуклым;
+ Число вершин многоугольника должно быть не более 10.

#### Требования к поверхности
+ Поверхность достаточно велика, чтобы фоном занимать все изображение;
+ Поверхность едина для всех фотографий.

### План решения задачи
1. Поиск многоугольника на изображении  
  1.1 Выделение фильтром листа бумаги  
  1.2 Вырезание листа из изображения  
  1.3 Наложение фильтра для выделения границ (sobel)  
  1.4 Применение морфологических преобразований для замыкания границы фигуры и избавления от шумов  
  1.5 Выделение контура многоугольника  
2. Поиск объектов на тестовом изображении  
  2.1 Вырезание области под листом из изображения  
  2.2 Выделение контуров комбинированной цветовой маской  
  2.3 Отсеивание слишком маленьких контуров  
3. Поиск объектов на эталонных изображениях  
  3.1 Выделение фильтром листа бумаги  
  3.2 Вырезание листа из изображения  
  3.3 Наложение порогового фильтра (threshold_otsu)  
4. Идентификация объектов  
  4.1 Выделение основых цветов на обрезанных изображениях объектов k-means кластеризацией  
  4.2 Идентификация объектов по полученным цветам  
  4.3 Сопоставление объектам контуров с эталонных изображений  
  4.4 Изменение размеров контура эталонного объекта для соответсвия тестовому изображению  
5. Размещение объектов  
  5.1 Тривиальные проверки на невозможность размещения (по площади, длине, ширине)  
  5.2 Рекурсивное размещение объектов перебором по сетке с заданными шагами смещения и поворота  
### Оценка результатов
+ Полученная точность на тестовом наборе - 75%
+ Более частыми являются ложноотрицательные ошибки
+ Из-за особенностей входных данных иногда возникают проблемы с получением контура многоугольника, это можно исправить более грамотным применением морфологических преобразований
+ Также ожидаемо есть проблемы с нахождением на изображении предметов, близких по цвету с фоном. Однако иногда удается обнаружить и их, что подталкивает на мысли, что можно достичь идеального результата более тщательным подбором параметров цветовых масок

## Предметы
https://drive.google.com/drive/folders/14sYtuP_SSFzUdqtuA2_dA-zmxr-Egy--?usp=sharing
