# Intelligent_Placer
## Постановка задачи
### Общее описание
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику определить, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Также заранее известно направление вертикальной оси Z у этих предметов.

### Входные/выходные данные
**Вход**: Фотография, на которой изображены:
+ предметы, лежащие на поверхности;
+ многоугольник, нарисованный темным маркером на белом листе бумаги А4, сфотографированный вместе с предметами.

**Вывод**: Ответ в текстовом формате, записанный в файл answers.txt:
+ "Yes, [имя исходной фотографии]" - предметы можно поместить в многоугольник;
+ "No, [имя исходной фотографии]" - предметы нельзя поместить в многоугольник.

### Требования к входным данным
#### Требования к фотографиям
+ Без сжатия, цветовой коррекции;
+ Формат фотографии .jpg;
+ Съемка производится горизонтально с допустимым отклонением (возможна ошибка до 10°);
+ Высота съемки для всех фотографий 30-45 см;
+ Отсутствие размытости фотографии: толщина линий границ предметов не более 10px;
+ Освещение фотографии должно быть одинаковым.

#### Требования к предметам
+ Предметы не касаются и не перекрывают друг друга;
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

## Предметы
https://drive.google.com/file/d/1Y9jZStHf-LM3eoG-qy7lYzYjrFvImGJS/view?usp=sharing
