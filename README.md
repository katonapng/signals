# signals

Постановка:

На вход подаётся изображение арки и прямоугольного объекта. Требуется установить, пролезает ли прямоугольный объект в арку (с некоторой погрешностью), если перемещать его параллельным переносом не отрывая от поверхности.



Условия:

На вход подаётся цветное изображение формата jpg или png. Минимальное разрешение 800 на 600. Оба предмета должны стоять на одной поверхности. Оба предмета должны быть на фотографии. Предметы должны быть хорошо освещены искусственным или дневным светом, без резких теней, без засветов. Предметы должны находится в фокусе. Предметы не должны касаться друг друга. На изображении должны быть только арка и прямоугольный объект. Оба предмета не должны вылезать за рамки изображения. Предметы не перекрывают друг друга. Фотографии не могут быть сделаны сверху (так что отверстие арки не видно). Рамка должна стоять на поверхности так, чтобы было видно отверстие. Рамка не может быть поставлена на соединяющую перекладину.

План:

1. Проведём бинаризацию с помощью фильтра threshold_yen 
2. Уберём пробелы в маске с помощью binary_closing и binary_opening
3. Выделим две наибольшие компоненты
4. Компонента, у которой значение по координате y меньше - арка, вторая компонента - коробка
5. Найдем y координату основания арки и ширину арки (предположение: коробка находится недалеко от арки и всегда лежит на столе, поэтому достаточно знать ширину основания арки)
6. Найдем линии границ коробки с помощью преобразования Хафа (перед этим найдем границы фильтром Кенни) и уберем лишние линии
7. Найдем пересечение y координаты арки с линиями, найденными через преобразование Хафа и с помощью этих координат получим масштабированную ширину коробки
8. Спроецируем оставшуюся сторону коробки на арку (сейчас не учитывается случай, если коробка лежит большей стороной к арке)
9. Таким образом знаем ширину арки, спроецированную ширину коробки для обоих сторон коробки, сравниваем их и проверяем пролезит ли коробка
