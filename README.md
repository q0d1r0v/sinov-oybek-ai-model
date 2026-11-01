# Sinov-Oybek

Digital Group'ning sinov modeli: noldan o'qitilgan, 10 million parametrli o'zbek tilidagi kichik model.
U Linux'da, shu kompyuterning o'zida, internetsiz ishlaydi.

> **Bu sinov modeli, tayyor mahsulot emas.** Javoblari ko'pincha xato bo'ladi: 36 ta bilim tekshiruvidan 5 tasi,
> 15 ta hisob tekshiruvidan 0 tasi to'g'ri chiqqan. "Kim?" savollarida o'zini tanishtirib yuboradi.
> Javoblarni tekshirmasdan ishlatmang. Modelda kontent filtri yo'q.

## Talablar

- Linux, Python 3.10+ va `python3-venv`
- 4 GB RAM (tavsiya 8 GB), 2 yadroli protsessor
- Videokarta shart emas
- O'rnatish uchun taxminan 1.2 GB bo'sh joy va internet

## O'rnatish va ishga tushirish

```bash
git clone https://github.com/q0d1r0v/sinov-oybek-ai-model.git
cd sinov-oybek-ai-model
./install.sh
./run.sh
```

Keyin brauzerda oching: <http://127.0.0.1:8000>

Terminalda gaplashish: `.venv/bin/python chat.py`

## Senzura

Bu model senzurasiz: unga savollarni rad etish o'rgatilmagan va dasturda kontent filtri yo'q.
50 savolli sinovdagi 150 ta javobning faqat 1 tasida rad belgisi topilgan.

- Model nomaqbul, noto'g'ri yoki bir tomonlama javob berishi mumkin.
- O'qitilgan matnlardagi qarashlar javoblarga ham o'tishi mumkin.
- Javoblar va ulardan foydalanish uchun foydalanuvchining o'zi javob beradi. Bolalarga bermang.

## Maxfiylik

Server faqat `127.0.0.1` da ishlaydi, boshqa kompyuterlardan ochilmaydi. Yozishmalar saqlanmaydi va hech qayerga yuborilmaydi.
