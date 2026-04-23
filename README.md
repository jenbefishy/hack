# ATV Signal Processing Repository

Репозиторий для обработки IQ-данных с задачами:
- FM демодуляция;
- детекция HSYNC/VSYNC;
- реконструкция кадров;
- сохранение графиков и отчётов.

## Структура

```text
.
├── data/               # входные IQ-файлы
├── scripts/            # примеры bash-скриптов запуска
├── src/
│   ├── main.py         # CLI-точка входа
│   └── atv/            # пакет с логикой обработки
├── requirements.txt
└── README.md
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Запускайте из **корня репозитория** (или используйте `scripts/*.sh`, они сами переходят в корень).

### Через CLI

```bash
python3 src/main.py --input data/iq_capture.cf32 --fs 20000000 --out results/default_run
```

Модульный запуск **только из корня репозитория** (иначе Python не находит пакет `src`):

```bash
python3 -m src.main --input data/iq_capture.cf32 --fs 20000000 --out results/default_run
```

Скрипт `src/main.py` добавляет каталог `src` в `sys.path`, поэтому `python3 /полный/путь/к/репо/src/main.py ...` можно вызывать из любой текущей директории.

Полный проход по сигналу (без лимита кадров; для длинных записей задайте `--video-fps 0`, иначе будет собираться большой MP4):

```bash
python3 src/main.py --input data/iq_capture.npy --fs 20000000 --max-frames 0 --video-fps 0 --out results/full_signal_run
```

Параметр **`--video-fps`**: `< 0` — авто (PAL ~25 / NTSC ~29.97 по длительности строки), `0` — не писать MP4, `> 0` — явный FPS.

### Через bash-скрипты

```bash
bash scripts/run_default.sh
bash scripts/run_npy.sh
bash scripts/run_bin.sh
bash scripts/run_full_signal.sh
```

## Что сохраняется в `results/`

- `demodulated_signal.npy` — демодулированный сигнал;
- `hsync_starts.npy`, `vsync_starts.npy`, `vsync_ends.npy`, `vsync_intervals.npy` — синхроимпульсы;
- `sync_report.json` — отчёт по синхронизации;
- `reconstruction_report.json` — отчёт по восстановлению кадров;
- `sync_preview.png`, `window_*` — обзорные графики;
- `reconstructed_frame_*.png` — восстановленные кадры;
- `reconstructed_frames.mp4` — склейка кадров в видео (если установлен `ffmpeg` и не задан `--video-fps 0`);
- в `reconstruction_report.json` при успешной сборке кадров могут быть поля `grayscale_p1` / `grayscale_p99` (общая нормализация яркости по всем кадрам).

## Модули `src/atv`

- `signal_processing.py` — загрузка IQ и базовые DSP-утилиты;
- `sync_detection.py` — поиск HSYNC/VSYNC;
- `reconstruction.py` — восстановление кадров;
- `visualization.py` — сохранение графиков и изображений;
- `pipeline.py` — объединение этапов в единый конвейер.
