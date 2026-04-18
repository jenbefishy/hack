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

### Через CLI

```bash
python3 src/main.py --input data/iq_capture.cf32 --fs 20000000 --out results/default_run
```

Полный проход по сигналу (без лимита кадров):

```bash
python3 src/main.py --input data/iq_capture.npy --fs 20000000 --max-frames 0 --out results/full_signal_run
```

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
- `reconstructed_frame_*.png` — восстановленные кадры.

## Модули `src/atv`

- `signal_processing.py` — загрузка IQ и базовые DSP-утилиты;
- `sync_detection.py` — поиск HSYNC/VSYNC;
- `reconstruction.py` — восстановление кадров;
- `visualization.py` — сохранение графиков и изображений;
- `pipeline.py` — объединение этапов в единый конвейер.
