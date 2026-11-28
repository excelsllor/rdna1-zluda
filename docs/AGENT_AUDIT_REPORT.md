# Агентный анализ ZLUDA/RDNA1 — 2026-11-28

Метод: 6 агентов-исследователей (функционально: работает / сломано / можно
реализовать; инфраструктурно: работает / сломано / можно реализовать) + 7-й
агент-верификатор, который перепроверил каждый пункт по файлам репозитория,
`bench/results/*.json`, логам и первоисточникам в интернете.

Итог верификации: **77 из 90 пунктов перепроверены** по коду/JSON/логам/вебу;
13 пунктов помечены «по отчётам» (нет артефакта замера). Найдено **12
расхождений** между доками и фактическим состоянием (см. §8) и несколько
противоречий doc↔doc/doc↔JSON (см. §9). Два бага обвязки и один методический
баг исправлены сразу после анализа (см. §10).

Конфигурация: RX 5700 XT (gfx1010:xnack-, 8 ГБ), Windows; ZLUDA v7-preview.21
+ патчи форка, ROCm 10.1 user space (HIP 7.16), rocBLAS 7.1.1, torch
2.7.1+cu118, Triton 3.3.1.

## 1. Полностью работает (функционально)

| пункт | доказательство |
|---|---|
| PyTorch eager/inference на gfx1010 | `bench/results/infra/probe.json`; eager_math prefill 4611 tok/s, decode 20.9 tok/s (`model_bench-20261127-172021.json`) |
| GEMM fp32/fp16/bf16 (rocBLAS 7.1.1 + gfx1010 Tensile) | `regression-20261124-205502.json`: 8.16 / 6.91 / 4.40 TF; torch_mm median 8.19 TF |
| Triton FA fwd+bwd (маски/дропаут/окно/varlen) | `fa_mask-*`, `fa_dropout-*`, `fa_window-*`: err 2.4e-4..9.8e-4, grad ≤3.9e-3, окно 7.1x |
| Flash-decoding split-KV + GQA + автодиспетч | `flash_decode-20261125-111459.json`: 3.25–4.52x |
| Fused Triton-ядра (RMSNorm/LayerNorm/RoPE/SwiGLU/GeGLU/softmax/AdamW/CE) | `fused-20261125-130920.json`: RMSNorm 9.43x, RoPE 10.87x, AdamW 3.41x |
| int4/int8 weight-only + fused dequant GEMV | `quant-20261125-122852.json`: int4 2.09x, err 2.4e-4 |
| fp16-accumulation (opt-in, HH-ядра) | `fp16_ux-20261125-122741.json`: 13–16.5 TF; `fp16_math_acc1` 7449/51.0 tok/s |
| cuFFT (hipFFT), cuSPARSE (rocSPARSE), cublasSgemv | `probe.json`: round-trip err 0.0, spmm err 0.0, sgemv OK |
| CUDA-graph capture/replay + GraphDecoder | `graph_decode-20261125-222846.json`: 11.42 мс/ток (87.6 tok/s) vs eager 39.9; ctx 2k 72.8 tok/s |
| Эко-шимы flash_attn/xformers/bitsandbytes | `shim_smoke-20261126-090559.log`: 4/4 OK |
| llama.cpp custom compute_86-PTX, F16, `-fa 0`, `-ngl 99` | `llamacpp_zluda_vs_vulkan.json`: prefill 6144/3294/732, decode 184.2/184.6/183.9 tok/s @2k/8k/32k |
| Обучение eager (fwd+bwd) | `train_suite-20261126-091359.json`: 7/7 |
| q8/q4 (dequant→fp32) в PyTorch | `maxctx_matrix-20261127-190529.json`: max ctx 4096, prefill 2847 tok/s |

Отдельно: MMVCServerSIO (RVC) на ZLUDA — работает по отчёту `docs/MMVC_ZLUDA.md`.

## 2. Не работает / работает с проблемами (функционально)

| пункт | статус | доказательство |
|---|---|---|
| PyTorch fast SDPA (flash/mem-efficient/cudnn) | падает, иногда клинит драйвер | `docs/COMPAT_MATRIX.md`; `first_fail` «unspecified launch failure»; guard в `zluda_ops/auto.py` |
| fp64 BLAS и linalg (Dgemm/Dgemv, cuSOLVER) | не поддерживается | `probe.json` cusolver_linalg FAIL; нет f64 Tensile для gfx1010 |
| torch.compile decode (StaticCache) | падает | InductorError Triton Error [CUDA]; dynamic decode 0.267 tok/s |
| llama.cpp `-fa 1` | падает (named symbol not found) | `docs/COMPAT_MATRIX.md:45`; причина требует перепроверки (см. §8) |
| llama.cpp Q4_K_M/Q8_0 (MMQ) | зависание драйвера | `docs/LLAMACPP_VULKAN_VS_ZLUDA.md`; тестировать только с готовностью к ребуту |
| fp16 conv2d медленнее fp32 | обход: cast-to-fp32 Winograd 1.4–4.25x | `conv-20261125-105059.json` |
| ComfyUI e2e 0.5 MP | не достигнут | API не поднялся за 200 с; custom-nodes падают на `comfy_env` |
| bitsandbytes под ZLUDA | ядра не работают | заменено шимом `integrations/bitsandbytes` + `optim8bit` |
| bf16 native | медленно: 3450/11.1 tok/s | `cross_backend_summary.json` |
| fused sampler в hot loop | корректен, не оптимизирован | `phase2_kernels.json`: 429.7 мкс (с top-k масками) |
| int8 fused prefill | медленнее dequant-once | 462.5 vs 235.5 мкс |

## 3. Не реализовано, но реализуемо (функционально)

| пункт | приоритет | что нужно |
|---|---|---|
| Загрузка GPTQ/AWQ (nibble/zero-point/act-order) | P1 | расширить `zluda_ops/quant.py` (сейчас только RTN) |
| Проводка fused QKV+RoPE+KV-append и GQA-decode в GraphDecoder | P1 | ядра готовы (`zluda_ops/qkv_fused.py`, `gqa_decode.py`), нужна интеграция в слой |
| Persistent/fused decode-ядро, multi-row GEMV | P1 | новый Triton-кернел; по отчётам до +30% decode |
| MIOpen RNN-descriptor mapping | P1 | `cudnnCreateRNNDescriptor`→MIOpen; разблокирует RMVPE/GRU и ORT CUDA EP |
| llama.cpp FA через non-TC PTX / `GGML_CUDA_FORCE_CUBLAS=1` для Q4 | P1 | гипотезы, требуют перепроверки на текущем форке |
| torch.compile для decode | P1 | StaticCache + GEMV-путь; по отчётам +1.5–2x |
| ComfyUI минимальный профиль (704×704, GGUF+flux2k) | P1 | фикс `comfy_env`, отключение sam3 |
| HIP GEMV-библиотека с интеграцией в stream/graph | P2 | прототип `tools/hip_proto/gemv_fp16.cpp`; harness отсутствует |
| Paged attention / vLLM, квантование KV-кэша | P2 | крупные проекты |
| cuSOLVER mapping/CPU-fallback | P2 | есть только для fp64-путей |
| conv1d/conv3d/convTranspose, pooling 3d | P2 | не проверено |
| Нативный ROCm bitsandbytes | P2 | официально поддерживает gfx101X в ROCm-сборке |

## 4. Инфраструктурно работает без проблем

| пункт | доказательство |
|---|---|
| Сборка/запуск форка на Windows | `docs/RDNA1_BUILD_AND_USAGE.md`; `run-zluda.cmd` (ZLUDA_CC=8.6, Tensile override, кэши) |
| Версии стека | torch 2.7.1+cu118, Triton 3.3.1, HIP 7.16 (перепроверено в venv) |
| Runtime basics (alloc/copy/streams/events/memory stats) | `probe.json` OK (pinned — см. §5) |
| Приёмочные прогоны Phase 1 | `tools/run_infra_suite.ps1`: probe 21/25, train 7/7, shim 4/4 — PASS |
| Протокол измерений и отчётность | `bench/common.py` save_report; 171 JSON в `bench/results` |
| Автодиспетч с fallback и env-переключателями | `zluda_ops/auto.py`; `INTEGRATION.md` |
| Лаунчеры и watchdog (kill-only) | `run-zluda*.cmd`, `tools/gpu_watchdog.ps1` |
| llama.cpp PTX-сборка | `C:\build\llama.cpp\build-ptx86`, `86-virtual` |
| Эко-шимы и установщики | `tools/install_ecosystem.py`, `integrations/*`, shim smoke 4/4 |
| Диагностика нереализованных API | `ZLUDA_DEBUG_UNIMPLEMENTED=1`, крейты `zluda_trace*` |
| Патч-менеджмент | `tools/export_patch.ps1`/`check_patch.ps1`, патч 1.0 МБ |

## 5. Инфраструктурно не работает / работает с проблемами

| пункт | статус | доказательство |
|---|---|---|
| Стабильность драйвера | клинит (ребут) | SDPA-fast `unspecified launch failure`; llama.cpp Q4 hang; driver-restart ломал Windows |
| Телеметрия клоков/питания | недоступна | `clocks.errors` -8 во всех JSON; NVML NotSupported → разброс ±15–30% |
| GPU-CI и повторяемость | нет | одиночные прогоны, GPU-CI отсутствует |
| CUPTI/torch.profiler | пустые трейсы | по отчётам |
| Автозагрузка fallback'ов BN/RNN | **исправлено 2026-11-28** | не была подключена в `auto.py`; добавлен `cudnn_fallback.autoload()` |
| Подключение автотюн-таблицы GEMV | **исправлено 2026-11-28** | `gemv._config` не читал `tuned_gemv_ext`; добавлен merge |
| Проверка корректности CUDA-графов | **исправлено 2026-11-28** | сравнение шло до replay; добавлен `g.replay()` |
| `auto.uninstall()` | **исправлено 2026-11-28** | был объявлен, но не реализован |
| GraphDecoder: позиция/токен вне графа | ограничение | self-loop ~1 tok/s |
| Pinned memory | не подтверждена | `probe.json` pinned=false; ComfyUI/llama.cpp вынуждены отключать |
| Инфраструктура ComfyUI | не доводится до API | API за 200 с не поднялся; `comfy_env` |
| MIOpen system perf-db | холодный старт | `miopen_db-*.json`: init 1180 мс |
| Качество данных rematrix | метки/выбросы | «sdpa_fast» = math; выбросы F16/q8 |

## 6. Инфраструктурно можно реализовать

| пункт | приоритет | обоснование |
|---|---|---|
| Safe-run режим (watchdog по умолчанию, без fast SDPA, изоляция процессов, скрипт восстановления) | P0 | частые зависания драйвера |
| Исправленная автозагрузка fallback'ов + merge автотюна + graph-replay тест | P0 | уже сделано 2026-11-28 |
| GPU-CI на self-hosted gfx1010 + медианы нескольких прогонов + regression-gates | P1 | `bench/regression.py` есть |
| Клоки/телеметрия: ADL Overdrive8/ADL2 либо статистический протокол | P1 | `bench/gpu_clocks.py`; разброс ±15–30% |
| llama.cpp: non-TC PTX (61/70-virtual) и/или fallback MMA + FORCE_CUBLAS для Q4 | P1 | web: fattn.cu/mmq.cu |
| Профайлинг через `zluda_trace*`/HIP-events вместо CUPTI | P2 | CUPTI пуст |
| Pinned memory (`cudaHostAlloc`/`Register`, `is_pinned`) | P2 | `zluda/src/impl/memory.rs` |
| MIOpen system perf-db | P3 | выигрыш не доказан |
| Прочие API: cuGraph/cuMem/cuBLASLt f64/cuDNN RNN/cuSPARSE форматы/cuFFT планы | P2 | по списку `docs/KERNEL_GAPS.md` |

## 7. Итог и приоритеты (top-10)

Стек в целом работоспособен: eager PyTorch, GEMM (84% пика fp32), Triton
FA/flash-decoding, квантование, CUDA-графы, обучение, llama.cpp F16. Главные
функциональные дыры: fast SDPA, fp64, torch.compile-decode, llama.cpp FA/MMQ,
ComfyUI e2e. Главные инфраструктурные риски: зависания драйвера, незафиксированные
клоки/одиночные прогоны, нет GPU-CI/профайлера, ошибки обвязки (уже частично
исправлены) и неверная проверка графов (исправлена).

1. **P0** — стабилизация драйвера и safe-run режим.
2. **P0** — (сделано 2026-11-28) автозагрузка `cudnn_fallback`, merge `TUNED_M1_EXT`, корректный graph-replay тест, `uninstall()`.
3. **P1** — llama.cpp: non-tensor-core PTX / fallback MMA + `GGML_CUDA_FORCE_CUBLAS=1` (FA и Q4/Q8).
4. **P1** — torch.compile для decode (StaticCache + GEMV-путь).
5. **P1** — проводка fused QKV/GQA-decode и multi-row GEMV в GraphDecoder.
6. **P1** — ComfyUI e2e: минимальный набор нод, 704×704, 1–4 шага.
7. **P1** — MIOpen RNN-descriptor mapping (RMVPE/GRU, ORT CUDA EP).
8. **P1** — GPTQ/AWQ-совместимая загрузка (7B int4 на 8 ГБ).
9. **P1** — профайлер + статистический протокол + GPU-CI/медианы.
10. **P2** — нативный ROCm bitsandbytes, cuSOLVER fallback, pinned memory.

Аппаратно невозможно на gfx1010: int8 dot/int8 tensor units и MFMA/WMMA tensor
cores. `mma.sync` при этом транслируется software-fallback'ом (upstream PR #551),
т.е. это не аппаратный запрет.

## 8. Корректировки (расхождения доков с фактами)

| # | утверждение в доках | факт | что делать |
|---|---|---|---|
| 1 | `zluda_ops.auto` ставит fallback BN/RNN | не ставил; fallback вызывался только вручную в bench | **исправлено**: `cudnn_fallback.autoload()` в `install()` |
| 2 | автотюн GEMV подключён | `gemv._config` не читал `tuned_gemv_ext` | **исправлено**: импорт/merge |
| 3 | GraphDecoder logits bit-for-bit | бенч сравнивал буфер до replay (diff 10.85/7.71 — шум буфера) | **исправлено**: replay перед сравнением |
| 4 | llama.cpp FA невозможен из-за mma.sync на RDNA1 | PR #551 даёт software-fallback MMA (есть в форке); реальная причина — конкретный нереализованный вариант ядра | переклассифицировать, снять актуальную ошибку |
| 5 | Phase-2 числа: GQA 1.34x, add_rms_norm 1.37x | в JSON 1.24x и 1.48x; в roofline 31.6 мкс | приводить диапазон/артефакты |
| 6 | Regression 12/12 | в JSON 18 проверок, passed=18 | использовать 18/18 |
| 7 | cublasSgemv не реализован | реализован 2026-11-26 (probe OK) | **поправлено в KERNEL_GAPS** |
| 8 | flash_attn shim decode сломан (1.6 tok/s) | причина — GQA; после фикса shim_smoke OK, скорость не перемерена | статус «fixed, perf unmeasured» |
| 9 | `bench/hip_gemv_proto.py` | файла нет; есть `tools/hip_proto/*` | добавить harness или поправить доку |
| 10 | Phase-1 probe 20/24 | артефакт: 21/25, run_infra_suite PASS | **поправлено в INFRA_FIXES** |
| 11 | rematrix «sdpa_fast» — отдельный бэкенд | fast-бэкенды принудительно выключены, sdpa=math | переименовать в «sdpa_default (fast off)» |
| 12 | pinned memory работает | `pinned=false` в probe; ComfyUI/llama.cpp отключают | проверить/реализовать `cuMemHostAlloc` |

## 9. Ключевые противоречия doc↔doc / doc↔JSON

- `INTEGRATION.md` и `ENGINE_COMPARISON.md` одновременно содержат «llama.cpp
  заблокирован» и «llama.cpp работает (custom PTX, F16, fa=0)» — актуален второй.
- `KERNEL_GAPS.md` (sgemv) ↔ `INFRA_FIXES.md`/`probe.json` — актуально «работает» (исправлено).
- `COMPAT_MATRIX.md`/`CROSS_BACKEND_COMPARISON.md` (flash shim 1.6 tok/s) ↔
  `shim_smoke` (OK) — актуально «исправлено, скорость не перемерена».
- `ENGINE_COMPARISON.md` «CUDA graphs are not supported» ↔ `CUDA_GRAPHS_ZLUDA.md`
  и graph_decode JSON (graph_ok=true) — устаревшая секция.
- `ROADMAP.md` «bf16-emul ещё падает на ZLUDA (VRAM)» ↔ «работает 5488/29.6
  (раньше падал harness)» — актуально второе.
- `INFRA_FIXES.md` 20/24 ↔ `ROADMAP_ECOSYSTEM.md`/probe.json 21/25 (исправлено).
- `RESULTS_RDNA1.md`/`BEFORE_AFTER.md` 12/12 ↔ JSON 18/18.
- `PHASE2_KERNELS.md` числа ↔ `phase2_kernels.json`/`roofline.json`.
- `RDNA1_BUILD_AND_USAGE.md` «ComfyUI verified» ↔ `COMPAT_MATRIX.md` (e2e не выполнен).
- rematrix: F16 при запросе ctx 32768 фактически 8192 с prefill 5517; q8 8192
  decode 45.9 — выбросы одного прогона.

## 10. Что уже исправлено по итогам анализа (2026-11-28)

- `zluda_ops/auto.py`: автозагрузка `cudnn_fallback` (BN/RNN), реализован
  `uninstall()` (раньше был только в `__all__`).
- `zluda_ops/gemv.py`: `_config` теперь мёржит `TUNED_M1_EXT`.
- `bench/graph_decode.py`: перед сравнением logits выполняется `g.replay()`.
- Документация: поправлены `INFRA_FIXES.md` (21/25), `KERNEL_GAPS.md` (sgemv),
  `COMPAT_MATRIX.md`/`ENGINE_COMPARISON.md`/`INTEGRATION.md`/`ROADMAP.md` —
  устаревшие утверждения про llama.cpp/графы/bf16-emul помечены/обновлены.
- CPU-смок: `install()/uninstall()` roundtrip, импорт `TUNED_M1_EXT`, py_compile.

## 11. Веб-источники верификации

- https://github.com/vosen/ZLUDA/releases/tag/v7-preview.21 — база форка.
- https://raw.githubusercontent.com/vosen/ZLUDA/master/docs/src/faq.md — поддержка RX 5000+, статус PyTorch.
- https://github.com/vosen/ZLUDA/pull/551 — software-fallback MMA для RDNA1/RDNA2 (merged 2025-11-20).
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html — gfx1010 отсутствует в официальной матрице.
- https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md — gfx1010 есть, Release Ready.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/fattn.cu — выбор FA-ядра по tensor cores.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmq.cu — отключение MMQ при `GGML_CUDA_FORCE_CUBLAS`.
- https://rocm.docs.amd.com/projects/MIOpen/en/latest/doxygen/html/group__batchnorm.html — MIOpen batchnorm API существует.
- https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md — поддержка RDNA gfx101X в ROCm-сборке.
- https://github.com/excelsllor/rdna1-zluda — публичный форк.

Примечание: в среде верификатора `web_search` вернул HTTP 401, поэтому
веб-проверка сделана через `web_fetch` первоисточников.
