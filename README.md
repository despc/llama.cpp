> ## About this fork / О форке
>
> **EN.** This fork adds a `V100_CUDA` backend, so a Tesla V100 can serve a model
> together with consumer GPUs inside a single `llama-server` process.
>
> The V100 needs the proprietary NVIDIA driver, while the RTX 5080 / 5070 Ti need
> the open one. Two kernel modules already coexist; the problem is user space.
> Both drivers ship a `libcuda` with the same SONAME, so the dynamic loader keeps
> only one copy and whichever loads first ends up serving every GPU. This fork
> copies the V100 driver stack under distinct `libcv*` names, builds a second CUDA
> backend against it, and keeps its symbols private. A small `LD_PRELOAD` shim
> sends that stack's device opens to `/dev/nvidia-v100-*`.
>
> The result is a normal local device, `V100_CUDA0`, usable with `--device` and
> `--tensor-split` next to `CUDA0` / `CUDA1`. It replaces the older workaround of
> running the V100 behind an RPC server in a separate process.
>
> Everything lives in [`ggml/src/ggml-v100-cuda/`](ggml/src/ggml-v100-cuda/) - see
> its README for build and run instructions. Changes to shared llama.cpp code are
> deliberately kept to a handful of additive lines.
>
> **RU.** Форк добавляет бэкенд `V100_CUDA`: Tesla V100 работает над моделью
> вместе с потребительскими картами в одном процессе `llama-server`.
>
> V100 требует проприетарный драйвер NVIDIA, а RTX 5080 / 5070 Ti - открытый. Два
> модуля ядра уживаются и так, сложность в пользовательском пространстве. У обоих
> драйверов `libcuda` с одинаковым SONAME, поэтому загрузчик оставляет только одну
> копию, и первый загрузившийся драйвер обслуживает все карты. Форк копирует
> драйверный стек V100 под именами `libcv*`, собирает против него второй CUDA
> бэкенд и прячет его символы. Небольшая `LD_PRELOAD` библиотека направляет
> обращения этого стека к `/dev/nvidia-v100-*`.
>
> В итоге получается обычное локальное устройство `V100_CUDA0`, доступное в
> `--device` и `--tensor-split` наравне с `CUDA0` / `CUDA1`. Это заменяет прежний
> обходной путь, когда V100 работала через RPC-сервер в отдельном процессе.
>
> Всё лежит в [`ggml/src/ggml-v100-cuda/`](ggml/src/ggml-v100-cuda/), там же README
> со сборкой и запуском. Правки в общем коде llama.cpp намеренно сведены к
> нескольким добавленным строкам.
>
> ---
>
> ## Что здесь ускоряли
>
> Четыре карты разной скорости считают одну модель вместе: RTX 5080, RTX 5070 Ti
> и две Tesla V100. Прямой связи между ними нет — теслы живут за отдельным
> драйверным стеком, — поэтому каждый байт, которым карты обмениваются, дважды
> пересекает PCIe через оперативную память хоста. На Qwen3.8-27B в тензорном
> режиме на этот обмен уходило около 90% времени префила: узким местом были не
> вычисления, а пересылки.
>
> Результат на одном и том же запросе (1430 токенов на входе, 200 на выходе):
>
> | | было | стало |
> |---|---:|---:|
> | префил | 187 т/с | **630 т/с** |
> | генерация | 63 т/с | **78 т/с** |
>
> Всё, что осталось включённым, проверено поэлементным сравнением с эталонной
> свёрткой — 965 миллионов элементов на карту, ноль расхождений. Ничего из
> ускорений не меняет арифметику: точность не разменивалась ни разу, это было
> условием работы.
>
> ### Что именно сделано
>
> **Дерево сложения как у эталона.** Свёртка складывает вклады карт в том же
> порядке и с тем же округлением, что и штатный мета-бэкенд. Это не ускорение
> само по себе, но благодаря ему любое дальнейшее изменение можно проверять
> побитовым сравнением, а не «на глаз».
>
> **Reduce-scatter вместо полной свёртки у всех.** Раньше каждая карта тянула к
> себе вклады всех остальных и складывала всё целиком. Теперь каждая сводит
> только свою долю, а потом карты обмениваются готовыми долями. Трафика 2.75N
> вместо 4N.
>
> **Публикация порциями.** Карта не ждёт, пока запишет весь свой вклад, а
> объявляет его частями — соседи начинают читать раньше.
>
> **Своя доля не публикуется.** Область, которую карта сводит сама, никто у неё
> не читает, а раньше она всё равно отправлялась по шине.
>
> **Неравные доли свёртки.** Теслы медленнее на обмене, поэтому сводят меньшую
> часть: 35/35/15/15 вместо поровну.
>
> **Участие по слоям — самое крупное.** Раньше каждый слой резался на все четыре
> карты, и каждый слой заканчивался обменом между всеми четырьмя. Теперь
> большинство слоёв делит между собой только пара блэкволлов, а теслы владеют
> двенадцатью слоями целиком — вдвоём на слой. Карта без доли в слое не публикует
> ничего, её никто не ждёт, и она не участвует в свёртке этого слоя. Только это
> дало префилу прирост с 370 до 630 т/с.
>
> **Какие слои отдавать теслам, решает память, а не скорость.** Слой стоит около
> 0.45 ГиБ, а слой с KV-кешем — ещё гигабайт при большом контексте, и KV есть
> только у каждого четвёртого. Поэтому теслам отдаются именно слои с KV: так
> блэкволлы, у которых всего по 16 ГиБ, помещают 49 слоёв вместо 40.
>
> ### Что попробовали и отвергли
>
> Примерно половина работы — измеренные отказы. Они записаны так же подробно,
> как и удачи, чтобы к ним не возвращались по кругу:
>
> - **Дуплекс** — гонять обмен в обе стороны разом. Линия это умеет (замерено,
>   1.34x), но чтобы этим воспользоваться, коллектив надо резать на части, а
>   каждая часть добавляет круг синхронизации между картами. На картах, которые
>   различаются в 1.7 раза по скорости, круг стоит дороже, чем даёт перекрытие.
> - **Прямой обмен между картами (P2P)** — у тесл 1.38 ГБ/с против 3.29 через
>   хост, у блэкволлов недоступен вовсе.
> - **Больше блоков в ядре свёртки** — ноль пользы: каждая фаза и так насыщает
>   линию в своём направлении.
> - **Пропуск слоёв на картах, которые в них не участвуют** — реализовано,
>   корректно, убирает 3.35 ГиБ обмена из 3.81, и всё равно медленнее: передача
>   готового выхода слоя требует своей точки синхронизации, а она дороже.
> - **Перестановка карт по слотам** — расчёт дал ровный ноль.
>
> Общий вывод из всех отказов один: этот обмен упирается не в то, **сколько**
> карты пересылают, а в то, **как часто им приходится договариваться**.
>
> ### Где что лежит
>
> Подробный дневник со всеми числами и неудачами —
> [`docs/backend/CUDA-mixed-runtime-allreduce.md`](docs/backend/CUDA-mixed-runtime-allreduce.md).
> Настройки запуска (доли, размещение слоёв, контекст) — в пусковых скриптах
> рядом с бинарём, там же записаны замеры, по которым числа выбирались.
>
> Полезное для проверки: `GGML_CUDA_MIXED_AR_VERIFY=1` сверяет каждый коллектив
> с эталонной свёрткой поэлементно, `GGML_CUDA_MIXED_AR_PROFILE=1` показывает,
> куда уходит время и сколько байт пересылается по видам обмена.
>
> ## Вторая модель: Qwen3.8-Flash-Next
>
> Она работает иначе — слои раздаются картам целиком, а не режутся между ними,
> поэтому ничего из описанного выше к ней не относится: обмена между картами
> почти нет, и узкое место другое. Модель разрежённая (MoE), и почти всё время
> префила уходит на матрицы экспертов.
>
> Здесь выигрыш дали не пересылки, а **выбор ядер под Volta**. У V100 нет
> тензорных инструкций, которые появились в Turing, а правила выбора ядра в
> llama.cpp их наличие подразумевают. Из-за этого теслы сваливались в запасные
> пути, заметно худшие.
>
> Три исправления, все — про выбор ядра, ни одно не меняет арифметику:
>
> **Групповой квантованный `MUL_MAT_ID`.** Каждая проекция эксперта на префиле
> уходила в запасной путь, синхронизирующийся с хостом: 284 тысячи запусков ядер
> на один префил в 5000 токенов, и CUDA-граф при этом не собирался вовсе.
> 478 → 613 т/с.
>
> **Правильная таблица тайлов DP4A.** Для Volta выбиралась таблица, рассчитанная
> на другую раскладку вычислений. 613 → 811 т/с.
>
> **Тайл по столбцам на эксперта.** При 512 экспертах каждому в микробатче
> достаётся горстка токенов, а тайл рассчитывался на весь батч — то есть считался
> в основном на пустом месте. Ограничено раскладкой DP4A: на блэкволловской MMA
> то же правило почти весь выигрыш съедало. 811 → 907 т/с.
>
> Вместе на развёрнутой сборке при промпте 10 тысяч токенов: **460 → 830 т/с**
> префила, то есть в 1.8 раза. На генерацию не влияют совсем — декод идёт другим
> путём, где это правило не работает.
>
> **Вывод байт в байт совпадает** с версией без этих исправлений — проверено
> трижды, включая прямое A/B на одном промпте с генерацией в 3000 токенов. Именно
> поэтому их и вернули: правило форка исключает всё, что может переставить
> арифметику, а здесь переставлять нечего. Флаги
> `GGML_CUDA_MMID_MMQ_PREFILL=0` и `GGML_CUDA_MMQ_MMID_J_FIT=0` позволяют
> проверить это заново. У третьего исправления флага нет и быть не может: хост
> выбирает конфигурацию запуска, а устройство — по `__CUDA_ARCH__` на этапе
> компиляции, и переключатель во время работы их бы рассинхронизировал.
>
> ### Что отвергли здесь
>
> - **Компактное внимание** — давало 23% на 100 тысячах токенов и 34% на 150
>   тысячах, ускоряло и декод. Удалено: оно меняло численные результаты и
>   выдаваемый текст. Разница перплексии 0.048% не является границей потери
>   качества, а совпадение 21 из 24 позиций декода ничего не гарантирует для
>   остальных предсказаний. Правило простое: вариант, способный ухудшить точность
>   хотя бы теоретически, не рассматривается.
> - **Микробатч 640** — примерно 7% префила ценой 4% генерации после 150 тысяч и
>   лишних 284–350 МиБ на карту. Оставлен 512.
> - **Два слота конвейера копирования** — выигрыш около 0.5%, при этом запрос на
>   30 тысяч токенов не помещался в память.
> - **FP16 для экспертов** — выигрыша не показал, а проблему, ради которой
>   затевался, закрыли исправления выравнивания и диспетчеризации.
>
> Прототип компактного внимания попутно вскрыл настоящие ошибки — в размере
> выделения, форме эталонной маски, шаге индекса, времени жизни захваченного
> буфера и в проверке на бесконечности. Их исправили до удаления. Урок, который
> из этого вынесли: проверять промежуточные данные и поведение захвата графа, а
> не доверять правдоподобному выводу и скорости.
>
> Подробности и полная кривая по длинам контекста —
> [`docs/backend/CUDA-flash-next-prefill-plan.md`](docs/backend/CUDA-flash-next-prefill-plan.md).

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
