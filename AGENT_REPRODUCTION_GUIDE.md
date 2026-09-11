# 丝桐里 App 授权逆向采集复现手册

本文档面向接手本仓库的其它 agent。目标是在已授权登录状态下，复现“非本人原创/克隆/公开谱面”的结构化正文采集能力，输出 `raw_data.json` 与 `data.json`。本流程只面向当前账号可合法访问、可打开或可编辑的谱面；不要用于绕过账号权限、批量抓取无授权内容或传播第三方受保护数据。

## 1. 结论先行

真实正文来源不是“我的 -> 内容”列表里的元数据，也不是 OCR。

当前可复现的稳定入口是：

```text
已登录 Gadget 版 App
  -> 打开 sitongli://app/scores/<score_id>
  -> App 请求/解码 https://s.sitongli.net/v2/scores/<score_id>/data
  -> Flutter/Dart AOT 模型进入谱面渲染函数
  -> Frida 在 0x63f220 捕获窗口切分前的 source_jab_event 完整列表
  -> extract_score_runtime_windows.py 直接重建 notes/jians/lyric
  -> extract_jianpu_jianzi.py 生成简谱—减字谱映射与 ABC notation
  -> 若 source 列表未触发，再回退到 note_slur WZa 窗口拼接
```

旧的普通内存 JSON 扫描只能稳定拿到缓存里的元数据或已打开过的明文候选，不能保证拿到正文。非本人谱面应优先走 `runtime` 模式。

## 2. 关键工程位置

```text
cases/sitongli-guanshanyue/
  webapp/
    server.js                         Web 控制台后端
    public/index.html                 控制台主页
    public/non-owner-runtime.html     采集原理页
    public/agent-reproduction-guide.html  本手册的可视化版本
  scripts/
    collect_score_runtime.ps1         runtime 主采集入口
    frida_decode_score_jians.py       Frida Dart AOT hook 与对象解码
    extract_score_runtime_windows.py  从 jsonl 重建 raw_data/data
    extract_jianpu_jianzi.py         从 raw_data 生成映射 JSON、Markdown 与 ABC
    export_score_from_memory.ps1      旧 memory 回退路径
    extract_memory_json_candidates.py 内存字符串候选 JSON 提取
    normalize_score_export.py         旧 memory 路径标准化
  work/blutter_out/blutter_frida.js   blutter 生成的 Dart 对象解码辅助
  batch/<slug>/out/
    raw_data.json
    data.json
```

## 3. 环境前提

- Windows + PowerShell。
- Android 手机开启 USB 调试并授权当前电脑。
- 已安装并登录 Gadget 版 App，包名默认 `com.sitongli.app.gadget`。
- `adb` 默认路径：`C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe`。
- Python 可用，且已安装 `frida` 包。
- 当前仓库已经包含 `tools/frida/libfrida-gadget.so`、`tools/apktool.jar`、`tools/jadx` 和 `work/blutter_out/blutter_frida.js`。

检查设备：

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" shell pidof com.sitongli.app.gadget
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" forward tcp:27042 tcp:27042
```

如果 `pidof` 为空，先手动打开 Gadget 版 App。

## 4. 从 Web 系统复现

启动 Web 控制台：

```powershell
powershell -ExecutionPolicy Bypass -File cases\sitongli-guanshanyue\webapp\start.ps1
```

打开：

```text
http://127.0.0.1:8787
```

单曲采集：

1. 在“曲目清单”新增一行。
2. 填入 `title`、`score_key`、`score_id`，并按谱面标题处的 `1=X` 填写简谱调号（例如 `C`）。
3. 克隆谱建议填 `from_key/from_id`；公开谱可以留空。
4. 模式选择 `runtime`。
5. 点击“采集”。

Web 后端会调用：

```text
POST /api/capture
  -> startJob()
  -> powershell scripts/collect_score_runtime.ps1
```

完成后结果在：

```text
cases/sitongli-guanshanyue/batch/<slug>/out/raw_data.json
cases/sitongli-guanshanyue/batch/<slug>/out/data.json
```

## 5. 从命令行复现

推荐先手动打开目标谱面，确认 App 能正常进入详情或谱面正文页。然后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File cases\sitongli-guanshanyue\scripts\collect_score_runtime.ps1 `
  -Package com.sitongli.app.gadget `
  -ScoreKey SWDFmDZX `
  -ScoreId 161749 `
  -FromKey SkvmkgjX `
  -FromId 155861 `
  -Title 秋风词 `
  -Tonic C `
  -Slug qiufengci `
  -RootDir cases\sitongli-guanshanyue `
  -LaunchDeepLink `
  -AutoTap `
  -WarmupSeconds 4 `
  -ScrollPasses 8 `
  -PostGestureSeconds 10
```

参数说明：

- `-LaunchDeepLink`：通过 `sitongli://app/scores/<score_id>` 打开目标谱。
- `-AutoTap`：进入详情页后自动点击谱面入口，触发正文渲染。
- `-ScrollPasses`：通过滑动扩大渲染窗口覆盖面。
- `-FromKey/-FromId`：克隆谱来源信息，仅用于保留 provenance，不是 runtime hook 的必要条件。
- `-NotesLength`：来自元数据的长度字段，用于完整性辅助判断；它不是事件数。
- `-Tonic`：谱面显示的首调简谱调号，例如谱面写 `1=C` 就传 `C`。它是必填项。

注意：`正调`、`蕤宾调`等描述古琴定弦；`1=C`、`1=F`描述首调简谱调号。
二者是独立字段，不能根据“正调”推断 ABC 的 `K:`。

## 6. runtime 主路径的内部原理

`collect_score_runtime.ps1` 做四件事：

1. 建立 Gadget 转发：`adb forward tcp:27042 tcp:27042`
2. 启动 Frida hook：

```text
python scripts/frida_decode_score_jians.py
  --host 127.0.0.1:27042
  --out <batch>/<slug>/evidence/runtime_note_slur.jsonl
  --only note_slur
  --only note_render
  --only view_get_jians
  --only jianzi_component
  --depth 10
  --array-limit 1024
  --map-limit 1024
```

3. 让 App 打开目标谱并滚动，触发渲染。
4. 调用 `extract_score_runtime_windows.py` 输出 JSON。

### Hook 入口

当前版本的核心偏移在 `frida_decode_score_jians.py` 的 `TARGETS` 中：

```text
note_slur_eJk_hab_63f748        0x63F748
note_render_uJk_663eb8          0x663EB8
view_get_jians_XIk_QZa_65ec34   0x65EC34
view_get_jians_lJk_xab_67444c   0x67444C
view_get_jians_SIk_FZa_6745cc   0x6745CC
jianzi_component_nLa_GYi_5fd2a8 0x5FD2A8
score_data_decode_mMk_qgb       0x6C2B60
score_dataops_fkk_Lz            0x6C53E4
score_patch_jkk_Pz              0x6C59D0
```

首选正文入口是父构造函数内部的 `0x63f220`，采集名称为
`source_jab_event`。此时 `x0` 是窗口切分前的完整 `List<jab>`：

- `jab.off_8`：简谱音符对象。
- `jab.off_c`：减字谱组件。
- `jab.off_10/off_14/off_18`：歌词字段。
- 非 `jab` 条目是换行或布局控制项，重建时保留统计但不作为音符。

该入口能保留曲中合法重复事件，也不会遗漏 WZa 窗口边界音。若新版本
App 导致该内部偏移失效，再回退到 `note_slur_eJk_hab_63f748`。其
`onEnter` 寄存器含义为：

- `x1`：单个 note 参数，可恢复简谱音符。
- `x7`：WZa 渲染窗口对象，`off_8` 中包含窗口内的事件数组。

### Dart 对象清洗与字段映射

Frida 捕获的 Dart 对象会带有类似 `ClassName!field@heapaddr` 的临时键。`extract_score_runtime_windows.py` 使用 `clean_obj()` 去掉易变 heap 地址，保留稳定字段。

正文事件映射：

```text
WZa event
  off_8   -> note object
  off_c   -> jians array
  off_10  -> lyric
  off_14  -> lyric1
  off_18  -> lyric2
  off_1c  -> section-like field
```

简谱解码：

```text
decode_note()
  duration  <- off_8
  pitch     <- off_14
  octave    <- pitch.off_c
  slurs     <- off_1c
  note text <- duration prefix + pitch digit + octave suffix
```

减字谱组件解码：

```text
decode_jian()
  jian.off_8.off_8/off_c -> component
decode_jian_component()
  enum type ZHTYX/TH/D/... -> std.tp/std.raw_fields
```

## 7. 输出文件规范

`raw_data.json` 必须保留：

```text
provenance.method
provenance.source_jsonl
provenance.api_source
metadata.score_id
metadata.score_key
metadata.from_id/from_key
metadata.tonic
raw_score_data.meter
raw_score_data.tonic
raw_score_data.tuning
raw_score_data.sections
raw_score_data.dataops
raw_score_data.notes[]
```

`data.json` 必须保留：

```text
metadata
field_notes
stats.total_events
stats.jianpu_events
stats.jianzi_events
stats.jian_count
stats.successful_alignments
stats.unaligned_event_indexes
stats.unaligned_count
stats.capture_completeness
anomalies[]
events[]
```

保留规则：

- 不做 OCR。
- 不把减字谱转成相似汉字。
- 简谱和减字谱对齐关系来自同一个 runtime event，不靠视觉推断。
- `raw_record` 必须保留，便于后续版本重新解释字段。

## 8. 自动发现与来源字段

自动发现不是正文采集路径，它只扫描当前 App 内存里的列表/缓存元数据。

Web 入口：

```text
POST /api/discover
  -> discoverScores()
  -> frida_scan_memory_strings.py
  -> extract_memory_json_candidates.py
  -> collectScoreRecords()/collectSourceRecords()
```

过滤条件：

```text
uid == OWNER_UID
options.edit !== false
```

`from_key` 为空时的修复策略：

1. 当前候选中如果有 `from_id` 对应的源谱记录，用源谱 `score_key` 回填。
2. 如果当前扫描不全，从历史 `evidence/web_discovery/candidates_*` 中按 `from_id` 查找。
3. 如果历史中存在同一 `score_key/score_id` 且带 `from_key` 的 peer record，用它回填。

注意：`from_key` 为空通常不阻止 `runtime` 采集，因为正文来自当前目标谱面的渲染事件。但克隆谱最好保留 `from_key/from_id`，方便 provenance 和后续排查。

## 9. memory 回退路径

`runtime` 失败后，`collect_score_runtime.ps1` 会回退到 `scripts/export_score_from_memory.ps1`。

该路径适用于 App 已经把完整正文 JSON 留在内存或缓存里的情况。它的局限：

- 只对打开过的谱面更可靠。
- 容易扫到列表元数据而不是正文。
- 可能被旧缓存污染。
- 非本人谱面不稳定。

判断 memory 候选是否是正文：

```text
looks_like_score_data(obj):
  obj.notes is list
  obj.sections is list
  obj has meter or tuning
```

因此其它 agent 遇到“成功但事件数为 0”时，不要继续调 memory 阈值，应回到 runtime hook 是否触发渲染。

## 10. 常见故障定位

### App 点击采集后退出重启

`runtime` 模式会通过 deep link 打开目标谱，并可能重启/拉起 Gadget 版 App。这本身可以正常发生。异常判断标准不是是否重启，而是：

- `adb shell pidof com.sitongli.app.gadget` 是否重新出现 PID。
- `runtime_note_slur.jsonl` 是否出现 `hooked` 和 `enter_decoded`。
- `extract_score_runtime_windows.py` 是否捕获 `captured_events > 0`。

### Gadget 版 App 无响应

常见原因是上一次 Frida/Gadget 会话还在监听或 Python hook 进程未退出。处理：

```powershell
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -like "*frida_decode_score_jians.py*" -or $_.CommandLine -like "*frida_dump_score_bytes.py*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" shell am force-stop com.sitongli.app.gadget
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" forward tcp:27042 tcp:27042
```

### `No runtime score events captured`

说明 hook 挂上了，但 App 没有触发正文渲染。排查：

- 手机是否停在目标谱面正文/编辑页，而不是列表页或详情页。
- 采集期间是否自动点击到了“乐谱”入口。
- 增加 `-WarmupSeconds`、`-ScrollPasses`、`-PostGestureSeconds`。
- 手动进入目标谱面正文后不加 `-LaunchDeepLink` 再跑一次。

### 显示成功但 `total_events = 0`

这不是有效成功。正确成功条件应同时满足：

```text
raw_data.json exists
data.json exists
data.stats.total_events > 0
```

### 自动发现后 `from_key` 为空

这通常是当前缓存里只有克隆谱记录，没有源谱记录。解决：

- 先在 App 中打开过来源谱或曲库详情，使源谱元数据进缓存。
- 再点自动发现。
- 或手工填 `from_key/from_id`。

`runtime` 模式仍可采集当前谱正文；`from_key` 只影响来源记录完整性。

## 11. 验证清单

一首谱面采集完成后，必须检查：

```powershell
$p = "cases\sitongli-guanshanyue\batch\<slug>\out\data.json"
$d = Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json
$d.stats
```

有效输出标准：

- `stats.total_events > 0`
- `stats.jianpu_events > 0`
- `stats.jianzi_events > 0` 或明确说明该谱无减字事件
- `stats.successful_alignments` 合理
- `events[0].raw_record` 存在
- `metadata.score_key` 与目标一致
- `metadata.tonic` 与谱面显示的 `1=X` 一致
- 克隆谱的 `metadata.from_key/from_id` 尽量保留

### 生成简谱—减字谱映射和 ABC

采集完成后运行：

```powershell
python scripts\extract_jianpu_jianzi.py `
  --raw batch\<slug>\out\raw_data.json `
  --out-dir batch\<slug>\out\extracted_data
```

调号读取优先级为 `--tonic` 高于 `metadata.tonic`。正常情况下应由采集阶段写入
`metadata.tonic`；需要纠正旧数据时可显式覆盖：

```powershell
python scripts\extract_jianpu_jianzi.py `
  --raw batch\<slug>\out\raw_data.json `
  --out-dir batch\<slug>\out\extracted_data `
  --tonic C
```

若两处都没有调号，脚本会停止并报错，不会再根据调弦猜测。输出后检查 Markdown
ABC 块中的 `K:C` 等调号是否与谱面 `1=C` 一致。

## 12. 新版本 App 适配入口

如果 App 更新导致偏移失效：

1. 用 jadx 确认 API 字符串仍存在：`https://s.sitongli.net`、`/scores/:key`、`/scores`、`/scores/search/count`。
2. 用 blutter 重新生成 `work/blutter_out/blutter_frida.js` 和函数列表。
3. 搜索旧目标语义：`note_slur`、`note_render`、`view_get_jians`、`jianzi_component`、`score_data_decode`、`score_dataops`、`score_patch`。
4. 更新 `frida_decode_score_jians.py` 的 `TARGETS` 偏移。
5. 先只 hook `note_slur`，确认 `x1` 能解码 note，`x7.off_8` 有事件数组。
6. 再恢复 `view_get_jians` 和 `jianzi_component` 辅助目标。

## 13. 最小复现命令模板

```powershell
$root = "cases\sitongli-guanshanyue"
$scoreKey = "<score_key>"
$scoreId = <score_id>
$title = "<title>"
$slug = "<slug>"
$tonic = "<谱面 1=X 中的 X，例如 C>"

powershell -NoProfile -ExecutionPolicy Bypass `
  -File "$root\scripts\collect_score_runtime.ps1" `
  -Package "com.sitongli.app.gadget" `
  -ScoreKey $scoreKey `
  -ScoreId $scoreId `
  -Title $title `
  -Tonic $tonic `
  -Slug $slug `
  -RootDir $root `
  -LaunchDeepLink `
  -AutoTap
```

成功后查看：

```powershell
Get-Content "$root\batch\$slug\out\data.json" -Raw -Encoding UTF8
```

## 14. ABC_J 批量采集与扁平输出

本节描述针对 `ABC_J/seeds/gupu.txt`（90 曲古谱清单）的**批量采集任务**，与第
1–13 节的单曲 runtime 流程并存。批处理脚本统一放在 `ABC_J/` 下，每曲产物集中在
`ABC_J/final/<score_key>/`，不再嵌套 `results/scores/<key>/out/extracted_data/`。

### 14.1 流程与脚本

三阶段入口（`ABC_J/run_gupu_batch.ps1` → `ABC_J/gupu_batch_pipeline.py`）：

```powershell
# 1. 生成搜索清单（读 ABC_J/seeds/gupu.txt -> results/queries.json）
powershell -ExecutionPolicy Bypass -File .\ABC_J\run_gupu_batch.ps1 -Stage prepare

# 2. 在已登录 App 搜索 + 排序选 top3（candidates 目录见下）
powershell -ExecutionPolicy Bypass -File .\ABC_J\run_gupu_batch.ps1 `
  -Stage rank -Candidates .\ABC_J\candidates

# 3. 连接已授权 Gadget 版 App 后批量采集正文
powershell -ExecutionPolicy Bypass -File .\ABC_J\run_gupu_batch.ps1 -Stage collect
```

候选 JSON 来源（rank 阶段的输入边界，按任意一种获取后放进
`ABC_J/candidates/<曲名>/*.json`）：

- `scripts/capture_https_plaintext.py` 抓 HTTPS 明文；
- 内存候选导出：`ABC_J/collect_app_candidates.py` +
  `ABC_J/enrich_app_candidates.py`；
- 手动导出的 App 搜索响应。

排序指标依次为 `views、favorites、likes、comments、clones、shares、notes_length`；
标题完全匹配优先，无完全匹配才接受包含匹配。每曲输出 `selected_scores.json`，
每曲取前 3 候选。

### 14.2 输出布局

每曲所有产物集中在 `ABC_J/final/<score_key>/`，最终映射产物落在该目录根：

```text
ABC_J/final/<score_key>/
  collect.log                      runtime 采集日志
  extract.log                      简谱—减字谱提取日志
  state.json                       单曲状态（complete/failed/skipped）
  manifest.json                    采集 provenance
  evidence/                        frida 日志与 runtime_note_slur.jsonl（中间产物）
  out/raw_data.json                runtime 重建的原始正文（同 §7 规范）
  out/data.json                    对齐后事件流（同 §7 规范）
  jianpu_jianzi_readable.json      最终简谱—减字谱映射（目录根，交付件）
  jianpu_jianzi_mapped.json
  jianpu_jianzi_mapped.md
```

`out/` 与 `evidence/` 子目录由 `collect_score_runtime.ps1` 内部固定布局产生
（见 §6），是采集中间产物；最终映射产物（readable / mapped）落在 `final/<key>/`
目录根，便于直接引用。批量总状态写入 `ABC_J/results/run_status.json`。

### 14.3 断点续传

`collect` 阶段默认 `--resume`。判定某曲 `already_complete` 的条件是**同时**满足：

```text
ABC_J/final/<score_key>/out/raw_data.json            存在
ABC_J/final/<score_key>/jianpu_jianzi_readable.json  存在
```

缺任一则重跑该曲。缺可靠 `tonic`（谱面 `1=X`）的候选被标为 `missing_tonic`
并跳过，不根据调弦猜调号；确信整批同调时可传 `-DefaultTonic C`。

### 14.4 候选补全（score_id / tonic）

部分候选 JSON 缺 `score_id` 或 `tonic`，会被标 `ready_for_extraction=false`
而跳过。用 `ABC_J/resolve_selected_scores.py` 在已登录 Gadget 版 App 上补全：
通过分享页拿 `score_key/share_url`，进入乐谱正文页读 `1=X` 调号，再用 frida
内存扫描补 `score_id`。补全后 `ready_for_extraction=true` 才进入 collect。

### 14.5 音高审计（pitch audit）

采集完成后，用 `scripts/audit_jianpu_jianzi_pitch.py` 把每个 readable JSON
里的简谱音高与减字谱推算音高逐音比对（容差默认 50 音分）。批量入口：

```powershell
python ABC_J\run_pitch_audit.py
```

- 遍历所有 `ABC_J/final/<score_key>/jianpu_jianzi_readable.json`；
- 汇总写入 `ABC_J/results/pitch_audit_summary.json`，并在控制台打印总表；
- 每条记录含 `total_notes / compared_notes / matched / mismatched / skipped /
  match_rate / mean_absolute_cents`，`totals` 段为跨曲加权汇总。

**简谱 do 八度锚定**：`scripts/audit_jianpu_jianzi_pitch.py` 的
`parse_tonic_midi` 用 `TONIC_DEGREE1_MIDI` 查表把每个调号的 degree 1（do）
锚定到数据实测 + `docs/PITCH_ALGORITHM_REVERSE_ENGINEERED.md` 交叉验证的
绝对 MIDI：`1=F→F4(65)`、`1=C→C4(60)`、`1=B→B♭3(58)`。关键点：丝桐里的
`1=B` 实际是 **B-flat 调**（do 在 1 弦七徽六分 = B♭3），不是 B natural；
`1=B` 与 `1=F/C` 共用同一正调定弦（C D F G A C D），调号只是首调唱名维度。
未在表中登记的调号回退到字母启发式（octave 4），遇到新调号时应按同样数据驱动
方式补表。八度点（上加点 +12、下加点 −12）语义由数据验证正确，无需改动。

**判读 audit 结果**：单曲 match_rate 偏低时，先用 `--show mismatches` 看是否
整体差一个八度（指向该曲 `tonic` 标注或采集问题，而非 audit 逻辑）。已知少数
谱（如 `StwrXbDa 醉渔唱晚` 标 1=C 但 deg1 落在 C5、`SMcokJse 龙朔操`）存在
谱面/采集层面的调号或八度异常，属数据问题，需回 App 核实。
