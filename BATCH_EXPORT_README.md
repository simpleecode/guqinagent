# 丝桐里谱面批量采集流程

## 你需要准备

1. Android 手机开启 USB 调试，并连接到电脑。
2. 手机上安装并登录 Frida Gadget 克隆版 App，包名为 `com.sitongli.app.gadget`。
3. 每首要采集的曲目必须是你账号下有权访问/编辑的谱面。
4. 每首曲目准备三项信息：
   - `score_key`，例如 `SSG54sm8`
   - `score_id`，例如 `161577`
   - `title`，例如 `关山月`
5. 电脑 Python 环境可用，并已安装 `frida` 包。
6. adb 路径默认使用：
   `C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe`

## 曲目清单

编辑：

`cases/sitongli-guanshanyue/batch_scores.csv`

格式：

```csv
score_key,score_id,title_unicode,slug
SSG54sm8,161577,\u5173\u5c71\u6708,guanshanyue
```

- `slug` 是输出目录名，可以用拼音或英文，避免中文路径显示问题。
- `title_unicode` 用 Unicode escape，避免 PowerShell 控制台或 CSV 编码污染中文。
- 一行一首曲。

## 单曲采集

先在手机 App 里打开目标曲目的谱面页或编辑页，然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File cases\sitongli-guanshanyue\scripts\export_score_from_memory.ps1 -ScoreKey SSG54sm8 -ScoreId 161577 -Title 关山月 -Slug guanshanyue
```

输出目录：

`cases/sitongli-guanshanyue/batch/guanshanyue/out`

关键文件：

- `raw_data.json`：App 原始谱面结构。
- `data.json`：标准化事件数据。
- `manifest.json`：本次采集来源、时间、候选目录。

## 批量采集

先把所有曲目写入 `batch_scores.csv`，然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File cases\sitongli-guanshanyue\scripts\export_scores_batch.ps1
```

脚本每处理一行会暂停并提示：

1. 你在手机 App 里打开这一首曲目的谱面页或编辑页。
2. 回到 PowerShell 按 Enter。
3. 脚本扫描当前 App 进程内存，提取候选 JSON，生成 `raw_data.json` 和 `data.json`。

## 输出结构

```text
cases/sitongli-guanshanyue/batch/
  guanshanyue/
    evidence/
      memory_payload_*.jsonl
      memory_json_candidates_*/
    out/
      raw_data.json
      data.json
    manifest.json
```

## 质量检查

每首曲采完后检查 `data.json` 中：

- `metadata.score_key` 是否匹配清单。
- `metadata.score_id` 是否匹配清单。
- `stats.total_events` 是否等于元数据 `notes_length_metadata`。
- `stats.anomaly_count` 是否为 `0`。
- `events[].jianzi_components` 是否保留了 `std` 组件图。

## 注意事项

- 这套流程不是 OCR；它从已登录 App 进程里提取结构化谱面数据。
- 如果某首曲打开后没有提到正确数据，重新进入该曲谱面页或编辑页，再运行单曲脚本。
- 如果 App 没启动或 Gadget 没 attach，脚本会报 `package is not running`。
- PowerShell 控制台可能显示中文乱码，但 JSON 文件按 UTF-8 写入。用支持 UTF-8 的编辑器查看。

