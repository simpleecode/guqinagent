# jianpa — 在线旋律爬取 → 古琴 Agent 推理输入

从公开曲库爬取旋律（ABC / MIDI），转换成 `eval_two_stage_score.py` 可直接消费的
`agent-eval-input-2.0` 输入，喂给训好的两阶段模型（Base 定弦指位 → Guqinizer 润色）。

零第三方依赖（全部标准库）。

## 用法

```bash
# 1) 爬取（结果落在 jianpa/cache/）
python3 jianpa/jianpa.py crawl --source thesession --search "joy" --limit 3   # 爱尔兰传统曲调 ABC (CC-BY)
python3 jianpa/jianpa.py crawl --source mutopia  --search "fur elise" --limit 3  # 公有领域钢琴曲 MIDI
python3 jianpa/jianpa.py crawl --source url --url https://.../tune.abc      # 任意直链 .abc/.mid

# 2) 转换（产出在 jianpa/out/；.mid/.midi/.abc/.jp 均可）
python3 jianpa/jianpa.py convert jianpa/cache/mutopia_fur_Elise_WoO59.mid
python3 jianpa/jianpa.py convert jianpa/cache/xxx.abc --tonic 1=F --score-key MYTUNE

# 3) 推理（与既有评估完全一致；约束解码一个开关）
python3 train/scripts/eval_two_stage_score.py \
    --input jianpa/out/<score-key>_public.jsonl --score-key <score-key> --constrain-walk-hui ...
```

每次 convert 产出：
- `out/<score-key>/jianpu_jianzi_readable.json` — 谱面中间格式（可接 PDF 出谱工具）
- `out/<score-key>_public.jsonl` — 按 phrase 切好的公开输入（`input_sha256` 与既有构建器同构）

## 流行曲（华语流行等）

流行曲管线完全支持，但**没有合法的公开爬取源**——EveryonePiano 的 MIDI 在登录/VIP 墙后，
MuseScore 下载需要账号且受版权限制，GitHub 公开语料（如 250 首中文歌的
`midi_lyric_corpus`）多为九十年代老歌。三条实际可用的入口：

1. **本地 MIDI**：任何来源的 `.mid`（自己从 MuseScore/EOP/打谱软件导出）直接
   `convert xxx.mid`。卡拉OK式多轨文件会自动按旋律轨评分选轨（onset 密度 × 覆盖跨度 ×
   音区 × 单声性，实测把张宇《用心良苦》的 24 个伴奏长音纠正为 362 个旋律音）。
2. **简谱文本直录**（推荐，最快）：网上随手可得的简谱照着敲成 `.jp` 文本文件：

   ```
   T:像风一样
   1=F 4/4
   3' 3' 5 6 | 1' - 7 6 | 5 6 1'&5 | #4/2 0/2 5 2 |
   ```

   记法：音级 `1-7`；`'` 升八度、`,` 降八度；`#`/`b` 变化音；`0` 休止；
   时值 `/2` `/4` `/8` 八分十六分三十二分、`2` `4` 二分全分、`.` 附点；
   `-` 延音（可带倍数 `-2`）；`|` 小节线；`1'&5` 撮（高音为主音）。
   调号 `1=X` 按简谱原样保留；大小调式自动检测。八度记号是相对的——管线随后整体归中，
   照原调敲即可。`convert xxx.jp` 一步出推理输入。
3. **公有领域在线源**：TheSession / Mutopia（见上）。

个人研究用途；不要把绕过站点访问控制的抓取逻辑加进本工具。

## 转换管线（对 MIDI 源）

1. **标准库 SMF 解析**（format 0/1，丢鼓轨 channel 9）。
2. **声部选择**：多轨（钢琴左右手、吉他二重奏）时只保留音高中位数最高的轨 = 旋律声部。
3. **Skyline 旋律抽取**：同 onset 分组；组内最高音若被仍在发声的更高声部盖住则整组丢弃
   （左手伴奏在旋律长音下方时不混入）；组内八度以内的次高音保留为 `jianpu_alt`（撮）。
4. **Krumhansl-Schmuckler 调性检测**（时长加权）；`--tonic` 可手动指定 `1=X`（含 ♯/♭）。
5. **整体八度归中**：全曲按八度平移到中位音高 [60,72)。正调是移动 do，`1=X` 仅为标签，
   平移不改变谱面唱名。
6. **越界折叠**：归中后仍超出琴域 [48,79]（正调一弦空弦 ~ 七弦四徽区）的个别极端音按八度折回，
   保证每个目标音都在琴上，避免把天生弹不出的音符交给评估。

ABC 源（TheSession）本身是单旋律，跳过 2/3，仅做 4/5/6。

简谱时值、`－（延音）`、`0（休止）`、装饰音、小节线（不占音序）、phrase 切分
（`split_phrase_ranges`，max_sounding=16）与训练语料构建器逐字段一致；
测试用项目自身 `parse_jianpu` 验证每个 attack 行的音高往返。

## 测试

```bash
python3 jianpa/test_jianpa.py   # 12 项：SMF 往返、调性检测、ABC 解析、
                                # 运行时不变量、旋律选轨/skyline、八度归中/折叠、
                                # 简谱直录解析、撮行配对去重
```

## 数据源与礼貌爬取

- **TheSession**（`thesession.org`）：CC-BY-4.0 传统曲调 ABC。JSON API 走
  `?format=json` 查询参数（`.json` 路径后缀已废弃）；站点 WAF 拦截非浏览器 UA。
- **Mutopia**（`www.mutopiaproject.org`，注意 `www` 前缀）：公有领域古典乐 MIDI。
  搜索字段是 `searchingfor`（`search` 会被静默忽略）。
- 爬取结果缓存在 `jianpa/cache/`，重复转换不重新下载；请保持交互级请求量，仅作个人研究用。
