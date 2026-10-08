# legacy_capture — Windows 抓谱/逆向时代脚本归档

这里的脚本是 2026 年 7–8 月在 Windows 机器上对「思桐里」App 做动态抓包、
内存 dump 与逆向分析时使用的一次性工具（frida 注入、mitmproxy 抓包、adb
导航、内存扫描、密文探针等）。它们大多硬编码了当年 Windows 机器的路径
（`C:\Users\93638\...` / `C:\Users\30343\...`），在当前 macOS 环境下不可直接运行。

保留它们仅为了保留原始数据获取过程的可复现性说明；抓取到的原始谱面数据
已经落在 `ABC_J/final/`、`ABC_J/round2/`、`ABC_J/candidates/`（均被
.gitignore 忽略，仅本地保存）。当前管线（ABC 转减字谱、教师轨迹生成、
SFT 训练）不再依赖本目录的任何脚本。

逆向过程的方法学记录见 `DOCS/PITCH_ALGORITHM_REVERSE_ENGINEERED.md` 与
`AGENT_REPRODUCTION_GUIDE.md`。
