# 马克一号 · Mark1

已可用的 GuqinAgent 两阶段模型，基座 Qwen3.5-9B，LoRA 权重来自
`qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016`。

- `weights/`：最终适配器、分词器、训练记录及原检查点。
- `inference/`：五首曲谱当前正式结果，含输入、预测、可视化及已生成的 PDF。
- `runtime/`：归档时的运行时、解析器、可视化脚本和原训练配置。
- `manifest.json`：文件指纹和可恢复清理位置。

权重与运行时共同决定结果。梅花三弄、凤求凰的本次最终重放均为零 warning；
该结果不代表所有音乐与演奏问题均已解决。新训练模型的效果应与 Mark1 比较。

原正式推理路径保留为兼容链接。服务器基座位于
`/home/20223393ljw/models/Qwen3.5-9B`，不重复复制基座。

归档清理记录：日期版推理目录分别移至本地
`/Volumes/F/.Trashes/501/guqin_mark1_retired_itpz9n5l/`
及服务器 `~/.local/share/Trash/files/guqin_mark1_retired_ypic0q3c/`。
服务器7个未产出 safetensors 的废弃训练目录移至
`~/.local/share/Trash/files/guqin_abandoned_training_1zuf0zb4/`。
更早的完整实验权重仍保留，待实验用途确认后再清理。
