# 减字图片渲染探索（2026-10-12）

## 已验证

- 已授权Android设备通过仓库`tools/platform-tools/adb`连接，Gadget包`com.sitongli.app.gadget`版本2.3.0。只查看可访问谱面，未编辑或发布曲谱。
- 手机当前`libapp.so`的0x5fd2a8指令字节与本地AOT副本一致，但该nLa.GYi入口未捕获到此次谱面显示的转换调用；不能把它宣称为已经验证的绘制入口。
- 从Gadget进程只读识别SFNT字体缓冲，恢复一份127,244字节、199个glyph／194条cmap映射的OpenType CFF字体，名称fonteditor。包含右手、左手、徽位、弦序和技法部件，字形图集已生成。
- 原`enc.img`大小127,248字节，但它的封装／解码算法尚未还原。试探常见AES模式失败，不作为密钥或解密成功的证据。
- 自然语言基本单弦按音反向编码为`std.tp=zhyx`；使用`extract_jianpu_jianzi.decode_jianzi_glyph`回读检查含义一致。示例：大指七徽勾四弦→`a=d,b=7,d=g,e=4`。
- 复用恢复的真实字体部件，生成三张PNG：大指七徽勾四弦、名指九徽挑五弦、中指五徽剔七弦。**排版为本地原型，不是已验证的App精确排版。**

## 入口与产物

- 只读有时限探针：`scripts/probe_jianzi_renderer.py`；完成后分离，失败时有45秒以内看门狗退出保护。
- 只读字体SFNT识别：`scripts/capture_loaded_jianzi_fonts.py`，最多扫描256 MB，仅导出校验过的字体缓冲；支持TTF和OTF。
- 部件图集：`scripts/render_jianzi_font_atlas.py`。
- 自然语言编码及PNG原型：`scripts/render_jianzi_text_prototype.py`。
- 恢复字体：`work/jianzi_renderer_probe/fonts/font_1.ttf`，实际为OpenType/CFF，后缀只是探针统一输出名。
- 图集：`work/jianzi_renderer_probe/glyph_atlas.png`。
- 原型预览：`work/jianzi_renderer_probe/prototype/preview.png`；单字图片`jianzi_1.png`—`jianzi_3.png`；编码与限制记录`encoding.json`。

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/render_jianzi_text_prototype.py \
  --font work/jianzi_renderer_probe/fonts/font_1.ttf \
  --text '大指七徽勾四弦' --text '名指九徽挑五弦' --text '中指五徽剔七弦' \
  --output-dir work/jianzi_renderer_probe/prototype
```

## 尚未完成

- 复原App真实部件缩放、嵌套位置、分徽及十一至十三徽排版。
- 扩展自然语言解析：撮／泼／剌、多弦技法、泛音标记、走手及上下文省略；不得把不支持的输入默默猜成简单按音。
- 将反向码与完整PNG/SVG排版整合进评估可视化／PDF。
- 仅供用户本地授权研究使用；原App字体资产保留在工作目录，不默认公开提交或再分发。

## 本轮扩展与纠偏

- 用字体实际墨迹边界裁切、等比例缩放和居中，取代旧版 `mt` 锚点；勾、挑、剔等使用各自的弦序嵌入区域，改善部件相对位置。**区域参数仍为本地排版参数，尚非 App 几何参数的逆向结果。**
- 基本按音、显式散音、泛音、绰前缀；徽外、十一至十三徽、分徽；绰上／注下／进／退等走手均可生成原型 PNG。
- 撮／反撮／泼／剌／泼剌／弹／双弹／三弹双弦写法已加入解析与编码；支持按＋散、散＋按、双按（不同按指／不同徽位）、双散。双按通过原始槽位保留两组独立徽位，散＋按使用新版紧凑布局，生产解码器回读检查。
- 泛音复合保存完整的显式泛音语义到布局 AST，但 **尚未还原对应 App 泛音包装编码**；`app_encoding_complete=false`，不能直接把裸 `app_std` 当作完整泛音复合输入。
- 复合／走手专用字形部件目前来自图集的暂定辨认，`glyph_mapping_status=tentative_atlas_identification`；预览卡片也显示“部件映射／布局待 App 核验”。不能据此声称已完整还原 App 渲染器。
- 对省略左手位置的 `挑五弦` 等只绘制所写字段，不虚构继承状态。未支持的输入明确报错，不以普通汉字替代减字。不支持任意多弦拂／历、所有装饰后缀或任意省略上下文。
- 已运行 `scripts/test_render_jianzi_text_prototype.py`：10 项测试通过，含不同徽位双按、散＋按、分徽、徽外、语义范围、非法输入与图片无裁切检查。
- 当前示例仍只维护在 `work/jianzi_renderer_probe/prototype/`，覆盖更新原预览，不另建混淆版本。

## 改走真实排版逆向（不继续逐字手调）

本地 AOT 静态反汇编已定位到一条比先前 `nLa.GYi` 更直接的绘制链：

- `0x5fadb4`：读取节点尺寸、建立变换、调用 `0x5fb25c`，随后调用递归绘制。
- `0x5fae24`：从对象字段读取字形文本，创建使用 **Jianzi** 字体的文本样式（`0x5faeb4–0x5faebc` 引用 pool `0x13388`）；读取比例与位置，绘制文本；遍历子节点并再次调用自身（`0x5fb010`）。
- `0x5fb398`：读取节点横纵比例，与父尺寸相乘，创建子布局节点。比例规则存在 App 二进制内，不需要凭截图逐字猜。
- `0x5fb564`：节点初始化，比例字段初值为 1，子节点容器为 `List<sLa>`。
- `sLa`（CID 1322）的布局字段：`off_8` 字形字符串、`off_c` 比例值、`off_10/off_18` 两个 double、`off_20` 子节点列表。这些字段布局由读取指令及类 fbitmap 共同支持；两个 double 的精确坐标单位、最终变换组合仍待运行时验证。

以上是**静态定位证据，尚未动态捕获确认当前谱面调用**。本轮 `adb devices` 为空，无法验证。新探针 `scripts/capture_jianzi_layout.py`：只读 hook `0x5fae24`，先校验入口 16 字节；最多抓 40 棵树，每棵最多 128 节点、10 层、每节点最多 32 子节点，20 秒默认运行、45 秒上限，有退出看门狗。没有设备时不创建“成功产物”。

手机重新授权连接且 Gadget 可用后：

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python \
  scripts/capture_jianzi_layout.py --seconds 20 \
  --score-id 69348 --open-score-body
```

生成 `work/jianzi_renderer_probe/layout_capture.jsonl`。接下来的实现应依据捕获树和变换链移植统一排版，或调用原 App 渲染链导出；不再扩充手调 `cavities`。原型只作比较，不能当作精确 App 渲染器使用。
