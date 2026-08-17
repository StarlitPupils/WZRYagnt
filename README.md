# WZRY-Agent · 王者荣耀 AI 自动化研究项目

基于 **MuMu 模拟器屏幕视觉感知** 的王者荣耀 AI 研究项目：通过视觉模型观察屏幕、做出最优决策并执行操作，目标超越职业选手水平（当前阶段：M0-M1 工程基座与感知管线）。

## 当前进度（2026-08）

**✅ 已完成（M0 + M1 技术项）**

| 模块 | 状态 | 关键数字 |
|---|---|---|
| 坐标映射 | 设备像素空间统一（采集与输入同坐标系） | 技能点击/移动实测通过 |
| 采集 | scrcpy mkv 流式 + screencap 备用 | **33ms/帧**（30fps） |
| 对局状态机 | 小地图启发式 + 滞回 + 确认制（防菜单误触发） | 稳定 |
| YOLO 检测 | 11 类实验模型（v2 数据集） | 推理 20ms，mAP50 0.511 |
| 小地图全局视野 | 自适应定位 + 圆点分类 + 结构门控 + 固定圆心跟踪 | 跟踪 62ms，合成回归完美 |
| 数据工厂 | 采集器→编码器→past-action 对齐→BCNet→BC 训练 | 全链路就绪（合成冒烟通过） |
| 动作执行器 | 任意角度移动/技能拖瞄/连招/攻击切换 | 单元测试通过 |
| 检测数据 | 367 图 / 1317 框（4 类）+ 216 个 ally 自动建议 | 泄漏已修复 |

**⏳ 待办（需人工配合）**

1. 抽查 `temp/camp_check*/contact_sheet.png` → `camp_autolabel.py --apply` 落盘阵营建议
2. 补标 4 个空类（水晶×2/野怪/skill_effect，见 `docs/ANNOTATION_GUIDE.md`）
3. 全量 11 类重训 → 真局验证 → UI 阈值标定

## 快速开始

```bat
:: 1. 启动 MuMu 模拟器 + 王者荣耀，进入训练营/对局（人工操作）
:: 2. 确认设备（自动连接 16384/7555）
adb devices

:: 3. 实时感知管线（对局中自动采集状态）
venv\Scripts\python.exe scripts\m1_live_pipeline.py

:: 4. 延迟基准
venv\Scripts\python.exe scripts\m1_bench_latency.py

:: 5. 标注：阵营自动预判（先抽查联系表，满意后 --apply）
venv\Scripts\python.exe scripts\train\camp_autolabel.py --include-hero
venv\Scripts\python.exe scripts\train\make_contact_sheet.py
venv\Scripts\python.exe scripts\train\camp_autolabel.py --include-hero --apply

:: 6. 训练（11 类）
venv\Scripts\python.exe scripts\train\prepare_dataset.py --extra-dir data\screenshots\replay
venv\Scripts\python.exe scripts\train\train_11class.py --epochs 120 --data zhongkui --allow-empty
```

## 目录结构

```
wzry/                 主包
  capture/            window(mss) / adb(screencap) / scrcpy_stream(30fps)
  control/            executor(设备像素触摸, 事件流)
  state/              schema(GameState) / match_state(对局状态机) / fuser(融合)
  vision/             detector(YOLO) / minimap(小地图) / minimap_tracker / ui_reader
  action/             executor_v2(动作原语: 移动/技能/连招)
  train/              encoding(BC编码, past-action对齐)
  policy/             model(BCNet)
  data/               collector(MatchRecorder 会话采集)
scripts/
  m0_*                基座/校准/验证
  m1_*                感知管线/基准/小地图验证
  train/              标注/迁移/训练工具链
docs/                 PLAN.md(完整计划) / ANNOTATION_GUIDE.md(标注指引)
configs/              校准/类别/动作配置
data/                 (gitignore) 截图/标签/录像/会话/数据集
runs/                 (gitignore) 训练产物
```

## 技术路线

感知-决策分离：结构化 GameState（单位/小地图栅格/UI）→ 小策略网络（BC 后 RL），
参照 AlphaStar（像素输入）与绝悟（游戏状态）路线；详见 `docs/PLAN.md`。

## 环境

- Windows + MuMu 模拟器（ADB 端口 16384/7555，实例 0）
- Python 3.10 venv（torch 2.7+cu118 / ultralytics 8.4 / opencv / mss / paddleocr）
- RTX 4060 Laptop 8GB
