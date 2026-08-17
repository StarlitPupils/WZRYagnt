# 11 类标注复核与扩充指引（M1）

> 目标：把现有 4 类标签（已迁移为 11 类 id）复核正确，并补标缺失的 6 类
> （ally_hero / ally_minion / ally_turret / 水晶 / 野怪 / skill_effect），
> 为 11 类 YOLO 重训准备数据。

## 现状

| 类 id | 类名 | 框数 | 状态 |
|---|---|---|---|
| 0 | enemy_hero | 454 | ✅ 已标注 |
| 1 | ally_hero | 0 | ❌ 待补标 |
| 2 | enemy_minion | 622 | ⚠️ 含歧义（可能混有 ally_minion，需复核） |
| 3 | ally_minion | 0 | ❌ 待补标 |
| 4 | enemy_turret | 253 | ⚠️ 含歧义（可能混有 ally_turret，需复核） |
| 5 | ally_turret | 0 | ❌ 待补标 |
| 6/7 | enemy/ally_crystal | 0 | ❌ 待补标（水晶） |
| 8 | neutral_monster | 0 | ❌ 待补标（野怪） |
| 9 | hook_aim | 50 | ✅ 已标注 |
| 10 | skill_effect | 0 | ❌ 待补标（技能特效，钟馗钩子/大招） |

总计 1379 框（train 572 + val 120 + screenshots 687），平均 2.96 框/图。

## 工具用法

```bat
venv\Scripts\python.exe scripts\vision\annotate.py data\screenshots\zhongkui
```

按键：`c` 切换类别（11 类循环）｜鼠标拖框｜`l` 加载已有标签｜`s` 保存｜`n/p` 翻页（自动加载+自动保存增删）｜`d` 删除最后一个框｜`q` 退出

## 复核优先级（按收益排序）

0. **自动预判（推荐先跑）**：`venv\Scripts\python.exe scripts\train\camp_autolabel.py`
   - 按单位头顶血条颜色（绿=友/红=敌）自动给出 451 个歧义框的阵营建议：
     105 个高置信"改为友方(2→3/4→5)" + 弱倾向分层（CSV 明细 + 血条裁剪图集 temp/camp_check/）
   - 先不带 `--apply` 跑，抽查裁剪图集确认准确率；满意后 `--apply` 写入（自动备份 .txt.bak）
   - 玩红方时加 `--camp red`
1. **歧义清单**（`data\yolo_dataset\zhongkui\AMBIGUOUS_REVIEW.txt`，294 个文件）：
   逐张打开，把"实际是友方"的框从 id 2→3（ally_minion）、4→5（ally_turret）。
   判断依据：兵线位置（离己方塔近的是友方）、颜色（友方血条绿/敌方红，看框内单位血条颜色）。
2. **补标 ally_hero（id 1）**：训练营/人机局截图里我方英雄与 AI 队友。
3. **补标 neutral_monster（id 8）**：野区（龙/红蓝 buff/小野怪）截图。
4. **补标 skill_effect（id 10）**：钟馗 2 技能钩子飞行、1 技能、大招特效帧。
5. **水晶（id 6/7）**：推水晶时的画面。

## 建议采集场景

- 训练营：选不同英雄当对手，打满 10 分钟，截 200-300 张（脚本 `scripts/vision/capture_screenshots.py`）
- 人机 5v5：正常对局中采集（能覆盖 ally 兵线/塔/野怪）
- 推荐**半自动预标注**：`venv\Scripts\python.exe scripts\train\prelabel.py`（已有模型只能标 4 类，
  新类需手工框）

## 重训流程（标注到位后）

```bat
venv\Scripts\python.exe scripts\train\prepare_dataset.py   :: 同步 screenshots -> yolo_dataset
:: 训练（11 类）：
python -m ultralytics.train model=models/yolo/yolov8n.pt data=data/yolo_dataset/zhongkui/data.yaml epochs=100 imgsz=640 batch=8 device=0
```

注意：`data.yaml` 已更新为 11 类声明；训练前确认各类框数分布（`scripts\train\class_stats.py`）。
