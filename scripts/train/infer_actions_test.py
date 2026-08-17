# -*- coding: utf-8 -*-
"""合成帧测试：验证动作反推器 v2 各通道并报告准确率。

帧序列（640x360，fps=10）：
  seq1_move.mp4  摇杆箭头 none -> right_up -> left -> right（通道 A）
  seq2_skill.mp4 按钮高亮 skill1 -> skill2 -> attack（通道 B）
  seq3_aim.mp4   skill2 高亮 + 瞄准线（通道 D 拖瞄）
  seq4_c.mp4     通道 C（mock 检测器）：箭头+英雄位移一致 / 冲突 / skill_effect 证据

输出：
  - 各通道准确率（A 逐帧方向正确率、B 按下事件精确率/召回率、D 逐帧命中率、C 融合置信度）
  - temp/action_infer_test/result_summary.json + 控制台表格
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.train.action_infer import infer_actions, split_result  # noqa: E402

OUT = ROOT / "temp" / "action_infer_test"
OUT.mkdir(parents=True, exist_ok=True)

W, H, FPS = 640, 360, 10.0

CALIB = {
    "video_resolution": [W, H],
    "points": {
        "joystick_center": [160, 300],
        "joystick_arrow_tip": [200, 255],
        "skill1": [420, 300], "skill2": [470, 240], "skill3": [540, 210],
        "attack": [550, 300], "recall": [300, 320],
        "restore": [330, 320], "summoner": [360, 320],
    },
}
CALIB_PATH = OUT / "synthetic.calibration.json"
CALIB_PATH.write_text(json.dumps(CALIB, indent=1), encoding="utf-8")

JC = tuple(CALIB["points"]["joystick_center"])   # 摇杆中心
SK1, SK2, ATK = (420, 300), (470, 240), (550, 300)


# ---------------------------------------------------------------- 合成绘制


def bg_frame():
    return np.full((H, W, 3), (32, 40, 28), np.uint8)


def draw_joystick(frame):
    cv2.circle(frame, JC, 22, (150, 150, 150), 2)
    cv2.circle(frame, JC, 8, (120, 120, 120), -1)


def draw_arrow(frame, vec, length=55, width=12):
    v = np.array(vec, float)
    v = v / np.linalg.norm(v)
    perp = np.array([-v[1], v[0]])
    tip = np.array(JC, float) + v * length
    b1 = np.array(JC, float) + v * (0.55 * length) + perp * width
    b2 = np.array(JC, float) + v * (0.55 * length) - perp * width
    cv2.fillConvexPoly(frame, np.array([tip, b1, b2], np.int32), (255, 255, 255))


def draw_button(frame, pos, bright):
    if bright:
        cv2.rectangle(frame, (pos[0] - 23, pos[1] - 23),
                      (pos[0] + 23, pos[1] + 23), (235, 235, 235), -1)
        cv2.circle(frame, pos, 22, (252, 252, 252), -1)
    else:
        cv2.circle(frame, pos, 20, (70, 70, 90), -1)


def draw_aim_line(frame, start, up_len=120, thick=5):
    cv2.line(frame, start, (start[0], start[1] - up_len), (255, 255, 255), thick)


def write_video(name, n_frames, frame_fn):
    path = OUT / name
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n_frames):
        f = bg_frame()
        frame_fn(f, i)
        vw.write(f)
    vw.release()
    return path


# ---------------------------------------------------------------- 帧序列


def seq1_move():
    arrows = {  # 帧区间 -> 方向向量（屏幕坐标）
        (10, 30): (0.7071, -0.7071),   # right_up
        (30, 50): (-1.0, 0.0),         # left
        (50, 70): (1.0, 0.0),          # right
    }
    def fn(f, i):
        draw_joystick(f)
        for (a, b), vec in arrows.items():
            if a <= i < b:
                draw_arrow(f, vec)
    return write_video("seq1_move.mp4", 70, fn), arrows


def seq2_skill():
    brights = {(10, 15): SK1, (25, 30): SK2, (35, 40): ATK}
    def fn(f, i):
        for (a, b), pos in brights.items():
            draw_button(f, pos, a <= i < b)
    return write_video("seq2_skill.mp4", 70, fn), brights


def seq3_aim():
    def fn(f, i):
        draw_button(f, SK2, 20 <= i < 35)
        if 20 <= i < 35:
            draw_aim_line(f, SK2)
    return write_video("seq3_aim.mp4", 50, fn)


class MockDet:
    """模拟 YOLO：ally_hero 持续向右上位移 + 特定帧 skill_effect。"""
    def __init__(self, hero_start=(300, 250), step=(8, -5)):
        self.hero = list(hero_start)
        self.step = step
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        out = []
        if 6 <= self.calls <= 30:          # 帧 5..29
            x, y = self.hero
            out.append(("ally_hero", 0.92, (x - 25, y - 25, x + 25, y + 25)))
            self.hero[0] += self.step[0]
            self.hero[1] += self.step[1]
        if 31 <= self.calls <= 35:         # 帧 30..34
            out.append(("skill_effect", 0.85, (200, 150, 320, 220)))
        return out


def seq4_c():
    arrows = {(5, 20): (0.7071, -0.7071), (20, 30): (-1.0, 0.0)}
    def fn(f, i):
        draw_joystick(f)
        for (a, b), vec in arrows.items():
            if a <= i < b:
                draw_arrow(f, vec)
        draw_button(f, SK1, 30 <= i < 35)
    return write_video("seq4_c.mp4", 40, fn), arrows


# ---------------------------------------------------------------- 指标


def accuracy_A(events, arrows, n_frames):
    """逐帧：箭头方向 vs 注入方向。"""
    injected = {}
    for (a, b), vec in arrows.items():
        for i in range(a, b):
            injected[i] = _vec_label(vec)
    det = {e["frame"]: e["direction"] for e in events if e["ch"] == "A"}
    hits = sum(1 for f, d in injected.items() if det.get(f) == d)
    return hits / len(injected), len(injected)


def _vec_label(vec):
    from wzry.train.action_infer import vec_to_dir
    return vec_to_dir(vec[0], vec[1])


def accuracy_B(events, brights, n_frames):
    """按下事件级：每个亮起区间应恰好产生一次对应按钮的 press。"""
    expected = []
    for (a, b), pos in brights.items():
        btn = {SK1: "skill1", SK2: "skill2", ATK: "attack"}[pos]
        expected.append((btn, a / FPS, b / FPS))
    presses = [(e["button"], e["t"]) for e in events if e["ch"] == "B"]
    matched = 0
    used = set()
    for btn, t0, t1 in expected:
        hit = [i for i, (b_, t_) in enumerate(presses)
               if b_ == btn and t0 <= t_ <= t1 and i not in used]
        if hit:
            matched += 1
            used.add(hit[0])
    prec = matched / max(len(presses), 1)
    rec = matched / max(len(expected), 1)
    return prec, rec, expected, presses


def accuracy_D(events, n_frames):
    """逐帧：aim 事件覆盖帧 vs 注入（seq3: 帧 20..34）。"""
    det_frames = {e["frame"] for e in events if e["ch"] == "D"}
    inj = set(range(20, 35))
    hits = len(inj & det_frames)
    return hits / len(inj), len(inj)


# ---------------------------------------------------------------- 主流程


def find_actions(actions, **kw):
    return [a for a in actions if all(a.get(k) == v for k, v in kw.items())]


def run():
    results = {}

    # -- 序列 1：通道 A ------------------------------------------------
    path1, arrows1 = seq1_move()
    res1 = infer_actions(str(path1), str(CALIB_PATH), sample_every=1, model_path=None)
    acts1, meta1 = split_result(res1)
    evs1 = meta1["events"]
    acc_a, n_a = accuracy_A(evs1, arrows1, 70)
    d_ru = find_actions(acts1, type="move", direction="right_up")
    d_l = find_actions(acts1, type="move", direction="left")
    d_r = find_actions(acts1, type="move", direction="right")
    t_ru = [a["t"] for a in d_ru]
    t_l = [a["t"] for a in d_l]
    t_r = [a["t"] for a in d_r]
    ok_ru = any(1.0 <= t <= 3.0 for t in t_ru)
    ok_l = any(3.0 <= t <= 5.0 for t in t_l)
    ok_r = any(5.0 <= t <= 7.0 for t in t_r)
    conf_ok = all(a["confidence"] >= 0.5 for a in d_ru + d_l + d_r)
    ch_ok = all("A" in a["channels"] for a in d_ru + d_l + d_r)
    seq1_ok = acc_a >= 0.95 and ok_ru and ok_l and ok_r and conf_ok and ch_ok
    results["seq1_move"] = {
        "ok": seq1_ok,
        "channel_A_accuracy": round(acc_a, 3),
        "A_frames": n_a,
        "detected_right_up_frames": len(t_ru),
        "detected_left_frames": len(t_l),
        "detected_right_frames": len(t_r),
        "right_up_present": ok_ru, "left_present": ok_l, "right_present": ok_r,
        "confidence_ok": conf_ok, "channels_ok": ch_ok,
    }
    print(f"[seq1] 通道A准确率 {acc_a:.2%}（{n_a} 帧） right_up={ok_ru} left={ok_l} "
          f"right={ok_r} conf_ok={conf_ok} ch_ok={ch_ok} -> {'PASS' if seq1_ok else 'FAIL'}")

    # -- 序列 2：通道 B ------------------------------------------------
    path2, brights2 = seq2_skill()
    res2 = infer_actions(str(path2), str(CALIB_PATH), sample_every=1, model_path=None)
    acts2, meta2 = split_result(res2)
    evs2 = meta2["events"]
    prec_b, rec_b, exp_b, presses_b = accuracy_B(evs2, brights2, 70)
    s1 = find_actions(acts2, type="skill", skill_id=1)
    s2 = find_actions(acts2, type="skill", skill_id=2)
    atk = find_actions(acts2, type="attack")
    s1_ok = any(0.9 <= a["t"] <= 1.6 and a["confidence"] >= 0.5
                and "B" in a["channels"] for a in s1)
    s2_ok = any(2.4 <= a["t"] <= 3.1 for a in s2)
    atk_ok = any(3.4 <= a["t"] <= 4.1 for a in atk)
    seq2_ok = prec_b == 1.0 and rec_b == 1.0 and s1_ok and s2_ok and atk_ok
    results["seq2_skill"] = {
        "ok": seq2_ok,
        "B_press_precision": prec_b, "B_press_recall": rec_b,
        "expected": [f"{b}@{t0:.1f}-{t1:.1f}" for b, t0, t1 in exp_b],
        "detected": [f"{b}@{t:.2f}" for b, t in presses_b],
        "skill1_ok": s1_ok, "skill2_ok": s2_ok, "attack_ok": atk_ok,
    }
    print(f"[seq2] 通道B 精确率 {prec_b:.0%} 召回率 {rec_b:.0%} "
          f"skill1={s1_ok} skill2={s2_ok} attack={atk_ok} -> {'PASS' if seq2_ok else 'FAIL'}")

    # -- 序列 3：通道 D ------------------------------------------------
    path3 = seq3_aim()
    res3 = infer_actions(str(path3), str(CALIB_PATH), sample_every=1, model_path=None)
    acts3, meta3 = split_result(res3)
    evs3 = meta3["events"]
    acc_d, n_d = accuracy_D(evs3, 50)
    s2_aim = [a for a in acts3 if a["type"] == "skill" and a.get("skill_id") == 2
              and "aim" in a["flags"]]
    d_aim_ok = any(1.9 <= a["t"] <= 2.6 for a in s2_aim)
    seq3_ok = acc_d >= 0.9 and d_aim_ok
    results["seq3_aim"] = {
        "ok": seq3_ok,
        "channel_D_accuracy": round(acc_d, 3),
        "D_frames": n_d,
        "aim_skill2_action": d_aim_ok,
    }
    print(f"[seq3] 通道D 命中率 {acc_d:.2%}（{n_d} 帧） aim_skill2={d_aim_ok} "
          f"-> {'PASS' if seq3_ok else 'FAIL'}")

    # -- 序列 4：通道 C（mock 检测器）-----------------------------------
    path4, arrows4 = seq4_c()
    mock = MockDet()
    res4 = infer_actions(str(path4), str(CALIB_PATH), sample_every=1,
                         model_path=None, mock_detector=mock)
    acts4, meta4 = split_result(res4)
    evs4 = meta4["events"]
    agree = [a for a in acts4 if a["type"] == "move" and a.get("direction") == "right_up"
             and set(a["channels"]) == {"A", "C"}]
    agree_ok = any(0.5 <= a["t"] <= 2.0 and a["confidence"] >= 0.7 for a in agree)
    conflict = [a for a in acts4 if a["type"] == "move"
                and a.get("direction") == "left" and "A_C_conflict" in a["flags"]]
    conflict_ok = any(2.0 <= a["t"] <= 3.0 and a["confidence"] <= 0.45 for a in conflict)
    sk = [a for a in acts4 if a["type"] == "skill" and a.get("skill_id") == 1]
    sk_ok = any(2.9 <= a["t"] <= 3.5 and a["confidence"] >= 0.85
                and set(a["channels"]) == {"B", "C"} for a in sk)
    c_moves = [e for e in evs4 if e["ch"] == "C" and e["kind"] == "hero_move"]
    c_se = [e for e in evs4 if e["ch"] == "C" and e["kind"] == "skill_effect"]
    hero_dir_correct = sum(1 for e in c_moves if e["direction"] == "right_up")
    c_acc = hero_dir_correct / max(len(c_moves), 1)
    seq4_ok = agree_ok and conflict_ok and sk_ok and c_acc >= 0.9 and len(c_se) >= 3
    results["seq4_c"] = {
        "ok": seq4_ok,
        "C_hero_move_direction_accuracy": round(c_acc, 3),
        "C_hero_move_events": len(c_moves),
        "C_skill_effect_events": len(c_se),
        "A_C_agree_high_conf": agree_ok,
        "A_C_conflict_low_conf": conflict_ok,
        "B_C_skill_high_conf": sk_ok,
    }
    print(f"[seq4] 通道C mock 位移方向准确率 {c_acc:.2%}（{len(c_moves)} 事件） "
          f"一致高置信={agree_ok} 冲突低置信={conflict_ok} B+C技能高置信={sk_ok} "
          f"-> {'PASS' if seq4_ok else 'FAIL'}")

    all_ok = all(r["ok"] for r in results.values())
    summary = {"all_pass": all_ok, "cases": results}
    (OUT / "result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'='*60}\n总结果: {'全部通过 PASS' if all_ok else '存在失败 FAIL'}"
          f"\n结果保存: {OUT / 'result_summary.json'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
