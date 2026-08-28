# -*- coding: utf-8 -*-
"""OSD 屏幕显示 (v10.0): 置顶小窗显示 阵营判断 + 支援方向 + 当前决策。
用户: 判断蓝方还要写成屏幕上面 -> 用户直接看到 agent 判断结果。
tkinter 无边框置顶 (overrideredirect + topmost), 不遮挡太多游戏画面。
"""
import threading
import tkinter as tk

_lock = threading.Lock()
_root = None
_label = None


def _init():
    global _root, _label
    _root = tk.Tk()
    _root.overrideredirect(True)
    _root.attributes("-topmost", True)
    _root.geometry("+30+60")   # 左上角 (避开小地图)
    _label = tk.Label(_root, text="初始化...", font=("Microsoft YaHei", 14, "bold"),
                      fg="white", bg="#2222AA", padx=14, pady=8)
    _label.pack()
    # 线程守护, 主线程消息循环
    def _loop():
        try:
            _root.mainloop()
        except Exception:
            pass
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def show(camp="?", direction="", decision="", reason=""):
    """更新 OSD 文本。camp: blue/red/?; direction: 支援方向; decision/reason: 当前决策。"""
    global _label
    try:
        if _label is None:
            _init()
            return
        bg = "#2222AA" if camp == "blue" else ("#AA2222" if camp == "red" else "#555555")
        txt = f"[{camp}] {direction}"
        if decision:
            txt += f"\n{decision}: {reason}"
        with _lock:
            _label.config(text=txt, bg=bg)
    except Exception:
        pass


def close():
    global _root
    try:
        if _root is not None:
            _root.after(0, _root.destroy)
    except Exception:
        pass
