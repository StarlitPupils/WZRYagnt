# -*- coding: utf-8 -*-
"""OSD 屏幕显示 (v10.2 健壮版): 置顶小窗显示 阵营判断 + 支援方向 + 当前决策。
tkinter 无边框置顶; 线程安全 (首次 show 即 init 并显示)。
"""
import threading
import tkinter as tk

_lock = threading.Lock()
_state = {"root": None, "label": None, "started": False}


def _ensure():
    with _lock:
        if _state["started"]:
            return True
        try:
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.geometry("+30+60")
            lbl = tk.Label(root, text="初始化...", font=("Microsoft YaHei", 13, "bold"),
                           fg="white", bg="#2222AA", padx=12, pady=6, justify="left")
            lbl.pack()
            root.deiconify()
            _state["root"] = root
            _state["label"] = lbl
            _state["started"] = True

            def _loop():
                try:
                    root.mainloop()
                except Exception:
                    pass
            t = threading.Thread(target=_loop, daemon=True)
            t.start()
            return True
        except Exception:
            return False


def show(camp="?", direction="", decision="", reason=""):
    """更新 OSD。camp: blue/red/?; direction: 支援方向; decision/reason 当前决策。"""
    try:
        if not _state["started"] and not _ensure():
            return
        lbl = _state["label"]
        if lbl is None:
            return
        bg = "#1A48B8" if camp == "blue" else ("#C22" if camp == "red" else "#555")
        emoji = "[蓝方]" if camp == "blue" else ("[红方]" if camp == "red" else "[判定中]")
        txt = f"{emoji} {direction}"
        if decision:
            txt += f"\n{decision}: {reason}"
        lbl.config(text=txt, bg=bg)
    except Exception:
        pass
