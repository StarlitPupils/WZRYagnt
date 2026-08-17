# -*- coding: utf-8 -*-
"""感知融合：帧 + 检测结果 + 小地图 -> GameState。"""
import time
from typing import List, Optional

from wzry.state.schema import GameState, Unit
from wzry.vision.detector import Det


def build_state(frame, dets: List[Det], phase: str, minimap: Optional[dict] = None,
                frame_id: int = 0, source: str = "live") -> GameState:
    h, w = frame.shape[:2]
    st = GameState(t=time.time(), phase=phase, frame_id=frame_id,
                   source=source, screen_size=[w, h],
                   minimap=minimap or {})
    for d in dets:
        st.units.append(Unit(
            cls=d.cls,
            screen=[d.center[0] / w, d.center[1] / h,
                    (d.xyxy[2] - d.xyxy[0]) / w, (d.xyxy[3] - d.xyxy[1]) / h],
        ))
    return st
