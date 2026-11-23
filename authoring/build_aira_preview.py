"""Build the file:// action preview alongside Aira's pose atlases."""
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "assets" / "aira"
POSES = ART / "poses"
GROUPS = {
    "演出": {"maid-duster": "羽毛掸子", "maid-flying-squirrel": "挑衅飞踢"},
    "待机": dict(breathe="站立呼吸", idle_sway="站立轻摇", idle_squat="蹲着", idle_kneel="跪坐",
               idle_sit="盘坐", idle_side="侧卧", idle_prone="趴着", idle_tokens="盘坐拨珠", idle_pipeline="拨动流水灯"),
    "姿势过渡": dict(stand_squat="站立 → 蹲下", squat_kneel="蹲下 → 跪坐", squat_sit="蹲下 → 盘坐",
                   sit_side="盘坐 → 侧卧", side_prone="侧卧 → 趴下"),
    "触碰与表情": dict(poke="轻戳", tinyheart="指尖比心", hearthands="双手比心", winkpose="眨眼示意",
                   react_kneel="跪坐时被戳", react_prone="趴着时被戳", react_side="侧卧时被戳",
                   react_sit="盘坐时被戳", react_squat="蹲着时被戳", amused="偷笑", curious="好奇",
                   downcast="垂下眼睛", grateful="道谢", pout="噘嘴", proud="得意", relieved="松口气", shy="害羞", startle="吓一跳"),
    "组件互动": dict(sidehide="从侧边探头", ledgehide="从上沿探头"),
    "学习与算子": dict(chart="展示图表", fold="折纸", inspect="检查", kvpage="整理 KV 页", matrixtiles="摆矩阵分块",
                   reducepairs="成对规约", relaychip="传芯片", sketch="画草图", stack="堆叠", study="读书",
                   think="思考", tokencard="展示 token 卡", type="敲键盘"),
    "其他动作": dict(balance="保持平衡", bell="摇铃", bow="鞠躬", bubbles="吹泡泡", catch="接住", cheer="欢呼",
                   jump="跳起", lace="整鞋带", land="落地", lift="举起", nap="打盹", parasol="撑伞", peek="探头",
                   plane="扔纸飞机", polish="擦亮", pour="倒水", run="小跑", seal="盖章", sip="喝一口",
                   spin="转圈", starhug="抱星星", stretch="伸懒腰", sweep="扫地", watch="看表",
                   water="浇水", wave="挥手", whisk="搅拌", yawn="打哈欠"),
}
DIRECTIONS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")


def native_pages(source):
    # Keep the exact native clock and a common frame origin. Image strips can be scrubbed
    # on file:// without a server, file picker, cross-origin fetch or video-codec dependency.
    spans, sheets = [], []
    with Image.open(source) as clip:
        frames = []
        for index in range(clip.n_frames):
            clip.seek(index)
            pic = clip.convert("RGBA")
            spans.append(clip.info["duration"])
            frames.append(pic.resize((360, 336), Image.Resampling.LANCZOS))
            if len(frames) == 12 or index + 1 == clip.n_frames:
                atlas = Image.new("RGBA", (1440, 336 * ((len(frames) + 3) // 4)))
                for cell, frame in enumerate(frames):
                    atlas.paste(frame, (cell % 4 * 360, cell // 4 * 336))
                stream = io.BytesIO()
                atlas.save(stream, "WEBP", quality=92, method=4)
                sheets.append(base64.b64encode(stream.getvalue()).decode("ascii"))
                frames.clear()
    return {"spans": spans, "sheets": sheets}


def build():
    rows, native = [], {}
    runtime = (ART / "aira.js").read_text("utf8")
    daily = set(json.loads(re.search(r"const dailyActions=(\[[^;]+\]);", runtime)[1]))
    daily_rate = float(re.search(r"const dailyRate=([0-9.]+)", runtime)[1])
    names = {name for items in GROUPS.values() for name in items}
    aliases = {f"{name}-{direction}" for name in GROUPS["待机"] for direction in DIRECTIONS}
    files = {p.stem for p in POSES.glob("*.webp")}
    missing = (names - set(GROUPS["演出"]) | aliases) - files
    if missing:
        raise ValueError(f"Missing pose atlas: {sorted(missing)}")
    if daily - names:
        raise ValueError(f"Unknown daily actions: {daily - names}")
    for group, actions in GROUPS.items():
        for code, title in actions.items():
            rows.append([code, title, "日常随机动作" if code in daily else group])
            if group == "演出":
                native[code] = native_pages(ART / (code + ".webp"))
    for code in sorted(files - names - aliases):
        rows.append([code, code, "其他动作"])
    layout = (HERE / "aira_preview_template.html").read_text("utf8")
    for token, data in (("ACTION_ROWS", rows), ("NATIVE_MEDIA", native), ("DAILY_RATE", daily_rate)):
        layout = layout.replace("{{" + token + "}}", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (POSES / "preview.html").write_bytes(layout.encode("utf8"))
    print(f"{len(rows)} actions, {len(aliases)} gaze overlays, {len(native)} embedded performances")


if __name__ == "__main__":
    build()
