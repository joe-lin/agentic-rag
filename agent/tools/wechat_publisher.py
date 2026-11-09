"""微信公众号一键发布pro（wechat-publisher-maxpro）的 praisonai 工具适配器。

praisonai 的 Agent 没有 shell/exec 工具，无法自己跑 node/playwright，
所以把技能的 6 步 CLI 流水线（WeDraft 排版 → 渲染 Mermaid → 生成封面 →
上传图片 → 微信兼容化 → publish.py 发布）整条包进一个 Python 函数，
作为 praisonai 的 callable 工具挂上去，Agent 通过 tool-calling 触发。

技能路径（默认，可用环境变量覆盖）：
  WECHAT_SKILL_ROOT  技能根目录
  WECHAT_NODE        workbuddy 的 node 二进制（WeDraft 用）
  WECHAT_WEDRAFT     WeDraft CLI 入口 cli.mjs
  WECHAT_PY          workbuddy 的 python 环境（脚本依赖 PIL/playwright）

凭证（技能自带逻辑，不在此处处理）：
  环境变量 WECHAT_APPID / WECHAT_APPSECRET，或技能根目录放 .wechat_pub_key
  （两行：AppID\\nAppSecret）。所有子进程都 cwd=技能根目录，确保能找到凭证。
"""

import os
import re
import sys
import json
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

SKILL_ROOT = Path(os.environ.get(
    "WECHAT_SKILL_ROOT",
    "/Users/joe/.workbuddy/skills/wechat-publisher-maxpro",
)).expanduser()

NODE = os.environ.get(
    "WECHAT_NODE",
    "/Users/joe/.workbuddy/binaries/node/versions/22.22.2-6/bin/node",
)
WEDRAFT = os.environ.get(
    "WECHAT_WEDRAFT",
    "/Users/joe/.local/share/wedraft/cli.mjs",
)
PY = os.environ.get(
    "WECHAT_PY",
    "/Users/joe/.workbuddy/binaries/python/envs/default/bin/python",
)


class WeChatPublishError(RuntimeError):
    pass


def _run(cmd, step):
    """在技能根目录执行子进程，失败抛出带上下文的异常。"""
    proc = subprocess.run(
        cmd,
        cwd=str(SKILL_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise WeChatPublishError(
            f"[{step}] 失败 (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    return proc.stdout


def _extract_mermaid(md_text):
    """抽取 ```mermaid 块，返回 (仅含 mermaid 块的新 md, 是否有图)。"""
    blocks = re.findall(r"```mermaid\s*\n(.*?)```", md_text, re.DOTALL)
    md = "\n\n".join(f"```mermaid\n{b.strip()}\n```" for b in blocks)
    return md, len(blocks) > 0


def publish_wechat_oa(
    title: str,
    content: str,
    author: str = "",
    digest: str = "",
    mode: str = "draft",
) -> str:
    """
    把一篇文章一键排版并发布到微信公众号（草稿箱或群发）。

    当用户诉求是「发公众号 / 推送到公众号 / 公众号发布 / 微信图文群发 /
    存到草稿箱 / 公众号排版美化」，或提到公众号发布、草稿箱、微信公众平台时调用。
    文章（content）支持 Markdown 或纯文本，可含 ```mermaid 架构图/时序图```，
    会自动本地渲染成图片并注入正文；代码块会自动微信兼容化。

    参数:
      title:   文章标题（必填）
      content: 文章正文（Markdown/纯文本；首行非空行即标题）
      author:  作者名（可选）
      digest:  一句话摘要（可选；不填由排版阶段处理）
      mode:    "draft"（默认，存入草稿箱，所有号可用）或
               "publish"（群发，仅认证服务号，否则返回 48001 无权限）

    返回: 发布结果摘要（含 media_id / 状态）。
    """
    if mode not in ("draft", "publish"):
        return "❌ mode 只能是 draft 或 publish"

    # 前置检查：关键二进制是否存在
    for name, path in (("WeDraft node", NODE), ("WeDraft CLI", WEDRAFT), ("workbuddy python", PY)):
        if not os.path.exists(path):
            return (f"❌ 未找到 {name}：{path}\n"
                    f"   请确认 WeDraft / workbuddy 二进制已安装，"
                    f"或通过环境变量 WECHAT_NODE / WECHAT_WEDRAFT / WECHAT_PY 指定。")

    work = SKILL_ROOT / "_work"
    # 每次运行前清空：否则上次残留的 _work/render、_work/_mermaid 会让 WeDraft /
    # render_mermaid 在 mkdir 时报 EEXIST（同目录重跑必现）。_work 仅作调试留痕，可重建。
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    # 上传结果缓存跨稿累积（技能 SKILL.md 已明确提示的坑），每次运行前清掉，
    # 避免上一稿的 media_id 串入本次发布。
    (SKILL_ROOT / "scripts" / "_upload_results.json").unlink(missing_ok=True)
    try:
        # 1) 落盘文章（保留原文）
        article_md = work / "article.md"
        article_md.write_text(content, encoding="utf-8")

        # 1.5) 抽取 mermaid 块单独渲染
        diagrams_md, has_mermaid = _extract_mermaid(content)
        if has_mermaid:
            (work / "article.diagrams.md").write_text(diagrams_md, encoding="utf-8")

        # 2) WeDraft 排版：render 产出 body.html / preview.html
        render_dir = work / "render"
        # 不预先创建：交给 WeDraft 自己 mkdir（它发现目录已存在会 EEXIST 报错）
        shutil.rmtree(render_dir, ignore_errors=True)
        _run([NODE, WEDRAFT, "render",
              "--input", str(article_md),
              "--asset-root", str(work),
              "--out", str(render_dir)], "WeDraft 排版")

        html_candidates = sorted(glob.glob(str(render_dir / "*.html")))
        body_html = None
        for p in html_candidates:
            if Path(p).name == "body.html":
                body_html = p
                break
        if body_html is None and html_candidates:
            body_html = html_candidates[0]
        if body_html is None:
            return f"❌ WeDraft render 未产出 HTML，检查 {render_dir}"
        rendered_html = body_html

        # 3) 渲染 Mermaid 图（WeDraft 不渲染，必须本地预渲染）
        image_names = []
        if has_mermaid:
            mermaid_dir = work / "_mermaid"
            shutil.rmtree(mermaid_dir, ignore_errors=True)
            _run([PY, str(SKILL_ROOT / "scripts" / "render_mermaid.py"),
                  "--md", str(work / "article.diagrams.md"),
                  "--out", str(mermaid_dir)], "渲染 Mermaid")
            mermaid_pngs = sorted(glob.glob(str(mermaid_dir / "mermaid_*.png")))
            image_names = [Path(p).name for p in mermaid_pngs]

        # 4) 生成封面
        cover_png = work / "cover.png"
        _run([PY, str(SKILL_ROOT / "scripts" / "generate_cover.py"),
              "--title", title,
              "--out", str(cover_png)], "生成封面")

        # 5) 上传图片（封面 + mermaid 图），结果写 scripts/_upload_results.json
        upload_files = [str(cover_png)] + [
            str(mermaid_dir / name) for name in image_names
        ]
        _run([PY, str(SKILL_ROOT / "scripts" / "upload_image.py"), *upload_files],
             "上传图片")
        upload_results = json.loads(
            (SKILL_ROOT / "scripts" / "_upload_results.json").read_text(encoding="utf-8")
        )
        cover_media_id = upload_results.get("cover.png", {}).get("media_id", "")
        if not cover_media_id:
            return "❌ 未拿到封面 media_id，发布中止"

        # 6) 微信兼容化：代码块改写 + data:image/占位 → 托管 URL
        final_html = work / "final.html"
        fix_cmd = [
            PY, str(SKILL_ROOT / "scripts" / "fix_wechat_html.py"),
            "--input", rendered_html,
            "--out", str(final_html),
            "--upload-results", str(SKILL_ROOT / "scripts" / "_upload_results.json"),
        ]
        if image_names:
            fix_cmd += ["--image-names", *image_names]
        _run(fix_cmd, "微信兼容化")

        # 7) 发布
        if mode == "draft":
            out = _run([PY, str(SKILL_ROOT / "scripts" / "publish.py"), "draft",
                        "--title", title,
                        "--author", author,
                        "--content-file", str(final_html),
                        "--digest", digest,
                        "--thumb-media-id", cover_media_id], "发布到草稿箱")
            return f"✅ 已存入草稿箱\n{out.strip()[-500:]}"
        else:
            # 先 draft 拿 media_id，再 publish 群发
            draft_out = _run([PY, str(SKILL_ROOT / "scripts" / "publish.py"), "draft",
                              "--title", title,
                              "--author", author,
                              "--content-file", str(final_html),
                              "--digest", digest,
                              "--thumb-media-id", cover_media_id], "存草稿拿 media_id")
            m = re.search(r"media_id:\s*(\S+)", draft_out)
            if not m:
                return f"❌ 未从 draft 输出解析到 media_id：\n{draft_out[-500:]}"
            media_id = m.group(1)
            pub_out = _run([PY, str(SKILL_ROOT / "scripts" / "publish.py"), "publish",
                            "--media-id", media_id], "群发发布")
            return f"✅ 已提交群发\n{pub_out.strip()[-500:]}"
    except WeChatPublishError as e:
        return f"❌ 发布失败：{e}"
    finally:
        # 保留 _work 便于排查；如需自动清理取消下行注释
        # shutil.rmtree(work, ignore_errors=True)
        pass


if __name__ == "__main__":
    # 本地自测（不会真实发布，用 dry-run 风格需手动加）
    sample = "# 测试标题\n\n正文一段。\n\n```mermaid\nflowchart TD;A-->B;\n```\n"
    print(publish_wechat_oa("测试标题", sample, mode="draft"))
