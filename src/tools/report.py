"""
工作日报工具集

支持：日报生成 / 历史查询
预留：周报/月报汇总 / 自定义模板
"""

import json
from datetime import datetime, date
from typing import Optional
from pathlib import Path
from langchain_core.tools import tool
from src.utils.logger import logger
from src.utils.execution import timed_tool
from src.config.settings import settings


DAILY_REPORT_TEMPLATE = """# 工作日报

## 基本信息
- **姓名**: {name}
- **日期**: {report_date}
- **部门**: {department}

## 今日完成工作
{today_work}

## 明日工作计划
{tomorrow_plan}

## 遇到的问题
{issues}

## 备注
{notes}
"""


@tool
@timed_tool(effect="write")
def generate_daily_report(
    name: str,
    department: str,
    today_work: str,
    tomorrow_plan: str = "暂无",
    issues: str = "无",
    notes: str = "无",
    report_date: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """生成工作日报

    Args:
        name: 姓名
        department: 部门
        today_work: 今日完成工作内容
        tomorrow_plan: 明日工作计划
        issues: 遇到的问题
        notes: 备注
        report_date: 日报日期，默认为今天
        output_path: 输出文件路径，默认自动生成到 data/reports/

    Returns:
        生成的日报文件路径和内容摘要
    """
    try:
        if report_date is None:
            report_date = date.today().isoformat()

        report_content = DAILY_REPORT_TEMPLATE.format(
            name=name,
            report_date=report_date,
            department=department,
            today_work=today_work,
            tomorrow_plan=tomorrow_plan,
            issues=issues,
            notes=notes,
        )

        if output_path is None:
            output_dir = Path(settings.report_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"日报_{name}_{report_date}.md")

        Path(output_path).write_text(report_content, encoding="utf-8")
        logger.info(f"日报已生成: {output_path}")

        return json.dumps({
            "success": True,
            "file_path": output_path,
            "summary": f"已生成 {name} 的 {report_date} 工作日报",
            "content": report_content,
            "preview": report_content[:200] + "...",
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"日报生成失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
@timed_tool()
def list_recent_reports(
    name: Optional[str] = None,
    days: int = 7,
) -> str:
    """列出最近生成的工作日报文件

    Args:
        name: 按姓名筛选（可选）
        days: 最近几天，默认7天

    Returns:
        日报文件列表
    """
    try:
        report_dir = Path(settings.report_output_dir)
        if not report_dir.exists():
            return json.dumps({"success": True, "files": []}, ensure_ascii=False)

        files = list(report_dir.glob("*.md"))
        if name:
            files = [f for f in files if name in f.stem]

        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        files = files[:days]

        result = []
        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            result.append({
                "name": f.name,
                "path": str(f),
                "modified": mtime,
            })

        return json.dumps({"success": True, "files": result}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"列出日报文件失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
@timed_tool()
def get_report_by_date(name: str, report_date: str) -> str:
    """按日期和姓名查询指定日报

    Args:
        name: 姓名
        report_date: 日报日期（YYYY-MM-DD）

    Returns:
        日报内容
    """
    try:
        report_path = Path(settings.report_output_dir) / f"日报_{name}_{report_date}.md"
        if not report_path.exists():
            return json.dumps({
                "success": False,
                "error": f"未找到 {name} 在 {report_date} 的日报",
            }, ensure_ascii=False)

        content = report_path.read_text(encoding="utf-8")
        return json.dumps({
            "success": True,
            "name": name,
            "date": report_date,
            "content": content,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"查询日报失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def aggregate_weekly_report(name: str, start_date: str, end_date: str) -> str:
    """【预留】汇总周报/月报

    Args:
        name: 姓名
        start_date: 起始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）

    Returns:
        汇总报告内容
    """
    # TODO: 实现日报汇总功能
    logger.warning("周报/月报汇总功能尚未实现")
    return json.dumps({
        "success": False,
        "error": "周报/月报汇总功能尚未实现，请在 src/tools/report.py 中启用",
    }, ensure_ascii=False)
