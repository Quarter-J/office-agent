"""
定时任务管理 - 基于 APScheduler

功能预留：
  - 定时发送消息（每日报备、定时提醒）
  - 定时生成日报
  - 周期任务（每日/每周重复）

【待完善】根据实际需求启用
"""

# TODO: 接入 APScheduler 实现定时任务
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.cron import CronTrigger

# scheduler = BackgroundScheduler()
# scheduler.start()

# def add_cron_task(task_id, cron_expr, func, **kwargs):
#     trigger = CronTrigger.from_crontab(cron_expr)
#     scheduler.add_job(func, trigger, id=task_id, **kwargs)

import warnings
warnings.warn("定时任务模块尚未配置，请在 src/scheduler/tasks.py 中启用")
